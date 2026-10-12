# SPDX-License-Identifier: Apache-2.0
"""Rebuild only routed SwiGLU stages from authenticated frozen source bytes."""

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

from .native_expert_probe import digest_file

SWIGLU_LIMIT = 10.0
NATIVE_INSERTION = "    Muls(accum, low, -1.0f, activeElements);"
NATIVE_CLIP = """    Mins(low, low, 10.0f, activeElements);
    PipeBarrier<PIPE_V>();
    Mins(high, high, 10.0f, activeElements);
    PipeBarrier<PIPE_V>();
    Maxs(high, high, -10.0f, activeElements);
    PipeBarrier<PIPE_V>();
"""
GATE_OPTIONS = (
    "streaming_pipeline",
    "prepared_weight_layout",
    "prerounded_weight_scales",
    "fp16_weight_scales",
    "pair_scale_groups",
    "pair_hidden_quant",
    "strided_product_copy",
    "repeat_product_cast",
    "prefill_weight_cache",
    "vector_scale_products",
    "group_major_down_scales",
    "prefill_rows_32",
    "active_cube_rows",
    "direct_w4_l1",
    "prepared_offset_tables",
    "product_pipe_events",
    "fused_scale_accumulation",
    "gather_product_matrix",
    "fp16_route_workspace",
    "weight_decode_lut",
    "pair_prefill_scale_groups",
    "route_packed_input",
    "group_major_input_scales",
    "route_packed_down",
    "route_compact_down_scales",
    "raw_hidden_scales",
    "nz_prefill_accumulator",
    "cache_expert_ends",
    "prefill_product_cast",
    "direct_hidden_gather",
    "share_gate_up_input",
    "cache_gate_up_activations",
)


def clip_native(source):
    if source.count(NATIVE_INSERTION) != 1:
        raise ValueError("native SwiGLU insertion differs from audited frozen source")
    return source.replace(NATIVE_INSERTION, NATIVE_CLIP + NATIVE_INSERTION)


def clip_cuda(source):
    for name in ("gate", "up"):
        original = f"  float {name} = __half2float(gate_up[row * 2 * intermediate + "
        lines = [line for line in source.splitlines() if line.startswith(original)]
        if len(lines) != 1:
            raise ValueError("CUDA SwiGLU insertion differs from audited source")
        expression = lines[0].split(" = ", 1)[1].rstrip(";")
        clipped = f"fminf(10.0F, {expression})" if name == "gate" else f"fminf(10.0F, fmaxf(-10.0F, {expression}))"
        source = source.replace(lines[0], f"  float {name} = {clipped};")
    return source


def build(output, bundle, native_source, cann, cuda_source):
    output.mkdir(parents=True, exist_ok=False)
    frozen = json.loads((bundle / "provenance.json").read_bytes())
    options = frozen["_build"]
    if options.get("fp16_swiglu") or not options.get("streaming_pipeline") or options.get("wide_cube_k"):
        raise ValueError("clipping override requires the audited FP32 nonlinear M32 bundle")
    native = output / "native"
    native.mkdir()
    originals = {}
    for name, checksum in frozen["_streaming_sources"].items():
        filename = Path(name).name
        if filename not in ("compile_reconstruction.cpp", "glm_fused_moe.cpp") and not filename.endswith(".h"):
            continue
        source = native_source / filename
        if digest_file(source) != checksum:
            raise ValueError(f"frozen native source authentication failed: {filename}")
        originals[filename] = checksum
        shutil.copyfile(source, native / filename)
    source = native / "glm_fused_moe.cpp"
    source.write_text(clip_native(source.read_text()))
    compiler = output / "compile-reconstruction"
    commands = [
        [
            "c++",
            "-std=c++17",
            "-O2",
            str(native / "compile_reconstruction.cpp"),
            f"-I{cann / 'include'}",
            f"-L{cann / 'lib64'}",
            "-lacl_rtc",
            "-lascendcl",
            f"-Wl,-rpath,{cann / 'lib64'}",
            "-o",
            str(compiler),
        ]
    ]
    subprocess.run(commands[-1], check=True)
    target = cann / "tools/hcc/aarch64-target-linux-gnu/include/c++/7.3.0"
    flags = [
        "--npu-arch=dav-2002",
        f"--sysroot={cann / 'tools/hcc/sysroot'}",
        f"-isystem{target}",
        f"-isystem{target / 'aarch64-target-linux-gnu'}",
        f"-I{native}",
        "-DGLM_FUSED_GATE_UP",
    ]
    for name in GATE_OPTIONS:
        macro = "COMPACT_DOWN_SCALES" if name == "route_compact_down_scales" else name.upper()
        if options.get(name):
            flags.append(f"-DGLM_{macro}")
    if options.get("quad_hidden_quant"):
        flags.extend(("-DGLM_QUAD_HIDDEN_QUANT", "-DGLM_HIDDEN_QUANT_BATCH=16"))
    records = {}
    for suffix in ("", "_w4"):
        filename = f"glm_fused_gate_up{suffix}.bin"
        command = [str(compiler), str(source), str(output / filename), *flags]
        if suffix:
            command.extend(("-DGLM_STATIC_WEIGHT_BITS=4", "-DGLM_COMPACT_W4_SCRATCH"))
        commands.append(command)
        subprocess.run(command, check=True)
        records[filename] = digest_file(output / filename)
    cuda = output / "cuda"
    cuda.mkdir()
    for name in ("cuda_oracle.cu", "cuda_compact.cu"):
        shutil.copyfile(cuda_source / name, cuda / name)
    (cuda / "cuda_oracle.cu").write_text(clip_cuda((cuda / "cuda_oracle.cu").read_text()))
    library = output / "cuda-clipped.so"
    commands.append(
        [
            "/usr/local/cuda/bin/nvcc",
            "-std=c++17",
            "-O3",
            "--fmad=false",
            "-arch=sm_120",
            "-shared",
            "-Xcompiler=-fPIC",
            str(cuda / "cuda_compact.cu"),
            "-o",
            str(library),
        ]
    )
    subprocess.run(commands[-1], check=True)
    report = {
        "swiglu_limit": SWIGLU_LIMIT,
        "original_provenance_sha256": digest_file(bundle / "provenance.json"),
        "authenticated_sources": originals,
        "native_source_sha256": digest_file(source),
        "native_gate_files": records,
        "cuda_library": str(library),
        "cuda_library_sha256": digest_file(library),
        "commands": commands,
        "hardware_validated": False,
    }
    (output / "build.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("output", "bundle", "native-source", "cann", "cuda-source"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--environment-file", type=Path)
    args = parser.parse_args()
    if args.environment_file:
        os.environ.update(json.loads(args.environment_file.read_bytes()))
    result = build(
        args.output.resolve(),
        args.bundle.resolve(),
        args.native_source.resolve(),
        args.cann.resolve(),
        args.cuda_source.resolve(),
    )
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
