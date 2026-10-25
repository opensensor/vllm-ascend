# SPDX-License-Identifier: Apache-2.0
"""Compile and seal a versioned direct CANN W4 kernel without opening a device."""

import argparse
import json
import shutil
import subprocess
import tarfile
from pathlib import Path

from tools.glm_perf.build_query_bf16_vector import compile_bridge
from tools.glm_perf.resident_native import file_digest


def build(
    output,
    source_root,
    namespace,
    cann,
    *,
    singleton_l1=False,
    l1_max_k=4096,
    wide_l1=False,
    scale_cache=False,
    rint_unpack=False,
    expert_lanes=0,
    int8_compute=False,
    byte_unpack=False,
    int8_original_layout=False,
    paired_gate_up=False,
    source_nvfp4=False,
):
    if not namespace.isidentifier() or not namespace.startswith("minimax_w4_"):
        raise ValueError("use a unique minimax_w4_ namespace for each build")
    if source_nvfp4 and (
        not wide_l1 or scale_cache or int8_compute or singleton_l1 or expert_lanes or byte_unpack or paired_gate_up
    ):
        raise ValueError("source NVFP4 requires wide L1 and excludes W4-specific compute/scale options")
    if l1_max_k not in (4096, 6144) or (l1_max_k != 4096 and not singleton_l1):
        raise ValueError("L1 max K must be 4096 or 6144, with singleton L1 enabled for 6144")
    if singleton_l1 and wide_l1:
        raise ValueError("choose singleton L1 or wide L1, not both")
    if scale_cache and not wide_l1:
        raise ValueError("scale cache requires wide L1")
    if expert_lanes not in (0, 1, 2, 4, 8) or (expert_lanes and not wide_l1):
        raise ValueError("expert lanes must be 0, 1, 2, 4 or 8 and require wide L1")
    if int8_original_layout and not int8_compute:
        raise ValueError("INT8 original layout requires INT8 compute")
    if paired_gate_up and (not int8_compute or int8_original_layout):
        raise ValueError("paired gate/up requires repacked INT8 compute")
    if byte_unpack and (rint_unpack or int8_compute):
        raise ValueError("byte unpack is a separate FP16 candidate")
    if int8_compute and (singleton_l1 or expert_lanes or not wide_l1 or not rint_unpack):
        raise ValueError("INT8 prototype requires wide L1 and RINT, with optional scale cache only")
    output, source_root, cann = output.resolve(), source_root.resolve(strict=True), cann.resolve(strict=True)
    output.mkdir(parents=True, exist_ok=False)
    sources = output / "sources"
    sources.mkdir()
    for directory in (
        "gmm/w2_blocked_dequant_matmul_v310/op_kernel",
        "gmm/w2_grouped_blocked_dequant_matmul_v310/op_kernel",
        "third_party/catlass/include",
        "moe/common/kernel_utils",
        "common/include",
    ):
        shutil.copytree(source_root / directory, sources / directory)
    root = Path(__file__).resolve().parents[1]
    shutil.copy2(root / "glm_perf/compile_reconstruction.cpp", sources / "compiler.cpp")
    shutil.copy2(root / "glm_perf/channel_scale_bridge.cpp", output / "reconstruction_bridge.cpp")
    compiler = output / "compile-kernel"
    subprocess.run(
        [
            "c++",
            "-std=c++17",
            "-O2",
            str(sources / "compiler.cpp"),
            f"-I{cann / 'include'}",
            f"-L{cann / 'lib64'}",
            "-lacl_rtc",
            "-lascendcl",
            f"-Wl,-rpath,{cann / 'lib64'}",
            "-o",
            str(compiler),
        ],
        check=True,
    )
    grouped = sources / "gmm/w2_grouped_blocked_dequant_matmul_v310/op_kernel"
    blocked = sources / "gmm/w2_blocked_dequant_matmul_v310/op_kernel"
    if l1_max_k != 4096:
        header = blocked / "w2_blocked_dequant_matmul_v310.h"
        original = "constexpr uint32_t W2_L1_MAX_K = 4096;"
        contents = header.read_text()
        if contents.count(original) != 1:
            raise ValueError("grouped L1 workspace contract changed")
        header.write_text(contents.replace(original, f"constexpr uint32_t W2_L1_MAX_K = {l1_max_k};"))
    kernel_file = "w4a8_grouped.cpp" if int8_compute else "w2_grouped_blocked_dequant_matmul_v310.cpp"
    entry = namespace + "_kernel"
    binary = output / "grouped_w4.bin"
    target = cann / "tools/hcc/aarch64-target-linux-gnu/include/c++/7.3.0"
    compile_options = [
        "--npu-arch=dav-2002",
        f"--sysroot={cann / 'tools/hcc/sysroot'}",
        f"-isystem{target}",
        f"-isystem{target / 'aarch64-target-linux-gnu'}",
        f"-I{grouped}",
        f"-I{blocked}",
        f"-I{sources / 'third_party/catlass/include'}",
        f"-I{sources / 'moe/common'}",
        f"-include{sources / 'common/include/cann_compat.h'}",
        "-Wno-ignored-attributes",
        *(["-DGLM_W2_GROUPED_CANDIDATE"] if singleton_l1 else []),
        *(["-DGLM_W2_GROUPED_L1_WIDE", "-DGLM_W2_GROUPED_L1_WIDE_PIPELINED"] if wide_l1 else []),
        *(["-DGLM_W2_GROUPED_L1_SCALE_CACHE"] if scale_cache else []),
        *(["-DGLM_W2_GROUPED_RINT_UNPACK"] if rint_unpack else []),
        *(["-DGLM_W4_BYTE_UNPACK"] if byte_unpack else []),
        *(["-DMINIMAX_INT8_ORIGINAL_LAYOUT"] if int8_original_layout else []),
        *(["-DMINIMAX_SOURCE_NVFP4"] if source_nvfp4 else []),
        *([f"-DGLM_W2_GROUPED_EXPERT_LANES={expert_lanes}"] if expert_lanes else []),
    ]
    kernels = [(kernel_file, entry, binary)]
    paired_binary, paired_entry = output / "gate_up_w4.bin", namespace + "_gate_up_kernel"
    if paired_gate_up:
        kernels.append(("w4a8_gate_up.cpp", paired_entry, paired_binary))
    for filename, symbol, destination in kernels:
        kernel_source = (grouped / filename).read_text()
        # A direct binary owns its workspace and cannot depend on OPP state.
        old = "GM_ADDR user = AscendC::GetUserWorkspace(workspace);"
        if kernel_source.count(old) != 1:
            raise ValueError("grouped kernel workspace contract changed")
        kernel_source = kernel_source.replace(old, "GM_ADDR user = workspace;")
        kernel_source = kernel_source.replace("KERNEL_TYPE_MIX_AIC", "KERNEL_TYPE_AICORE")
        kernel_source = kernel_source.replace("void w2_grouped_blocked_dequant_matmul_v310(", f"void {symbol}(")
        source = sources / (symbol + ".cpp")
        source.write_text(kernel_source)
        subprocess.run([str(compiler), str(source), str(destination), *compile_options], check=True)
    compile_bridge(output / "bridge.so", namespace, cann)
    with tarfile.open(output / "sources.tar.gz", "w:gz") as archive:
        archive.add(sources, arcname="sources")
    helper = output / "resident_native.py"
    helper_name = "nvfp4_native.py" if source_nvfp4 else ("int8_native.py" if int8_compute else "resident_native.py")
    shutil.copy2(Path(__file__).with_name(helper_name), helper)
    if int8_original_layout:
        helper.write_text(helper.read_text().replace("requires_int8_layout = True", "requires_int8_layout = False"))
    (output / "build-options.json").write_text(
        json.dumps(
            {
                "singleton_l1": singleton_l1,
                "l1_max_k": l1_max_k,
                "wide_l1": wide_l1,
                "scale_cache": scale_cache,
                "rint_unpack": rint_unpack,
                "expert_lanes": expert_lanes,
                "int8_compute": int8_compute,
                "byte_unpack": byte_unpack,
                "int8_original_layout": int8_original_layout,
                "paired_gate_up": paired_gate_up,
                "source_nvfp4": source_nvfp4,
            },
            indent=2,
        )
        + "\n"
    )
    paired_args = f", paired_binary={str(paired_binary)!r}, paired_entry={paired_entry!r}" if paired_gate_up else ""
    validation = (
        "import types\n"
        f"_module = types.ModuleType({namespace!r})\n"
        f"exec(compile(open({str(helper)!r}).read(), {str(helper)!r}, 'exec'), _module.__dict__)\n"
        f"def prepare():\n    return _module.GroupedW4Kernel({namespace!r}, {str(binary)!r}, {entry!r}{paired_args})\n"
        "def validate(resource):\n    return _module.validate_kernel(resource)\n"
    )

    def record(path):
        return {"path": str(path), "sha256": file_digest(path)}

    manifest = {
        "name": namespace,
        "libraries": [record(output / "bridge.so")],
        "assets": [
            record(output / name)
            for name in (
                "grouped_w4.bin",
                "sources.tar.gz",
                "resident_native.py",
                "reconstruction_bridge.cpp",
                "build-options.json",
                *(["gate_up_w4.bin"] if paired_gate_up else []),
            )
        ],
        "operators": [namespace + "::launch"],
        "validation_source": validation,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return output / "manifest.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--source-root", type=Path, default=Path(__file__).resolve().parents[2] / "csrc")
    parser.add_argument("--namespace", required=True)
    parser.add_argument("--cann", type=Path, default=Path("/usr/local/Ascend/ascend-toolkit/latest"))
    parser.add_argument("--singleton-l1", action="store_true", help="Keep single-route dequantized tiles in L1")
    parser.add_argument("--l1-max-k", type=int, choices=(4096, 6144), default=4096)
    parser.add_argument("--wide-l1", action="store_true", help="Pipeline 128x1024 L1 tiles for small expert groups")
    parser.add_argument("--scale-cache", action="store_true", help="Reuse scale rows across wide-L1 K tiles")
    parser.add_argument("--rint-unpack", action="store_true", help="Use exact biased-RINT W4 field extraction")
    parser.add_argument(
        "--expert-lanes",
        type=int,
        default=0,
        choices=(0, 1, 2, 4, 8),
        help="Cores per expert team for prefill; 0 retains all-core scheduling",
    )
    parser.add_argument("--int8-compute", action="store_true", help="Experimental packed W4 / INT8 Cube kernel")
    parser.add_argument("--byte-unpack", action="store_true", help="Signed byte-pair W4 extraction, FP16 compute")
    parser.add_argument("--int8-original-layout", action="store_true", help="Retain existing resident W4 byte order")
    parser.add_argument("--paired-gate-up", action="store_true", help="Build opt-in compact decode gate/up fusion")
    parser.add_argument("--source-nvfp4", action="store_true", help="Preserve source FP4 codes and per-row K16 scales")
    args = parser.parse_args()
    print(
        build(
            args.output,
            args.source_root,
            args.namespace,
            args.cann,
            singleton_l1=args.singleton_l1,
            l1_max_k=args.l1_max_k,
            wide_l1=args.wide_l1,
            scale_cache=args.scale_cache,
            rint_unpack=args.rint_unpack,
            expert_lanes=args.expert_lanes,
            int8_compute=args.int8_compute,
            byte_unpack=args.byte_unpack,
            int8_original_layout=args.int8_original_layout,
            paired_gate_up=args.paired_gate_up,
            source_nvfp4=args.source_nvfp4,
        )
    )


if __name__ == "__main__":
    main()
