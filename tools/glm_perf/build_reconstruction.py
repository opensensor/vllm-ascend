# SPDX-License-Identifier: Apache-2.0
"""Build append-only reconstruction probes; never connect to a serving engine."""

import argparse
import hashlib
import importlib.util
import json
import shutil
import subprocess
import sysconfig
from pathlib import Path

import torch
from torch.utils.cpp_extension import include_paths, library_paths

HERE = Path(__file__).resolve().parent
W3_DEFINES = (
    "GLM_W2_GROUPED_L1_WIDE",
    "GLM_W2_GROUPED_L1_WIDE_PIPELINED",
    "GLM_W2_GROUPED_L1_W3",
)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(
    build_dir,
    cann_root,
    source_root,
    include_w3=False,
    version=1,
    output_columns=16,
    tile_pipeline=False,
    all_bits=False,
    fused_moe=False,
    prepared_weight_layout=False,
    pair_scale_groups=False,
    fp16_swiglu=False,
    pair_hidden_quant=False,
    wide_cube_k=0,
    pair_prefill_scale_groups=False,
    weight_decode_lut=False,
    strided_product_copy=False,
):
    """Freeze helper sources and compile a unique, append-only native version."""
    if type(version) is not int or version < 1 or output_columns not in (16, 32, 64, 128):
        raise ValueError("version must be positive and output tile 16, 32, 64 or 128")
    if output_columns == 128 and not tile_pipeline:
        raise ValueError("128 columns requires the tile pipeline")
    if all_bits and not tile_pipeline:
        raise ValueError("all bit widths require the tile pipeline")
    if fused_moe and (not all_bits or output_columns != 128):
        raise ValueError("fused MoE requires all bits and a 128-column tile")
    if prepared_weight_layout and not fused_moe:
        raise ValueError("prepared weight layout requires fused MoE")
    if pair_scale_groups and not fused_moe:
        raise ValueError("paired scale groups require fused MoE")
    if fp16_swiglu and not fused_moe:
        raise ValueError("FP16 SwiGLU requires fused MoE")
    if pair_hidden_quant and not fused_moe:
        raise ValueError("paired hidden quantization requires fused MoE")
    if strided_product_copy and not fused_moe:
        raise ValueError("strided product copy requires fused MoE")
    if type(wide_cube_k) is not int or wide_cube_k not in (0, 128, 256):
        raise ValueError("wide Cube K must be 0 (disabled), 128 or 256")
    if (wide_cube_k or pair_prefill_scale_groups or weight_decode_lut) and not (
        fused_moe and prepared_weight_layout and pair_scale_groups
    ):
        raise ValueError("wide Cube and prefill scale pairing require prepared paired fused MoE")
    namespace = f"glm_reconstruction_v{version}"
    build_dir = build_dir.resolve()
    build_dir.mkdir(parents=True, exist_ok=False)
    helper_package = namespace + "_helpers"
    helper_root = build_dir / helper_package
    helper_root.mkdir()
    (helper_root / "__init__.py").write_text("# SPDX-License-Identifier: Apache-2.0\n")
    helpers = ("glm_int4.py", "reconstruction_native.py", "reconstruction_probe.py")
    if fused_moe:
        helpers += ("glm_fused_moe.py", "fused_moe_probe.py", "fused_weight_layout.py")
    for name in helpers:
        shutil.copy2(HERE / name, helper_root / name)
    shutil.copy2(HERE / "resident_candidates/expert_reconstruction.py", helper_root / "expert_reconstruction.py")
    cann_root = cann_root.resolve(strict=True)
    source_root = source_root.resolve(strict=True)
    compiler = build_dir / "compile-reconstruction"
    subprocess.run(
        [
            "c++",
            "-std=c++17",
            "-O2",
            str(HERE / "compile_reconstruction.cpp"),
            f"-I{cann_root / 'include'}",
            f"-L{cann_root / 'lib64'}",
            "-lacl_rtc",
            "-lascendcl",
            f"-Wl,-rpath,{cann_root / 'lib64'}",
            "-o",
            str(compiler),
        ],
        check=True,
    )
    target = cann_root / "tools/hcc/aarch64-target-linux-gnu/include/c++/7.3.0"
    options = [
        "--npu-arch=dav-2002",
        f"--sysroot={cann_root / 'tools/hcc/sysroot'}",
        f"-isystem{target}",
        f"-isystem{target / 'aarch64-target-linux-gnu'}",
    ]
    matrix_source = "glm_w4a8_pipeline.cpp" if tile_pipeline else "glm_w4a8_matmul.cpp"
    if all_bits:
        matrix_source = "glm_lowbit_a8_pipeline.cpp"
    sources = ["glm_w4a8_pack.cpp", matrix_source]
    provenance = {
        "_build": {
            "namespace": namespace,
            "version": version,
            "output_columns": output_columns,
            "helper_package": helper_package,
            "tile_pipeline": tile_pipeline,
            "all_bits": all_bits,
            "prefill_native": all_bits,
            "fused_moe": fused_moe,
            "prepared_weight_layout": prepared_weight_layout,
            "pair_scale_groups": pair_scale_groups,
            "fp16_swiglu": fp16_swiglu,
            "pair_hidden_quant": pair_hidden_quant,
            "strided_product_copy": strided_product_copy,
            "weight_decode_lut": weight_decode_lut,
            "wide_cube_k": wide_cube_k,
            "pair_prefill_scale_groups": pair_prefill_scale_groups,
        }
    }
    provenance["_helpers"] = {p.name: sha256(p) for p in helper_root.glob("*.py")}
    for name in sources:
        source = HERE / name
        output = build_dir / ("glm_w4a8_matmul.bin" if name == matrix_source else source.stem + ".bin")
        subprocess.run(
            [
                str(compiler),
                str(source),
                str(output),
                *options,
                f"-DGLM_INT4_OUTPUT_COLUMNS={output_columns}",
                *(["-DGLM_INT4_METADATA_GATHER"] if tile_pipeline else []),
            ],
            check=True,
        )
        provenance[name] = {"source_sha256": sha256(source), "binary_sha256": sha256(output)}
    if fused_moe:
        source = HERE / "glm_fused_moe.cpp"
        for stage in ("gate_up", "down"):
            output = build_dir / f"glm_fused_{stage}.bin"
            subprocess.run(
                [
                    str(compiler),
                    str(source),
                    str(output),
                    *options,
                    f"-I{HERE}",
                    *(["-DGLM_PREPARED_WEIGHT_LAYOUT"] if prepared_weight_layout else []),
                    *(["-DGLM_PAIR_SCALE_GROUPS"] if pair_scale_groups else []),
                    *(["-DGLM_FP16_SWIGLU"] if fp16_swiglu else []),
                    *(["-DGLM_PAIR_HIDDEN_QUANT"] if pair_hidden_quant else []),
                    *([f"-DGLM_NATIVE_WIDE_CUBE_K={wide_cube_k}"] if wide_cube_k else []),
                    *(["-DGLM_STRIDED_PRODUCT_COPY"] if strided_product_copy else []),
                    *(["-DGLM_WEIGHT_DECODE_LUT"] if weight_decode_lut else []),
                    *(["-DGLM_PAIR_PREFILL_SCALE_GROUPS"] if pair_prefill_scale_groups else []),
                    *(["-DGLM_FUSED_GATE_UP"] if stage == "gate_up" else []),
                ],
                check=True,
            )
            provenance[output.name] = {"source_sha256": sha256(source), "binary_sha256": sha256(output)}
        source = HERE / "glm_fused_pack.cpp"
        output = build_dir / "glm_fused_pack.bin"
        subprocess.run([str(compiler), str(source), str(output), *options, f"-I{HERE}"], check=True)
        provenance[output.name] = {"source_sha256": sha256(source), "binary_sha256": sha256(output)}
        provenance["glm_fused_quantize.h"] = {"source_sha256": sha256(HERE / "glm_fused_quantize.h")}
        source = HERE / "glm_fused_reduce.cpp"
        output = build_dir / "glm_fused_reduce.bin"
        subprocess.run([str(compiler), str(source), str(output), *options], check=True)
        provenance[output.name] = {"source_sha256": sha256(source), "binary_sha256": sha256(output)}
    if include_w3:
        csrc = source_root / "csrc" if (source_root / "csrc").is_dir() else source_root
        kernel = csrc / "gmm/w2_blocked_dequant_matmul_v310/op_kernel"
        common = csrc / "moe/common"
        catlass = source_root / "third_party/catlass/include"
        headers = ("w2_blocked_dequant_matmul_v310.h", "row_reuse_geometry.h", "compat_310p.h")
        snapshot = build_dir / "decoder"
        snapshot.mkdir()
        for name in headers:
            data = (kernel / name).read_bytes()
            (snapshot / name).write_bytes(data)
            provenance[name] = {"source_sha256": hashlib.sha256(data).hexdigest()}
        extra = [f"-I{snapshot}", f"-I{common}", f"-I{catlass}", *[f"-D{name}" for name in W3_DEFINES]]
        source = HERE / "reconstruction_kernel.cpp"
        output = build_dir / "reconstruction_kernel.bin"
        subprocess.run([str(compiler), str(source), str(output), *options, *extra], check=True)
        provenance[source.name] = {"source_sha256": sha256(source), "binary_sha256": sha256(output)}
    spec = importlib.util.find_spec("torch_npu")
    if spec is None or spec.origin is None:
        raise RuntimeError("build the bridge with the serving torch-npu environment")
    npu_root = Path(spec.origin).parent
    includes = [*include_paths(), str(npu_root / "include"), str(cann_root / "include"), sysconfig.get_path("include")]
    libraries = [*library_paths(), str(npu_root / "lib"), str(cann_root / "lib64")]
    bridge = build_dir / f"glm_reconstruction_bridge_v{version}.so"
    subprocess.run(
        [
            "c++",
            "-shared",
            "-fPIC",
            "-std=c++20",
            "-O2",
            f"-DGLM_RECONSTRUCTION_NAMESPACE={namespace}",
            f"-D_GLIBCXX_USE_CXX11_ABI={int(torch._C._GLIBCXX_USE_CXX11_ABI)}",
            *[f"-I{path}" for path in includes],
            str(HERE / "reconstruction_bridge.cpp"),
            *[f"-L{path}" for path in libraries],
            *[f"-Wl,-rpath,{path}" for path in libraries],
            "-ltorch_npu",
            "-lascendcl",
            "-lc10",
            "-ltorch_cpu",
            "-ltorch",
            "-ltorch_python",
            "-o",
            str(bridge),
        ],
        check=True,
    )
    provenance["reconstruction_bridge.cpp"] = {
        "source_sha256": sha256(HERE / "reconstruction_bridge.cpp"),
        "binary_sha256": sha256(bridge),
    }
    (build_dir / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    return build_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", required=True, type=Path)
    parser.add_argument("--cann-root", type=Path, default=Path("/usr/local/Ascend/ascend-toolkit/latest"))
    parser.add_argument("--source-root", type=Path, default=HERE.parents[1])
    parser.add_argument("--include-w3", action="store_true")
    parser.add_argument("--version", type=int, default=1)
    parser.add_argument("--output-columns", type=int, choices=(16, 32, 64, 128), default=16)
    parser.add_argument("--tile-pipeline", action="store_true")
    parser.add_argument("--all-bits", action="store_true")
    parser.add_argument("--fused-moe", action="store_true")
    parser.add_argument("--prepared-weight-layout", action="store_true")
    parser.add_argument("--pair-scale-groups", action="store_true")
    parser.add_argument("--fp16-swiglu", action="store_true", help="quality-gated FP16 SwiGLU experiment")
    parser.add_argument("--pair-hidden-quant", action="store_true", help="quality-gated paired hidden quantization")
    parser.add_argument(
        "--wide-cube-k",
        type=int,
        choices=(0, 128, 256),
        default=0,
        help="pack independent block32 dots into a wider native Cube (0 disables)",
    )
    parser.add_argument(
        "--pair-prefill-scale-groups",
        action="store_true",
        help="use M32 for paired prefill block32 dots; preserve FP32 accumulation order",
    )
    parser.add_argument("--weight-decode-lut", action="store_true", help="exact W2/W3 reconstruction with byte lookup")
    parser.add_argument("--strided-product-copy", action="store_true", help="contiguous sparse Cube product casts")
    args = parser.parse_args()
    print(
        build(
            args.build_dir,
            args.cann_root,
            args.source_root,
            args.include_w3,
            args.version,
            args.output_columns,
            args.tile_pipeline,
            args.all_bits,
            args.fused_moe,
            args.prepared_weight_layout,
            args.pair_scale_groups,
            args.fp16_swiglu,
            args.pair_hidden_quant,
            args.wide_cube_k,
            args.pair_prefill_scale_groups,
            args.weight_decode_lut,
            args.strided_product_copy,
        )
    )


if __name__ == "__main__":
    main()
