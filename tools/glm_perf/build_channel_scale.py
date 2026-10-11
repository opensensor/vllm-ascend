# SPDX-License-Identifier: Apache-2.0
"""Build an append-only channel-scale candidate without opening an NPU device."""

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

from .channel_scale_math import LAYOUT
from .channel_scale_runtime import KERNEL_NAMES

HERE = Path(__file__).resolve().parent
HELPERS = (
    "channel_scale_math.py",
    "channel_scale_layout.py",
    "channel_scale_runtime.py",
    "fused_weight_layout.py",
    "fused_offset_tables.py",
    "glm_int4.py",
    "glm_fused_moe.py",
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build(
    destination,
    cann_root,
    version,
    cube_rows=32,
    fused_gate_up=False,
    fused_decode=False,
    identity_smoothing=False,
    reciprocal_quant=False,
    fused_prefill=False,
    integer_reconstruct=False,
    cache_source_rows=False,
    stream_weights=False,
    token_prefill=False,
    cache_prefill_routes=False,
):
    """Snapshot the admitted contract, compile kernels/bridge, and seal SHA provenance."""
    if reciprocal_quant and not identity_smoothing:
        raise ValueError("reciprocal quantization currently requires proven identity smoothing")
    if fused_prefill and not fused_gate_up:
        raise ValueError("fused prefill requires UB gate/up fusion")
    if fused_decode and not fused_gate_up:
        raise ValueError("fused decode requires UB gate/up fusion")
    if cache_prefill_routes and (not fused_prefill or token_prefill):
        raise ValueError("prefill route cache requires bulk fused prefill")
    if token_prefill and not fused_prefill:
        raise ValueError("token prefill requires fused prefill")
    if cube_rows not in (32, 64):
        raise ValueError("native product rows must be 32 or 64")
    if type(version) is not int or version < 1:
        raise ValueError("candidate version must be positive")
    destination, cann_root = Path(destination).resolve(), Path(cann_root).resolve(strict=True)
    spec = importlib.util.find_spec("torch_npu")
    if spec is None or spec.origin is None:
        raise RuntimeError("build the bridge in the serving torch-npu environment")
    namespace = f"glm_channel_scale_v{version}"
    destination.mkdir(parents=True, exist_ok=False)
    snapshot = destination / "sources"
    snapshot.mkdir()
    names = (*KERNEL_NAMES, "glm_channel_gate_up") if fused_gate_up else KERNEL_NAMES
    if fused_decode:
        names = (*names, "glm_channel_down_decode")
    if fused_prefill:
        names = (*names, "glm_channel_down_prefill")
    sources = (
        "compile_reconstruction.cpp",
        "channel_scale_bridge.cpp",
        "glm_channel_broadcast.h",
        "glm_channel_integer.h",
        "glm_channel_row_cache.h",
        "glm_channel_down_token.cpp",
        *[n + ".cpp" for n in names],
    )
    for name in (*sources, *HELPERS):
        shutil.copy2(HERE / name, snapshot / name)
    compiler = destination / "compile-channel-scale"
    subprocess.run(
        [
            "c++",
            "-std=c++17",
            "-O2",
            str(snapshot / "compile_reconstruction.cpp"),
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
        f"-DGLM_CHANNEL_ROWS={cube_rows}",
        f"-I{snapshot}",
    ]
    if identity_smoothing:
        options.append("-DGLM_CHANNEL_IDENTITY_SMOOTHING")
    if reciprocal_quant:
        options.append("-DGLM_CHANNEL_RECIPROCAL_QUANT")
    if integer_reconstruct:
        options.append("-DGLM_CHANNEL_INTEGER_RECONSTRUCT")
    if cache_source_rows:
        options.append("-DGLM_CHANNEL_CACHE_SOURCE_ROWS")
    if stream_weights:
        options.append("-DGLM_CHANNEL_STREAM_WEIGHTS")
    if cache_prefill_routes:
        options.append("-DGLM_CHANNEL_CACHE_PREFILL_ROUTES")
    for name in names:
        source_name = "glm_channel_down_token" if token_prefill and name == "glm_channel_down_prefill" else name
        subprocess.run(
            [str(compiler), str(snapshot / (source_name + ".cpp")), str(destination / (name + ".bin")), *options],
            check=True,
        )
    npu_root = Path(spec.origin).parent
    includes = [*include_paths(), str(npu_root / "include"), str(cann_root / "include"), sysconfig.get_path("include")]
    libraries = [*library_paths(), str(npu_root / "lib"), str(cann_root / "lib64")]
    subprocess.run(
        [
            "c++",
            "-shared",
            "-fPIC",
            "-std=c++20",
            "-O2",
            f"-DGLM_RECONSTRUCTION_NAMESPACE={namespace}",
            f"-D_GLIBCXX_USE_CXX11_ABI={int(torch._C._GLIBCXX_USE_CXX11_ABI)}",
            *[f"-I{p}" for p in includes],
            str(snapshot / "channel_scale_bridge.cpp"),
            *[f"-L{p}" for p in libraries],
            *[f"-Wl,-rpath,{p}" for p in libraries],
            "-ltorch_npu",
            "-lascendcl",
            "-lc10",
            "-ltorch_cpu",
            "-ltorch",
            "-ltorch_python",
            "-o",
            str(destination / "bridge.so"),
        ],
        check=True,
    )
    helpers = destination / (namespace + "_helpers")
    helpers.mkdir()
    (helpers / "__init__.py").write_text("# SPDX-License-Identifier: Apache-2.0\n")
    for name in HELPERS:
        shutil.copy2(snapshot / name, helpers / name)
    files = (*[n + ".bin" for n in names], "bridge.so")
    manifest = {
        "cube_rows": cube_rows,
        "fused_gate_up": fused_gate_up,
        "fused_decode": fused_decode,
        "fused_prefill": fused_prefill,
        "abi_version": 4 if token_prefill else 3,
        "identity_smoothing": identity_smoothing,
        "reciprocal_quant": reciprocal_quant,
        "integer_reconstruct": integer_reconstruct,
        "cache_source_rows": cache_source_rows,
        "stream_weights": stream_weights,
        "token_prefill": token_prefill,
        "cache_prefill_routes": cache_prefill_routes,
        "kernel_sources": {
            n: ("glm_channel_down_token.cpp" if token_prefill and n == "glm_channel_down_prefill" else n + ".cpp")
            for n in names
        },
        "complete": True,
        "layout": LAYOUT,
        "namespace": namespace,
        "files": {name: digest(destination / name) for name in files},
        "sources": {name: digest(snapshot / name) for name in (*sources, *HELPERS)},
        "helpers": {name: digest(helpers / name) for name in (*HELPERS, "__init__.py")},
        "compiler_options": options,
        "cann_root": str(cann_root),
        "torch": torch.__version__,
        "validation": "CPU compilation only; no device opened or kernels executed",
    }
    (destination / "provenance.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--cann-root", type=Path, default=Path("/usr/local/Ascend/ascend-toolkit/latest"))
    parser.add_argument("--version", type=int, required=True)
    parser.add_argument("--cube-rows", type=int, choices=(32, 64), default=32)
    parser.add_argument("--fused-gate-up", action="store_true")
    parser.add_argument("--fused-decode", action="store_true")
    parser.add_argument("--identity-smoothing", action="store_true")
    parser.add_argument("--reciprocal-quant", action="store_true")
    parser.add_argument("--fused-prefill", action="store_true")
    parser.add_argument("--integer-reconstruct", action="store_true")
    parser.add_argument("--cache-source-rows", action="store_true")
    parser.add_argument("--stream-weights", action="store_true")
    parser.add_argument("--token-prefill", action="store_true")
    parser.add_argument("--cache-prefill-routes", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            build(
                args.destination,
                args.cann_root,
                args.version,
                args.cube_rows,
                args.fused_gate_up,
                args.fused_decode,
                args.identity_smoothing,
                args.reciprocal_quant,
                args.fused_prefill,
                args.integer_reconstruct,
                args.cache_source_rows,
                args.stream_weights,
                args.token_prefill,
                args.cache_prefill_routes,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
