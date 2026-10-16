"""Build an isolated RTC dense kernel and versioned bridge without modifying OPP."""

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sysconfig
from pathlib import Path

from tools.rtsp_scene.dense_shapes import (
    DENSE_INNER_TILES,
    DENSE_OUTPUT_TILES,
    LEGACY_NAMESPACE,
    decode_rows,
    kernel_namespace,
    validate_cube_tile,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cann", type=Path, default=Path("/usr/local/Ascend/ascend-toolkit/latest"))
    parser.add_argument("--columns", type=int, choices=DENSE_OUTPUT_TILES, default=128)
    parser.add_argument("--inner-tile", type=int, choices=DENSE_INNER_TILES, default=256)
    parser.add_argument("--pipeline", action="store_true", help="Overlap weight DMA with Cube work")
    parser.add_argument(
        "--nd-scatter", action="store_true", help="Write NZ products directly to ND GM with strided DMA"
    )
    parser.add_argument("--max-decode-rows", type=int, choices=(8, 16), default=8)
    parser.add_argument("--active-epilogue", action="store_true", help="Skip padded NZ rows in the matrix epilogue")
    parser.add_argument("--namespace", default=LEGACY_NAMESPACE, help="Separate bridge registry for paired builds")
    args = parser.parse_args()
    try:
        validate_cube_tile(args.columns, args.inner_tile)
        kernel_namespace({"namespace": args.namespace})
        decode_rows({"max_decode_rows": args.max_decode_rows})
    except ValueError as error:
        parser.error(str(error))
    args.output.mkdir(parents=True, exist_ok=False)
    source = Path(__file__).resolve().parent
    cann = args.cann.resolve()
    compiler = args.output / "compile-dense"
    subprocess.run(
        [
            "c++",
            "-std=c++17",
            "-O2",
            str(source / "compile_dense.cpp"),
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
    target = cann / "tools/hcc/aarch64-target-linux-gnu/include/c++/7.3.0"
    binaries = []
    for inner in (2048, 4096, 6144):
        binary = args.output / f"dense-k{inner}.bin"
        subprocess.run(
            [
                str(compiler),
                str(source / "dense_i8.cpp"),
                str(binary),
                "--npu-arch=dav-2002",
                f"--sysroot={cann / 'tools/hcc/sysroot'}",
                f"-isystem{target}",
                f"-isystem{target / 'aarch64-target-linux-gnu'}",
                f"-DRTSP_DENSE_K={inner}",
                f"-DRTSP_DENSE_N={args.columns}",
                f"-DRTSP_DENSE_K_TILE={args.inner_tile}",
                f"-DRTSP_DENSE_PIPELINE={int(args.pipeline)}",
                f"-DRTSP_DENSE_ND_SCATTER={int(args.nd_scatter)}",
                f"-DRTSP_DENSE_ACTIVE_EPILOGUE={int(args.active_epilogue)}",
            ],
            check=True,
        )
        binaries.append(binary)
    activation = args.output / "nz-swiglu.bin"
    subprocess.run(
        [
            str(compiler),
            str(source / "nz_swiglu.cpp"),
            str(activation),
            "--npu-arch=dav-2002",
            f"--sysroot={cann / 'tools/hcc/sysroot'}",
            f"-isystem{target}",
            f"-isystem{target / 'aarch64-target-linux-gnu'}",
            f"-DRTSP_DENSE_MAX_ROWS={args.max_decode_rows}",
        ],
        check=True,
    )
    binaries.append(activation)
    # Build-only imports; do not initialize an NPU in the compiler process.
    import torch
    from torch.utils.cpp_extension import include_paths, library_paths

    spec = importlib.util.find_spec("torch_npu")
    if spec is None or spec.origin is None:
        raise RuntimeError("Build using the pinned torch-npu environment")
    npu = Path(spec.origin).parent
    includes = [*include_paths(), str(npu / "include"), str(cann / "include"), sysconfig.get_path("include")]
    libraries = [*library_paths(), str(npu / "lib"), str(cann / "lib64")]
    bridge = args.output / "dense-bridge.so"
    subprocess.run(
        [
            "c++",
            "-shared",
            "-fPIC",
            "-std=c++20",
            "-O2",
            f"-D_GLIBCXX_USE_CXX11_ABI={int(torch._C._GLIBCXX_USE_CXX11_ABI)}",
            f"-DRTSP_DENSE_NAMESPACE={args.namespace}",
            *[f"-I{path}" for path in includes],
            str(source / "dense_bridge.cpp"),
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
    hashes = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [
            *binaries,
            bridge,
            source / "dense_i8.cpp",
            source / "dense_bridge.cpp",
            source / "nz_swiglu.cpp",
            source / "compile_dense.cpp",
            source / "build_dense.py",
        ]
    }
    (args.output / "build.json").write_text(
        json.dumps(
            {
                "columns": args.columns,
                "inner_tile": args.inner_tile,
                "pipeline": args.pipeline,
                "nd_scatter": args.nd_scatter,
                "active_epilogue": args.active_epilogue,
                "max_decode_rows": args.max_decode_rows,
                "namespace": args.namespace,
                "sha256": hashes,
                "cann_root": str(cann),
                "architecture": "dav-2002",
                "torch_version": torch.__version__,
                "native_arguments": {"rtsp_dense_i8": 6, "rtsp_nz_swiglu": 5},
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
