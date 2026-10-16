"""Compile and gate isolated large-row INT8 kernels against the checkpoint."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

PREFILL_WIDTHS = (2048, 4096, 6144)
PREFILL_COLUMNS = 128
PREFILL_ROWS = 64
CORE_COUNT = 8


def build(directory: Path, cann: Path) -> dict:
    directory.mkdir(mode=0o700, parents=True, exist_ok=False)
    source = Path(__file__).resolve().parent
    compiler = directory / "compile-prefill"
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
    hashes = {}
    for inner in PREFILL_WIDTHS:
        binary = directory / f"prefill-k{inner}.bin"
        subprocess.run(
            [
                str(compiler),
                str(source / "prefill_i8.cpp"),
                str(binary),
                "--npu-arch=dav-2002",
                f"--sysroot={cann / 'tools/hcc/sysroot'}",
                f"-isystem{target}",
                f"-isystem{target / 'aarch64-target-linux-gnu'}",
                f"-DRTSP_PREFILL_K={inner}",
            ],
            check=True,
        )
        hashes[binary.name] = hashlib.sha256(binary.read_bytes()).hexdigest()
    hashes["prefill_i8.cpp"] = hashlib.sha256((source / "prefill_i8.cpp").read_bytes()).hexdigest()
    manifest = {"columns": PREFILL_COLUMNS, "row_tile": PREFILL_ROWS, "arguments": 6, "sha256": hashes}
    (directory / "build.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def checked_build(directory: Path) -> dict:
    manifest = json.loads((directory / "build.json").read_text())
    if (manifest.get("columns"), manifest.get("row_tile"), manifest.get("arguments")) != (
        PREFILL_COLUMNS,
        PREFILL_ROWS,
        6,
    ):
        raise ValueError("Unsupported frozen prefill tile contract")
    for inner in PREFILL_WIDTHS:
        name = f"prefill-k{inner}.bin"
        if hashlib.sha256((directory / name).read_bytes()).hexdigest() != manifest["sha256"].get(name):
            raise ValueError(f"Prefill fingerprint mismatch: {name}")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--cann", type=Path, default=Path("/usr/local/Ascend/cann-9.1.0"))
    parser.add_argument("--model", type=Path)
    parser.add_argument("--matrix-build", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", type=int, choices=(0, 1), default=0)
    parser.add_argument("--rows", type=int, nargs="+", default=[17, 63, 64, 65, 464, 946, 984, 1024])
    args = parser.parse_args()
    if min(args.rows) < 1:
        parser.error("Use positive row counts")
    if args.compile:
        build(args.build, args.cann.resolve())
        return
    if not all((args.model, args.matrix_build, args.output)) or args.output.exists():
        parser.error("Supply the model, matrix bridge and a fresh output")
    checked_build(args.build)
    # Worker-only imports keep compiler and host test collection off the NPU.
    import torch
    import torch_npu

    from tools.rtsp_scene.dense_bench import measure_pair, pack_weight
    from tools.rtsp_scene.dense_runtime import KernelBackend
    from tools.rtsp_scene.dense_shapes import kernel_namespace
    from tools.rtsp_scene.kernel_compare import ReferenceLinear, compare_case

    torch.set_num_threads(4)
    torch.npu.set_device(args.device)
    torch.npu.set_compile_mode(jit_compile=False)
    backend = KernelBackend(args.matrix_build)
    kernel_class = getattr(torch.classes, kernel_namespace(backend.manifest)).Kernel
    kernels = {k: kernel_class(str(args.build / f"prefill-k{k}.bin"), "rtsp_prefill_i8") for k in PREFILL_WIDTHS}
    records = []
    with torch.inference_mode():
        for name in ("qkv", "output", "gate_up", "down"):
            reference = ReferenceLinear(args.model, name, args.device, torch_npu)
            inner, columns = reference.weight.shape
            packed = pack_weight(torch_npu.npu_format_cast(reference.weight.t(), 2).contiguous(), PREFILL_COLUMNS)
            for rows in args.rows:
                config = torch.tensor([rows, columns, 0, 1, 1], dtype=torch.int64, device="npu")
                x = torch.empty(rows, inner, dtype=torch.float16, device="npu")

                def candidate(x, config=config, packed=packed, inner=inner, columns=columns, reference=reference):
                    q, scales = backend.dynamic_quant(x)
                    q = backend.format_cast(q, 29)
                    y = backend.empty(x.shape[0], columns, torch.float16, x.device)
                    backend.launch(kernels[inner], [q, packed, reference.weight_scale, scales, y, config])
                    return y

                # Serving captures only decode graphs. Large-row prefill remains
                # eager; its ND-to-NZ Identity operator is not graph-capturable.
                result = compare_case(reference, (reference, candidate), x, iterations=10, repeats=5, check_only=True)
                result["execution"] = "eager prefill; graph capture not attempted"
                if result["passed"]:
                    left, right = measure_pair(
                        lambda x=x, reference=reference: reference(x),
                        lambda x=x, candidate=candidate: candidate(x),
                        torch.npu.synchronize,
                        20,
                        8,
                    )
                    result.update(
                        {
                            "control": left,
                            "candidate": right,
                            "reduction_percent": 100 * (1 - right["median_us"] / left["median_us"]),
                        }
                    )
                result.update({"projection": name, "rows": rows, "device": args.device})
                records.append(result)
                args.output.write_text(json.dumps(records, indent=2) + "\n")
                print(
                    json.dumps(
                        {
                            k: v
                            for k, v in result.items()
                            if k in ("projection", "rows", "passed", "graph_passed", "reduction_percent")
                        }
                    ),
                    flush=True,
                )


if __name__ == "__main__":
    main()
