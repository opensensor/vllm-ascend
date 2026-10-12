# SPDX-License-Identifier: Apache-2.0
"""Export validated receipts to a spreadsheet and standalone comparison plots."""

import argparse
import csv
import json
from pathlib import Path
from types import MappingProxyType

RECEIPTS = MappingProxyType(
    {
        "cuda_matrix": "cuda-matrix-final.json",
        "npu_matrix": "npu-matrix-final.json",
        "cuda_vector": "cuda-vector-final.json",
        "npu_vector": "npu-vector-final.json",
        "cuda_copy": "cuda-copy-final.json",
        "npu_copy": "npu-copy-final.json",
        "npu_rounding": "npu-fp32-rounding-final.json",
        "npu_card": "npu-card-final.json",
        "npu_capability": "npu-capability-final.json",
    }
)


def load_receipts(directory):
    result = {name: json.loads((directory / filename).read_text()) for name, filename in RECEIPTS.items()}
    for name, report in result.items():
        if not report.get("completed"):
            raise ValueError(f"incomplete receipt: {name}")
        if name == "npu_card":
            for case in report["cases"]:
                if not all(worker["final_validation"]["passed"] for worker in case["final_validation"]):
                    raise ValueError("dual-chip receipt failed final validation")
        else:
            for case in report["cases"]:
                if case["status"] == "passed" and not all(
                    case[key]["passed"]
                    for key in ("initial_validation", "changed_input_validation", "final_validation")
                ):
                    raise ValueError(f"passed status without numerical gates: {name}")
    return result


def select(report, name, dtype, layout="ND", *, pinned=False):
    matches = [
        row
        for row in report["cases"]
        if row["name"] == name and row["dtype"] == dtype and row["layout"] == layout and row["pinned"] == pinned
    ]
    if len(matches) != 1 or matches[0]["status"] != "passed":
        raise ValueError(f"no unique validated case: {name}/{dtype}/{layout}/{pinned}")
    return matches[0]


def export_csv(receipts, destination):
    fields = (
        "receipt",
        "backend",
        "name",
        "op",
        "dtype",
        "m",
        "n",
        "k",
        "layout",
        "pinned",
        "status",
        "reference_fp16_inputs",
        "mode",
        "repeats",
        "device_median_ms",
        "wall_median_ms",
        "tera_ops_per_second",
        "payload_gbytes_per_second",
        "read_plus_write_gbytes_per_second",
        "elements_per_second",
        "error",
    )
    with destination.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for name, report in receipts.items():
            if name == "npu_card":
                continue
            for case in report["cases"]:
                row = {key: case[key] for key in fields if key in case}
                row.update(
                    receipt=name,
                    backend=report["backend"],
                    reference_fp16_inputs=report.get("reference_fp16_inputs", False),
                )
                if case["status"] != "passed":
                    writer.writerow(row)
                    continue
                for mode, timing in case["timings"].items():
                    writer.writerow(
                        {
                            **row,
                            "mode": mode,
                            "repeats": timing["repeats_per_sample"],
                            "device_median_ms": timing["device"]["median_ms"],
                            "wall_median_ms": timing["wall"]["median_ms"],
                            **{key: value for key, value in timing["rates"].items() if key in fields},
                        }
                    )


def tables(receipts):
    lines = [
        "# Measured comparisons",
        "",
        "Queued device-event medians; three warmups, 15 samples. "
        "GEMM multiply and add each count as one operation. Ascend uses one chip with NZ weights unless stated.",
        "",
        "| Resident GEMM | RTX ms | Ascend ms | RTX TFLOP/s or TOP/s | Ascend TFLOP/s or TOP/s | RTX speed ratio |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for dtype in ("fp16", "int8"):
        for name in ("vector", "batch128", "square4096", "square8192"):
            cuda = select(receipts["cuda_matrix"], name, dtype)["timings"]["queued"]
            npu = select(receipts["npu_matrix"], name, dtype, "NZ")["timings"]["queued"]
            ct, nt = cuda["device"]["median_ms"], npu["device"]["median_ms"]
            lines.append(
                f"| {dtype.upper()} {name} | {ct:.6f} | {nt:.6f} | "
                f"{cuda['rates']['tera_ops_per_second']:.2f} | {npu['rates']['tera_ops_per_second']:.2f} | "
                f"{nt / ct:.2f}× |"
            )
    lines += [
        "",
        "| Two independent Ascend jobs | Both chips combined TFLOP/s or TOP/s | Wall ms for four jobs per chip |",
        "| --- | ---: | ---: |",
    ]
    for row in receipts["npu_card"]["cases"]:
        lines.append(
            f"| {row['case']['dtype'].upper()} {row['case']['name']} | "
            f"{row['aggregate_tera_ops_per_second']:.2f} | {row['coordinator_wall']['median_ms']:.3f} |"
        )
    lines += ["", "| FP32 vector operation | RTX µs | Ascend µs | RTX speed ratio |", "| --- | ---: | ---: | ---: |"]
    for name in (
        "add_67108864",
        "exp_67108864",
        "softmax_1x4096",
        "softmax_2048x4096",
        "rmsnorm_1x4096",
        "rmsnorm_2048x4096",
    ):
        ct = select(receipts["cuda_vector"], name, "fp32")["timings"]["queued"]["device"]["median_ms"]
        nt = select(receipts["npu_vector"], name, "fp32")["timings"]["queued"]["device"]["median_ms"]
        lines.append(f"| {name} | {ct * 1000:.3f} | {nt * 1000:.3f} | {nt / ct:.2f}× |")
    lines += [
        "",
        "| Copy | Size | RTX wall µs | Ascend wall µs | RTX wall payload GB/s | Ascend wall payload GB/s |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for op in ("copy", "h2d", "d2h"):
        for size in (8192, 1024**3):
            name = f"{op}_{size}"
            ct = select(receipts["cuda_copy"], name, "uint8", pinned=op != "copy")["timings"]["single"]["wall"][
                "median_ms"
            ]
            nt = select(receipts["npu_copy"], name, "uint8", pinned=op != "copy")["timings"]["single"]["wall"][
                "median_ms"
            ]
            lines.append(
                f"| {op} {'pinned' if op != 'copy' else 'D2D'} | {'8 KiB' if size == 8192 else '1 GiB'} | "
                f"{ct * 1000:.3f} | {nt * 1000:.3f} | {size / (ct * 1e6):.3f} | {size / (nt * 1e6):.3f} |"
            )
    return "\n".join(lines) + "\n"


def plot(receipts, directory):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    batches = (1, 8, 128, 512, 2048, 4096)
    names = ("vector", "batch8", "batch128", "batch512", "batch2048", "square4096")
    for key, layout, label in (
        ("cuda_matrix", "ND", "RTX PRO 6000"),
        ("npu_matrix", "NZ", "310P3 chip, NZ"),
        ("npu_matrix", "ND", "310P3 chip, ND"),
    ):
        values = [
            select(receipts[key], name, "fp16", layout)["timings"]["queued"]["rates"]["tera_ops_per_second"]
            for name in names
        ]
        axes[0].plot(batches, values, "o-", label=label)
    axes[0].set(
        xscale="log",
        yscale="log",
        xlabel="M (N = K = 4096)",
        ylabel="Measured FP16 TFLOP/s",
        title="Resident matrix multiplication",
    )
    axes[0].legend()
    sizes = (8192, 1024**2, 16 * 1024**2, 256 * 1024**2, 1024**3)
    for key, label in (("cuda_copy", "RTX"), ("npu_copy", "310P3 chip")):
        for op, marker in (("h2d", "o"), ("d2h", "s")):
            values = [
                size
                / (
                    select(receipts[key], f"{op}_{size}", "uint8", pinned=True)["timings"]["single"]["wall"][
                        "median_ms"
                    ]
                    * 1e6
                )
                for size in sizes
            ]
            axes[1].plot(sizes, values, marker + "-", label=f"{label} {op.upper()}")
    axes[1].set(
        xscale="log",
        xlabel="Payload bytes",
        ylabel="Payload GB/s, synchronized wall time",
        title="Pinned host/device transfers",
    )
    axes[1].legend()
    for axis in axes:
        axis.grid(True, alpha=0.3)
    fig.savefig(directory / "comparison.svg")
    svg = directory / "comparison.svg"
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    fig.savefig(directory / "comparison.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--plots", action="store_true")
    args = parser.parse_args()
    receipts = load_receipts(args.directory)
    export_csv(receipts, args.directory / "measurements.csv")
    (args.directory / "comparisons.md").write_text(tables(receipts))
    if args.plots:
        plot(receipts, args.directory)


if __name__ == "__main__":
    main()
