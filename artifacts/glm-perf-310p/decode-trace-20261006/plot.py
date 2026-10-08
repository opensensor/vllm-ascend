# SPDX-License-Identifier: Apache-2.0
"""Create all-rank decode attribution PNGs from saved CANN exports."""

import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    root = Path(__file__).resolve().parent
    data = json.loads((root / "attribution.json").read_text())
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    categories = [
        "expert_projection",
        "other_matmul",
        "copy_cast_layout",
        "ai_cpu_cast",
        "ai_cpu_other",
        "other",
        "kda_convolution",
        "sinkhorn",
        "kpool",
        "sparse_attention",
    ]
    labels = [
        "Packed expert projections",
        "Other matrix multiplications",
        "Copies / layouts / casts",
        "AI CPU casts",
        "Other AI CPU operations",
        "Other vector operators",
        "KDA + convolution",
        "Fused Sinkhorn",
        "Live kpool kernels",
        "Sparse attention",
    ]
    fig, axes = plt.subplots(1, 2, figsize=(15, 7), sharey=True)
    for axis, (workload, ranks) in zip(axes, data["workloads"].items()):
        concurrency = int(workload[1:])
        values = []
        periods = []
        for rank in ranks:
            steps = [s for s in rank["steps"] if s["phase"] == "decode" and s["requests"] == concurrency]
            values.append(
                [statistics.mean(s["categories"].get(c, {}).get("summed_ms", 0) for s in steps) for c in categories]
            )
            periods.append(statistics.mean(s["span_ms"] for s in steps))
        values = np.array(values)
        means = values.mean(0)
        y = np.arange(len(labels))
        axis.barh(y, means, color=["#c75445"] + ["#a6b5c4"] * 2 + ["#d79b38"] * 2 + ["#a6b5c4"] * 2 + ["#326eac"] * 3)
        axis.errorbar(
            means, y, xerr=[means - values.min(0), values.max(0) - means], fmt="none", color="#263746", capsize=3
        )
        for pos, value in enumerate(means):
            axis.text(value + max(means) * 0.015, pos, f"{value:.2f}", va="center", fontsize=10)
        axis.set_xlim(0, max(means) * 1.2)
        axis.set_yticks(y, labels)
        axis.set_title(f"{workload}: {statistics.mean(periods):.1f} ms mean profiled device step")
        axis.set_xlabel("Summed task time per steady decode model step (ms)")
    axes[0].invert_yaxis()
    fig.suptitle("GLM decode hotspots — c1 / c4, all four ranks", fontsize=17, fontweight="bold")
    fig.text(
        0.04,
        0.04,
        "6 October 2026 · resident column cache + fused Sinkhorn · 311K configured context · MTP1 · full graphs.\n"
        "Bars are task sums, not additive latency. Error bars: rank min/max. c4 drain steps excluded.\n"
        "Profiler overhead is included. Communication labels inside replayed graphs may be incomplete; raw device tasks retained.",
        fontsize=10,
    )
    fig.tight_layout(rect=[0, 0.14, 1, 0.91])
    fig.savefig(root / "decode-hotspots.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for axis, (workload, ranks) in zip(axes, data["workloads"].items()):
        for rank in ranks:
            steps = [s for s in rank["steps"] if s["phase"] == "decode"]
            axis.plot(range(len(steps)), [s["span_ms"] for s in steps], marker=".", label=f"Rank {rank['rank']}")
        axis.set_title(workload)
        axis.set_xlabel("Decode model step (MTP target + draft)")
        axis.legend(ncol=2)
        axis.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("CANN device iteration duration (ms)")
    fig.suptitle("All-rank decode timing — measured device step boundaries", fontsize=16, fontweight="bold")
    fig.text(
        0.04,
        0.03,
        "Profiling affects timings; use separate unprofiled serving runs for throughput. c4 final steps include request drain.",
        fontsize=10,
    )
    fig.tight_layout(rect=[0, 0.08, 1, 0.9])
    fig.savefig(root / "decode-rank-steps.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
