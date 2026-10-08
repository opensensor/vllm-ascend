# SPDX-License-Identifier: Apache-2.0
"""Plot archived measured traces; task sums are not critical-path attribution."""

import gzip
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent


def rows(name):
    with gzip.open(str(ROOT / name) + ".gz", "rt") as source:
        return [json.loads(line) for line in source]


def profile(name, phase):
    return next(row for row in rows(name) if row["folder"].endswith(phase + "/rank0"))


def bar(axis, labels, values, title, unit):
    bars = axis.bar(labels, values, color=("#98a2b3", "#167d9a"))
    axis.set_title(title, fontsize=12)
    axis.set_ylabel(unit)
    axis.spines[["top", "right"]].set_visible(False)
    axis.set_ylim(0, max(values) * 1.2)
    for rectangle, value in zip(bars, values):
        axis.text(rectangle.get_x() + rectangle.get_width() / 2, value, f"{value:,.1f}", ha="center", va="bottom")


def main():
    first = "initial-all-op-attribution.jsonl"
    beta = "all-op-attribution.jsonl"
    labels = ("Initial", "Fusions + beta")
    figure, axes = plt.subplots(2, 2, figsize=(11, 8))
    bar(
        axes[0, 0],
        labels,
        [profile(name, "decode_c1")["span_ms"] for name in (first, beta)],
        "One profiled decode step",
        "Wall duration (ms)",
    )
    bar(
        axes[0, 1],
        labels,
        [profile(name, "prefill_1280")["span_ms"] for name in (first, beta)],
        "Last 640-token prefill step",
        "Wall duration (ms)",
    )
    counts = [
        next(
            group["count"]
            for group in profile(name, "decode_c1")["groups"]
            if group["name"] == "aclnnReduceSum_ReduceSumOpAiCore_ReduceSum"
        )
        for name in (first, beta)
    ]
    bar(axes[1, 0], labels, counts, "Decode ReduceSum launches", "Launch count")
    stages = []
    for name in ("final-kda-stages.jsonl", "kda-stage-attribution.jsonl"):
        rank = next(row for row in rows(name) if row["rank"] == "rank0")
        stages.append(next(stage["summed_ms"] for stage in rank["stages"] if stage["stage"] == 8))
    bar(
        axes[1, 1],
        ("Scalar beta", "Vector beta"),
        stages,
        "KDA W/U preparation: 34 layers",
        "Summed task duration (ms)",
    )
    figure.suptitle("GLM on 310P3: measured serving profiles, rank 0", fontsize=15)
    figure.text(
        0.5,
        0.025,
        "Single archived step per configuration; profiler overhead included.\n"
        "Multiple fusions/storage changes in upper panels. Lower-right stage mapping follows source launch order.",
        ha="center",
        fontsize=9,
    )
    figure.tight_layout(rect=(0, 0.075, 1, 0.96))
    figure.savefig(ROOT / "serving-profile.png", dpi=180)
    plt.close(figure)


if __name__ == "__main__":
    main()
