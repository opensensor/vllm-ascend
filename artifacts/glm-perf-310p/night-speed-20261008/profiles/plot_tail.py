# SPDX-License-Identifier: Apache-2.0
"""Plot measured KDA tail stage costs and separate unprofiled serving requests."""

import gzip
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent


def stage_time(name):
    with gzip.open(ROOT / name, "rt") as source:
        rows = [json.loads(line) for line in source]
    row = next(row for row in rows if row["rank"] == "rank0" and row["phase"] == "tail63")
    return next(stage["summed_ms"] for stage in row["stages"] if stage["stage"] == 4)


def ttft(name, count):
    rows = json.loads((ROOT.parent / "measurements" / name).read_text())["records"]
    return [row["ttft_s"] for row in rows if row["prompt_tokens"] == count][-1]


def bars(axis, values, title, unit):
    rectangles = axis.bar(("Scalar", "Vector"), values, color=("#98a2b3", "#167d9a"))
    axis.set_title(title)
    axis.set_ylabel(unit)
    axis.spines[["top", "right"]].set_visible(False)
    axis.set_ylim(0, max(values) * 1.2)
    for rectangle, value in zip(rectangles, values):
        axis.text(rectangle.get_x() + rectangle.get_width() / 2, value, f"{value:,.2f}", ha="center", va="bottom")


def main():
    figure, axes = plt.subplots(1, 3, figsize=(12, 4.5))
    bars(
        axes[0],
        [
            stage_time(name)
            for name in ("tails-kda-stage-attribution.jsonl.gz", "vector-tails-kda-stage-attribution.jsonl.gz")
        ],
        "63-token post-W/U: 34 layers",
        "Summed task duration (ms)",
    )
    for axis, count in zip(axes[1:], (63, 639)):
        bars(
            axis,
            [ttft(name, count) for name in ("tail-before-serving.json", "tail-vector-serving.json")],
            f"Cold {count}-token serving request",
            "TTFT (seconds)",
        )
    figure.suptitle("GLM 310P3: qualified vector tail W/U", fontsize=15)
    figure.text(
        0.5,
        0.025,
        "Left: fresh CANN traces, rank 0, source-order stage attribution; task sum is not critical path.\n"
        "Right: last matching unprofiled cold request per pass. Ordered samples, not statistical guarantees.",
        ha="center",
        fontsize=9,
    )
    figure.tight_layout(rect=(0, 0.11, 1, 0.94))
    figure.savefig(ROOT / "tail-profile.png", dpi=180)
    plt.close(figure)


if __name__ == "__main__":
    main()
