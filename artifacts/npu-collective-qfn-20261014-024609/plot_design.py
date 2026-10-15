# SPDX-License-Identifier: Apache-2.0
"""Render the N-rank research design and explicitly bounded qualification."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


def box(ax, x, y, width, height, text, color):
    ax.add_patch(
        FancyBboxPatch(
            (x, y), width, height, boxstyle="round,pad=0.02", facecolor=color, edgecolor="#526580", linewidth=1.2
        )
    )
    ax.text(x + width / 2, y + height / 2, text, ha="center", va="center", fontsize=11, color="#14243a")


def main():
    fig, ax = plt.subplots(figsize=(14, 8), dpi=150)
    fig.patch.set_facecolor("#f5f8fc")
    ax.set_facecolor("#f5f8fc")
    ax.set(xlim=(0, 14), ylim=(0, 8))
    ax.axis("off")
    ax.text(
        7,
        7.55,
        "Local NPU SUM: queued Torch → peer SDMA → local reduction",
        fontsize=18,
        weight="bold",
        ha="center",
        color="#14243a",
    )
    box(ax, 0.6, 5.55, 5.7, 1.35, "Ascend Duo A\nNPU 0                         NPU 1", "#dcecff")
    box(ax, 7.7, 5.55, 5.7, 1.35, "Ascend Duo B\nNPU 2                         NPU 3", "#dcecff")
    for left, right in [(1.4, 5.5), (8.5, 12.6)]:
        ax.annotate("", xy=(right, 5.85), xytext=(left, 5.85), arrowprops=dict(arrowstyle="<->", color="#1763a6", lw=2))
    ax.text(7, 5.1, "Stage 1: sibling exchange + local SUM (0 ↔ 1, 2 ↔ 3)", ha="center", fontsize=12)
    ax.annotate("", xy=(10.5, 4.55), xytext=(3.5, 4.55), arrowprops=dict(arrowstyle="<->", color="#b25d1d", lw=2))
    ax.text(7, 4.15, "Stage 2: cross-card exchange + local SUM (0 ↔ 2, 1 ↔ 3)", ha="center", fontsize=12)
    box(ax, 0.6, 2.55, 5.7, 1.05, "Each exchange\nSDMA write → peer READY → wait/reset → local add", "#e2f3e9")
    box(
        ax,
        7.7,
        2.55,
        5.7,
        1.05,
        "Two alternating inbox + READY slots\nOne in-order Torch stream; no remote vector reads",
        "#e2f3e9",
    )
    ax.text(
        7,
        2.0,
        "Arbitrary N: binomial reduce + reverse broadcast. Power-of-two N: recursive doubling.",
        ha="center",
        fontsize=11,
    )
    ax.text(
        7,
        1.45,
        "Measured raw N4, FP16, 2,560 elements: custom 80.30 µs | HCCL 142.09 µs",
        ha="center",
        fontsize=13,
        weight="bold",
        color="#174f3b",
    )
    ax.text(
        7,
        0.95,
        "Completed batches include host dispatch and arrival spread. Synthetic finite inputs; not model tok/s.",
        ha="center",
        fontsize=10,
    )
    ax.text(
        7,
        0.45,
        "QFN eager: C1 19.03 vs 18.03 tok/s | C4 58.52 vs 53.45 aggregate. Two trials each; graphs not admitted.",
        ha="center",
        fontsize=10,
        color="#7b4520",
    )
    fig.savefig(Path(__file__).with_name("design.png"), bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


if __name__ == "__main__":
    main()
