# SPDX-License-Identifier: Apache-2.0
"""Render the measured complete-layer design, without extrapolating speed."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parent


def main():
    # Show boundary ownership separately from measured component timing.
    plt.switch_backend("Agg")
    report = json.loads((ROOT / "summary.json").read_bytes())
    fig = plt.figure(figsize=(14, 8), facecolor="#101820")
    ax = fig.add_axes([0.03, 0.42, 0.94, 0.53])
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    fig.text(
        0.04, 0.96, "GLM sidecard: contiguous Ascend block over direct PCIe DMA", color="white", size=20, weight="bold"
    )
    fig.text(
        0.04,
        0.91,
        "Hardware tested on one RTX PRO 6000 + one Ascend 310P Duo (physical chips 2, 3)",
        color="#c4d4df",
        size=12,
    )

    def box(x, width, color, title, detail):
        ax.add_patch(
            FancyBboxPatch((x, 0.37), width, 0.38, boxstyle="round,pad=0.008", facecolor=color, edgecolor="none")
        )
        ax.text(x + width / 2, 0.64, title, ha="center", color="white", size=15, weight="bold")
        ax.text(x + width / 2, 0.49, detail, ha="center", va="center", color="white", size=10, linespacing=1.5)

    box(0.015, 0.22, "#22648b", "CUDA prefix", "Embedding + layers 0–17\nOwner: RTX\nKDA / DSA decode replay")
    box(
        0.365,
        0.27,
        "#277252",
        "Ascend layers 18–32",
        (
            "TP2 attention / shared MLP\nEP2: all 288 experts / layer\n"
            "Native MoE, Sinkhorn, BF16 round\nKDA spans replay; DSA eager"
        ),
    )
    box(0.765, 0.22, "#22648b", "CUDA suffix", "Layers 33–44 + vocabulary\nOwner: RTX\nKDA / DSA decode replay")
    for start, end, label in ((0.24, 0.355, "DMA input\nto both chips"), (0.64, 0.755, "DMA result\nfrom rank 0")):
        ax.annotate("", xy=(end, 0.57), xytext=(start, 0.57), arrowprops=dict(arrowstyle="->", lw=3, color="#f5ac45"))
        ax.text((start + end) / 2, 0.79, label, ha="center", color="#f5ac45", size=10)
    ax.text(
        0.5,
        0.27,
        "Complete deferred mHC state: hidden FP16 + residual / post / comb FP32 + positions",
        ha="center",
        color="#c4d4df",
        size=12,
    )
    ax.text(
        0.5,
        0.18,
        "2 architecture boundaries • 3 DMA copies • host control / acknowledgements • HCCL inside Ascend",
        ha="center",
        color="#c4d4df",
        size=11,
    )
    ax.text(
        0.5,
        0.08,
        "Serialized slot: 4 MiB capacity; up to 32 prefill rows; 1 row per ordinary decode step",
        ha="center",
        color="#c4d4df",
        size=11,
    )
    plot = fig.add_axes([0.09, 0.12, 0.49, 0.25], facecolor="#101820")
    rows = report["rows"][:2]
    labels = ["Socket", "Direct DMA"]
    left = [0, 0]
    colors = ["#409bc9", "#40ab7d", "#89c4e5"]
    for key, color, title in zip(
        ("prefix_ms", "middle_roundtrip_ms", "suffix_ms"),
        colors,
        ("CUDA prefix", "Ascend round trip", "CUDA suffix"),
        strict=True,
    ):
        values = [sum(r[f"{mode}_decode_component_median_ms"][key] for r in rows) / 2 for mode in ("socket", "dma")]
        plot.barh(labels, values, left=left, height=0.5, color=color, label=title)
        left = [a + b for a, b in zip(left, values, strict=True)]
    plot.set_xlabel("Median component time (ms); 2K prompt", color="white")
    plot.tick_params(colors="white")
    plot.spines[["top", "right", "left"]].set_visible(False)
    plot.legend(loc="upper center", bbox_to_anchor=(0.5, 1.23), ncol=3, frameon=False, labelcolor="white", fontsize=9)
    fig.text(
        0.63,
        0.34,
        (
            f"Matched block decode: {report['two_k_median_socket_tokens_per_second']:.2f} → "
            f"{report['two_k_median_dma_tokens_per_second']:.2f} tok/s"
        ),
        color="white",
        size=14,
        weight="bold",
    )
    fig.text(
        0.63,
        0.27,
        "Rank 0: 83.78 ms compute\n0.24 ms input DMA + 0.37 ms return DMA",
        color="#c4d4df",
        size=12,
        linespacing=1.7,
    )
    fig.text(
        0.63,
        0.16,
        (
            "Experimental block remains slower.\nFaster expert-only 128K service restored.\n"
            "2K / 8K checked; full block 128K boundary untested."
        ),
        color="#f5ac45",
        size=11,
        linespacing=1.6,
    )
    fig.text(
        0.04,
        0.035,
        (
            "192 complete vocabulary vectors match the socket block exactly. "
            "No general accuracy or long-context recall claim."
        ),
        color="#c4d4df",
        size=10,
    )
    fig.savefig(ROOT / "design.png", dpi=160, facecolor=fig.get_facecolor())


if __name__ == "__main__":
    main()
