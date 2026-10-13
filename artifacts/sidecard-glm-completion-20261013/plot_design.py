# SPDX-License-Identifier: Apache-2.0
"""Generate the measured GLM DMA completion design diagram."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

plt.switch_backend("Agg")

root = Path(__file__).parent
summary = json.loads((root / "summary.json").read_text())
fig, ax = plt.subplots(figsize=(16, 9), dpi=150)
fig.patch.set_facecolor("#0d1726")
ax.set_facecolor("#0d1726")
ax.set_xlim(0, 16)
ax.set_ylim(0, 9)
ax.axis("off")


def box(x, y, w, h, title, body, color):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.18", facecolor=color, edgecolor="#7693af"))
    ax.text(x + 0.2, y + h - 0.4, title, color="white", fontsize=15, fontweight="bold", va="top")
    ax.text(x + 0.2, y + h - 1.0, body, color="#e0edf6", fontsize=12, va="top", linespacing=1.6)


def arrow(x1, y1, x2, y2, label, color="#65d2d0"):
    ax.annotate("", (x2, y2), (x1, y1), arrowprops=dict(arrowstyle="->", color=color, lw=2))
    ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.17, label, fontsize=11, color=color, ha="center")


ax.text(
    0.35, 8.5, "GLM Sidecard: DMA completion integrated into generation", color="white", fontsize=22, fontweight="bold"
)
ax.text(0.35, 8.05, "RTX PRO 6000 Blackwell 96 GB + one Ascend 310P Duo card / two chips", color="#b6c8d8", fontsize=14)
box(
    0.5,
    3.7,
    6.3,
    3.5,
    "CUDA coordinator",
    "\n".join(
        [
            "Attention, dense layers, shared + hot experts",
            "GPU-generated invocation record",
            "Owns raw results and completion words",
            "Prequeues wait + ordered FP32 rank merges",
        ]
    ),
    "#183448",
)
box(
    9.1,
    3.7,
    6.3,
    3.5,
    "Ascend chips 2 and 3",
    "\n".join(
        [
            "Cold routed experts on both assigned chips",
            "Validate input record; compute in HBM",
            "DMA raw FP32 result directly into CUDA",
            "Then DMA the 8-byte completion sequence",
        ]
    ),
    "#343053",
)
arrow(6.95, 6.6, 8.9, 6.6, "Input + record")
arrow(8.9, 5.15, 6.95, 5.15, "FP32 result")
arrow(8.9, 4.25, 6.95, 4.25, "Final sequence")
box(
    0.5,
    1.3,
    14.9,
    1.65,
    "Host control remains",
    "\n".join(
        [
            "Schedules Ascend kernels and blocking DMA ioctls; handles admission and socket error replies.",
            "CUDA consumes results after DMA flags. Activation and result payloads bypass host staging.",
        ]
    ),
    "#263441",
)
v = summary["candidate_v2"]
ax.text(
    0.5,
    0.62,
    (
        f"Measured: {v['decode_median']:.2f} decode tok/s | "
        f"2K TTFT {v['ttft_2k_median']:.2f} s | 8K TTFT {v['ttft_8k']:.2f} s"
    ),
    color="#a2e4bf",
    fontsize=16,
)
ax.text(
    0.5,
    0.15,
    " ".join(
        [
            "Single request, uncached repeated prompt; 2K/64 outputs (3 runs),",
            "8K/8 outputs (1 run). 65,536-token configured limit.",
        ]
    ),
    color="#b6c8d8",
    fontsize=11,
)
fig.savefig(root / "design.png", facecolor=fig.get_facecolor(), bbox_inches="tight")
