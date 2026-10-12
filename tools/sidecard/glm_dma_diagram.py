# SPDX-License-Identifier: Apache-2.0
"""Draw the integrated GLM transport with explicit physical ownership."""

import argparse
from pathlib import Path

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


def draw(output):
    """Show GPU aliases, physical DMA, CPU control, and clean release ordering."""
    figure = Figure(figsize=(18, 11), dpi=160, facecolor="#f4f7fb")
    FigureCanvasAgg(figure)
    ax = figure.add_axes((0.025, 0.025, 0.95, 0.95))
    ax.set(xlim=(0, 18), ylim=(0, 11))
    ax.axis("off")

    def box(x, y, width, height, title, lines, color):
        ax.add_patch(
            FancyBboxPatch(
                (x, y), width, height, boxstyle="round,pad=0.12", linewidth=2, edgecolor=color, facecolor="white"
            )
        )
        ax.text(x + width / 2, y + height - 0.35, title, fontsize=14, weight="bold", color=color, ha="center", va="top")
        ax.text(
            x + width / 2,
            y + (height - 0.7) / 2,
            "\n".join(lines),
            fontsize=11.5,
            color="#28384f",
            ha="center",
            va="center",
            linespacing=1.6,
        )

    def arrow(start, end, label, color):
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="<->", mutation_scale=18, linewidth=2, color=color))
        ax.text(
            (start[0] + end[0]) / 2,
            (start[1] + end[1]) / 2 + 0.22,
            label,
            fontsize=10.5,
            color=color,
            ha="center",
            va="bottom",
        )

    ax.text(
        0.3,
        10.65,
        "GLM: CUDA backbone + Ascend cold experts over direct DMA",
        fontsize=23,
        color="#14243a",
        weight="bold",
        va="top",
    )
    ax.text(
        0.3,
        10.0,
        "One RTX PRO 6000 + both chips of one Ascend 310P card · other Ascend card remains available",
        fontsize=13,
        color="#52647e",
        va="top",
    )
    box(
        0.4,
        8.1,
        17.0,
        1.2,
        "CPU / private socket control",
        [
            "CUDA source event completes → sequence / layer / rows → matching completion after DMA and write flush",
            "12-byte invocation identity · authenticated checkpoint, placement, nonce, and peer PID/UID",
        ],
        "#697588",
    )
    box(
        0.4,
        3.35,
        4.65,
        4.2,
        "CUDA coordinator",
        [
            "All 45 backbone layers + caches",
            "Attention / routers / shared MLP",
            "Full CUDA banks outside layers 18–32",
            "Configured hot middle-layer experts",
            "GPU packing + fixed rank-order merge",
            "Serialized requests; CUDA graphs retained",
        ],
        "#26714d",
    )
    for y, rank in ((5.6, 0), (3.35, 1)):
        box(
            6.3,
            y,
            4.0,
            1.95,
            f"GPU slot pair · worker {rank}",
            [
                "Dedicated CUDA input / output buffers",
                "Worker owns allocations + DMA-BUF",
                "Coordinator borrows CUDA IPC views",
            ],
            "#26714d",
        )
        box(
            12.25,
            y,
            5.15,
            1.95,
            f"Ascend worker · physical chip {rank + 2}",
            [
                "DMA directly into Torch HBM input",
                "Resident cold expert banks",
                "DMA directly from native FP32 output",
            ],
            "#ac5327",
        )
        arrow((10.45, y + 0.9), (12.05, y + 0.9), "PCIe DMA", "#2f55a2")
    arrow((5.2, 6.5), (6.1, 6.5), "CUDA IPC", "#26714d")
    arrow((5.2, 4.25), (6.1, 4.25), "CUDA IPC", "#26714d")
    ax.text(
        11.3,
        2.9,
        "Ascend drives both directions; NVIDIA DMA-BUF supplies GPU addresses",
        fontsize=11,
        color="#2f55a2",
        ha="center",
    )
    box(
        0.4,
        0.9,
        8.1,
        1.45,
        "Payload / math boundary",
        [
            "No hidden-row or expert-output bounce through host DRAM",
            "One decode row: 8,256 bytes in; 16,384 bytes out per active worker",
            "Only bounded routing metadata is read back for validation",
        ],
        "#2f55a2",
    )
    box(
        9.2,
        0.9,
        8.2,
        1.45,
        "Ownership / failure boundary",
        [
            "Drain GPU consumers → worker quiescence → unmap IPC → release owner",
            "Uncertain DMA or lost release: fail closed and retain original owners",
            "Private poisoned host guard retains HBM; its SQ is never submitted",
        ],
        "#697588",
    )
    ax.text(
        0.4,
        0.38,
        "Full-model readiness checks all 154,880 logits against the socket baseline. "
        "Measurements and limits are in the report.",
        fontsize=11,
        color="#52647e",
        va="center",
    )
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160, facecolor=figure.get_facecolor())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    draw(parser.parse_args().output)


if __name__ == "__main__":
    main()
