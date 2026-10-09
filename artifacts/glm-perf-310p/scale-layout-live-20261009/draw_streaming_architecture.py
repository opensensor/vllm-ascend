# SPDX-License-Identifier: Apache-2.0
"""Render the source-backed GLM streaming proposal without changing the server."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

CURRENT = "#e6f1ff"
TARGET = "#fff0cf"
HANDOFF = "#eef0f4"
INK = "#172b46"


def render(destination: Path) -> None:
    """Draw the complete MoE path, then expand its proposed local streaming loop."""
    fig, ax = plt.subplots(figsize=(17, 13.5))
    fig.patch.set_facecolor("#ffffff")
    ax.set_xlim(0, 17)
    ax.set_ylim(0, 13.5)
    ax.axis("off")

    def label(x, y, text, size=12, weight="normal", color=INK):
        ax.text(x, y, text, fontsize=size, fontweight=weight, color=color, va="top", linespacing=1.5)

    def box(x, y, width, height, title, body, color=CURRENT):
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                width,
                height,
                boxstyle="round,pad=0.12,rounding_size=0.1",
                facecolor=color,
                edgecolor="#a4b3c5",
                linewidth=1.2,
            )
        )
        label(x + 0.15, y + height - 0.16, title, 13, "bold")
        label(x + 0.15, y + height - 0.58, body, 11)

    def arrow(points, dashed=False):
        for start, end in zip(points, points[1:]):
            ax.add_patch(
                FancyArrowPatch(
                    start,
                    end,
                    arrowstyle="-|>" if end == points[-1] else "-",
                    mutation_scale=16,
                    color="#506580",
                    linewidth=1.7,
                    linestyle="--" if dashed else "-",
                )
            )

    label(0.4, 13.22, "GLM expert streaming architecture", 24, "bold")
    label(0.4, 12.64, "Full MoE path • current v1021 as the starting point • target schedule is not implemented", 13)
    box(0.5, 11.36, 5.05, 0.85, "Model input", "Hidden states + router IDs / weights")
    box(10.95, 11.36, 5.05, 0.85, "Permanent checkpoint layout", "Packed W2 / W3 / W4 + qualified FP16 scales")

    box(
        0.5,
        8.98,
        4.55,
        1.68,
        "1  Quantize and route",
        "A4 once per token; stable expert rows\n"
        "Packed activation tiles + scalar scale banks\nNo device-to-host tensor inspection",
    )
    box(
        5.72,
        8.98,
        4.55,
        1.68,
        "2  Cache and reuse weights",
        "W4 loads directly; W2 / W3 rebuild INT4\nKeep an expert/output tile in L1\nReuse across its row batches",
    )
    box(
        10.95,
        8.98,
        5.05,
        1.68,
        "3  Stream gate and up projections",
        "INT4 Cube dots → vector scale / sum\nRetain FP32 accumulators on core\nTarget: batch readback and consumers",
        TARGET,
    )
    arrow([(3.0, 11.32), (3.0, 10.78)])
    arrow([(13.5, 11.32), (13.5, 11.0), (8.0, 11.0), (8.0, 10.78)])
    arrow([(5.15, 9.8), (5.58, 9.8)])
    arrow([(10.38, 9.8), (10.81, 9.8)])

    box(
        10.95,
        6.49,
        5.05,
        1.68,
        "4  Fused SwiGLU and requantization",
        "Finish gate and up over all K groups\n"
        "Apply the qualified FP16 rounding boundary\nVector SwiGLU → packed hidden A4",
    )
    box(
        5.72,
        6.49,
        4.55,
        1.68,
        "Compact inter-core handoff",
        "Packed hidden A4 + group-major scales\n"
        "One logical write/read through GM\nDown needs columns from multiple cores",
        HANDOFF,
    )
    box(
        0.5,
        6.49,
        4.55,
        1.68,
        "5  Stream down projection",
        "Reuse the same bounded Cube/vector loop\n"
        "Accumulate over the complete hidden axis\nStore qualified FP16 routed outputs",
        TARGET,
    )
    arrow([(13.5, 8.84), (13.5, 8.32)])
    arrow([(10.81, 7.32), (10.38, 7.32)])
    arrow([(5.58, 7.32), (5.15, 7.32)])

    box(
        0.5,
        4.35,
        4.55,
        1.35,
        "6  Combine local routes",
        "AI-Core reducer: route weights in FP32\nPreserve each token's addition order",
    )
    box(
        5.72,
        4.35,
        4.55,
        1.35,
        "7  Shared expert and ranks",
        "Add shared-expert output; HCCL reduce\nReturn the complete MoE result",
    )
    box(
        10.95,
        4.35,
        5.05,
        1.35,
        "Outer model continues",
        "mHC / residual, attention, KDA, next layer\nThose costs remain in end-to-end latency",
        HANDOFF,
    )
    arrow([(2.75, 6.35), (2.75, 5.84)])
    arrow([(5.15, 5.03), (5.58, 5.03)])
    arrow([(10.38, 5.03), (10.81, 5.03)])

    label(0.5, 3.86, "Inside each projection: bounded streaming over EVERY K32 group", 16, "bold")
    box(
        0.5,
        1.98,
        4.55,
        1.27,
        "Producer • MTE / L1",
        "Prefetch the next operand tile\nKeep resident weights; stage activations",
        TARGET,
    )
    box(
        5.72,
        1.98,
        4.55,
        1.27,
        "Compute • L0A / L0B / L0C",
        "Cube INT4 × INT4 → INT32\nKeep independent block32 dot products",
    )
    box(
        10.95,
        1.98,
        5.05,
        1.27,
        "Consumer • UB / vector",
        "Read bounded dot tiles; convert and scale\nAdd groups in order into FP32 accumulator",
        TARGET,
    )
    arrow([(5.15, 2.63), (5.58, 2.63)])
    arrow([(10.38, 2.63), (10.81, 2.63)])
    label(
        0.5,
        1.55,
        "Target overlap: load tile j+1 | compute tile j | consume tile j−1. Buffer ownership must be proven first.",
        12,
    )
    label(
        0.5,
        1.15,
        "Scalar control remains for addresses and events; "
        "tensor quantization, dots, scaling, activation and reduction use AI-Core units.",
        11,
    )
    label(
        0.5,
        0.76,
        "Memory: all K32 partials at 31 × 128 × K4096 require 1.94 MiB; "
        "nominal UB is 256 KiB, with 8 KiB reserved for SDK scratch.",
        11,
    )
    label(
        0.5,
        0.35,
        "Blue = existing stage   Amber = proposed scheduling redesign   "
        "Gray = explicit memory / model boundary. No predicted speed multiplier.",
        11,
    )
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    for extension in ("svg", "png"):
        path = destination / f"streaming-architecture.{extension}"
        fig.savefig(path, dpi=150)
        if extension == "svg":
            path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
    plt.close(fig)


if __name__ == "__main__":
    render(Path(__file__).resolve().parent)
