# SPDX-License-Identifier: Apache-2.0
"""Draw the implemented cold expert design from completed deployment receipts."""

import argparse
import json
from pathlib import Path


def draw(artifacts, output):
    """Render ownership, overlap, and completed measurements from receipts."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    decode = json.loads((artifacts / "decode2048.json").read_bytes())
    context = json.loads((artifacts / "context-gate.json").read_bytes())
    completed = max((case["usage"]["total_tokens"] for case in context.get("cases", [])), default=0)
    fig, ax = plt.subplots(figsize=(16, 10), facecolor="#f2f6fb")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    ink = "#12334d"

    def text(x, y, value, size=13, **kwargs):
        ax.text(x, y, value, fontsize=size, color=ink, va="top", **kwargs)

    def box(x, y, width, height, color):
        ax.add_patch(
            FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.01", facecolor=color, edgecolor="#a9bfd1")
        )

    def arrow(start, end):
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=23, color=ink, linewidth=2))

    text(0.045, 0.958, "GLM sidecard: RTX backbone + Ascend cold experts", 25, weight="bold")
    text(0.045, 0.905, "Implemented 45-layer text model • one RTX PRO 6000 + one Atlas 300I Duo", 15)
    box(0.045, 0.79, 0.91, 0.065, "#fff0c6")
    text(0.06, 0.842, "131.92 GiB main weights exceed RTX VRAM; 84.98 GiB weight payload assigned to RTX.", 14)
    text(
        0.06,
        0.813,
        "Expert weights remain resident. All attention, recurrent, KV, indexer, and mHC state stays on RTX.",
        12,
    )

    box(0.045, 0.46, 0.535, 0.28, "#dceaff")
    text(0.065, 0.719, "RTX: complete backbone, layers 0–44", 19, weight="bold")
    text(
        0.065,
        0.67,
        "Attention / indexer / mHC / router / shared MLPs\n"
        "Embedding, FP32 final normalization, head, sampling\n"
        "All 288 experts in layers 3–17 and 33–44\n"
        "22 static hot experts per layer 18–32\n"
        "KDA graph replay; DSA cache updates stay eager",
        13,
        linespacing=1.7,
    )
    box(0.67, 0.46, 0.285, 0.28, "#d7efdf")
    text(0.689, 0.719, "Ascend: cold experts", 18, weight="bold")
    text(
        0.689,
        0.67,
        "Physical chips 2 and 3\nIndependent resident workers\n"
        "144 W4 experts/chip/layer\nLayers 18–32; no HCCL\nWeighted FP32 partials returned",
        12,
        linespacing=1.8,
    )
    arrow((0.585, 0.65), (0.664, 0.65))
    arrow((0.664, 0.54), (0.585, 0.54))
    text(0.626, 0.687, "x + routes", 10, ha="center")
    text(0.626, 0.529, "partials", 10, ha="center")
    box(0.045, 0.292, 0.91, 0.105, "#ffffff")
    text(
        0.063,
        0.382,
        "Per middle layer: dispatch both active chips → GPU hot + shared → receive → merge",
        15,
        weight="bold",
    )
    text(
        0.063,
        0.345,
        "Pinned host IPC: 8,256 input bytes + 16,384 output bytes per row per active chip.\n"
        "Original route weights preserved; fixed GPU/rank-0/rank-1 merge. Chips with zero work are skipped.",
        12,
    )
    text(
        0.045,
        0.235,
        f"MEASURED: 2K prompt + 64 output → {decode['decode_tokens_per_second']:.2f} decode tokens/s; "
        f"first text {decode['time_to_first_text_seconds']:.2f}s.",
        16,
        weight="bold",
    )
    text(
        0.045,
        0.183,
        f"Completed context capacity: {completed:,} total tokens. Capacity does not establish recall quality.",
        13,
    )
    text(
        0.045,
        0.129,
        "Serial requests; 32-row prefill chunks; GPU/NPU overlap inside each expert call.\n"
        "Real mixed W2/W3/W4 weights, A4 expert activations; clipped SwiGLU; FP32 residual range preserved.\n"
        "No prefix caching, MTP, batching, tools, or vision. The other Ascend card remains available.",
        12,
        linespacing=1.6,
    )
    fig.savefig(output, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    draw(args.artifacts, args.output)


if __name__ == "__main__":
    main()
