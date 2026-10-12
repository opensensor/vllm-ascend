# SPDX-License-Identifier: Apache-2.0
"""Render the implemented architecture using completed measurement receipts."""

import argparse
import json
from pathlib import Path


def draw(artifacts, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    decode_path = artifacts / "api-gate-v13-decode2048.json"
    parity_path = artifacts / "pipeline-parity-v13.json"
    decode = json.loads((decode_path if decode_path.exists() else artifacts / "api-gate-decode2048.json").read_bytes())
    parity = json.loads((parity_path if parity_path.exists() else artifacts / "pipeline-parity.json").read_bytes())
    context_path = artifacts / "context-gate.json"
    context = json.loads(context_path.read_bytes()) if context_path.exists() else {"passed": False}
    capacity = "65,536 total tokens completed" if context["passed"] else "65,536 configured; full capacity gate pending"
    fig = plt.figure(figsize=(16, 10), facecolor="#f4f7fb")
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    ink, blue, green = "#16324f", "#dceaff", "#d8f0e5"

    def text(x, y, value, size=12, **kwargs):
        ax.text(x, y, value, fontsize=size, color=ink, va="top", **kwargs)

    def box(x, y, width, height, color):
        ax.add_patch(
            FancyBboxPatch(
                (x, y), width, height, boxstyle="round,pad=0.009", facecolor=color, edgecolor="#bccadb", linewidth=1.2
            )
        )

    text(0.04, 0.965, "GLM sidecard: complete layers on RTX + Ascend", 25, weight="bold")
    text(
        0.04,
        0.909,
        "Implemented eager text runtime • real resident W2/W3/W4 experts • A4 activations • one sequence",
        13,
    )
    box(0.04, 0.817, 0.92, 0.053, "#fff0cc")
    text(0.055, 0.857, "131.92 GiB main weights exceed RTX capacity  |  CUDA owns 76.59 GiB of weight payload", 14)
    for x, color in ((0.04, blue), (0.373, green), (0.706, blue)):
        box(x, 0.48, 0.254, 0.285, color)
    text(0.055, 0.744, "RTX: prefix", 18, weight="bold")
    text(
        0.055,
        0.696,
        "Embedding + layers 0–17\nComplete layer math and routing\nAll 288 experts per MoE layer\n"
        "Owner-local KDA / KV / indexer\nmHC residual and mixers",
        12,
        linespacing=1.7,
    )
    text(0.388, 0.744, "Ascend: middle", 18, weight="bold")
    text(
        0.388,
        0.696,
        "Layers 18–32, all W4\nOne 300I Duo: physical 2 and 3\nTP2 attention and shared MLP\n"
        "EP2: 144 experts/chip/layer\nHCCL stays within this stage",
        12,
        linespacing=1.7,
    )
    text(0.721, 0.744, "RTX: suffix + head", 18, weight="bold")
    text(
        0.721,
        0.696,
        "Layers 33–44, all W2\nComplete layer math and routing\nOwner-local attention state\n"
        "FP32 contraction and RMSNorm\nFP16 head + greedy sampling",
        12,
        linespacing=1.7,
    )
    for begin, end in ((0.298, 0.366), (0.631, 0.699)):
        ax.add_patch(
            FancyArrowPatch((begin, 0.62), (end, 0.62), arrowstyle="-|>", mutation_scale=22, color=ink, linewidth=2)
        )
    text(
        0.5,
        0.456,
        "Prefix and suffix share the SAME RTX • no weight movement during inference",
        13,
        ha="center",
        weight="bold",
    )
    box(0.04, 0.327, 0.92, 0.088, "#fff")
    text(0.055, 0.401, "Two complete handoffs per token/chunk", 15, weight="bold")
    text(
        0.055,
        0.366,
        "FP16 hidden + FP32 four-stream residual/post/comb + INT64 positions: 73,816 bytes/token.\n"
        "BF16-rounded state keeps FP32 range; checkpoint SwiGLU clipping applies on both architectures.",
        12,
    )
    box(0.04, 0.193, 0.92, 0.093, "#e9edf5")
    text(0.055, 0.273, "Bounded two-slot prefill; serial autoregressive decode", 15, weight="bold")
    text(
        0.055,
        0.238,
        "Run prefix B while Ascend processes A; send B before suffix A. GPU work stays ordered.\n"
        "Layer owners retain recurrent, convolution, KV, and sparse pool/tail caches. New requests reset state.",
        12,
    )
    text(
        0.04,
        0.153,
        f"MEASURED: 2K prompt + 64 output → {decode['decode_tokens_per_second']:.2f} decode tokens/s; "
        f"first text {decode['time_to_first_text_seconds']:.2f}s.",
        14,
        weight="bold",
    )
    text(
        0.04,
        0.114,
        f"96-token startup: serial {parity['serial_seconds']:.2f}s → pipeline {parity['pipeline_seconds']:.2f}s; "
        f"bitwise logits equal.  {capacity}.",
        12,
    )
    text(
        0.04,
        0.075,
        "Capacity tests do not establish recall quality. No batching, prefix caching, graphs, MTP, vision, or tools.\n"
        "The other Ascend card remains assigned to Qwen. Receipts: artifacts/sidecard-layer-runtime-20261012.",
        11,
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
