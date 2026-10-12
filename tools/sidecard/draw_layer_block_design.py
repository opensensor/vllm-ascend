# SPDX-License-Identifier: Apache-2.0
"""Render the capacity ledger, complete handoff, and execution dependencies."""

import argparse
import json
from pathlib import Path

from .layer_block_design import GIB


def draw(design, probe, output):
    """Label measured transport separately from the planned model execution."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    plan = design["preferred"]
    transport = next(row for row in probe["cases"] if row["rows"] == 1 and row["residual_dtype"] == "fp32")["wall"][
        "median_ms"
    ]
    bulk = next(row for row in probe["cases"] if row["rows"] == 512 and row["residual_dtype"] == "fp32")["wall"][
        "median_ms"
    ]
    fig = plt.figure(figsize=(16, 11), facecolor="#f6f8fb")
    axis = fig.add_axes((0, 0, 1, 1))
    axis.set(xlim=(0, 1), ylim=(0, 1))
    axis.axis("off")
    blue, green, dark = "#e2edff", "#d9f2e8", "#172b4d"

    def text(x, y, value, size=12, **kwargs):
        axis.text(x, y, value, fontsize=size, color=kwargs.pop("color", dark), va="top", **kwargs)

    def box(x, y, width, height, color):
        axis.add_patch(
            FancyBboxPatch(
                (x, y),
                width,
                height,
                boxstyle="round,pad=0.009,rounding_size=0.012",
                facecolor=color,
                edgecolor="#b9c8d8",
                linewidth=1.3,
            )
        )

    def arrow(x1, y1, x2, y2, label=None):
        axis.add_patch(
            FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=20, color=dark, linewidth=2)
        )
        if label:
            text((x1 + x2) / 2, (y1 + y2) / 2 + 0.024, label, 10, ha="center")

    text(0.045, 0.965, "GLM sidecard: move whole layers, keep weights resident", 24, weight="bold")
    text(
        0.045,
        0.915,
        "CANDIDATE  •  checkpoint headers checked  •  full-state transport measured  •  model speed pending",
        12,
    )
    box(0.045, 0.827, 0.91, 0.055, "#fff0cc")
    text(
        0.065,
        0.865,
        f"Main weights: {design['inventory']['main_weights_bytes'] / GIB:.2f} GiB  |  RTX available: "
        f"{plan['gpu']['available'] / GIB:.2f} GiB  |  sidecard supplies the capacity needed to fit the model",
        13,
        weight="bold",
    )

    box(0.045, 0.47, 0.265, 0.31, blue)
    box(0.37, 0.47, 0.265, 0.31, green)
    box(0.695, 0.47, 0.26, 0.31, blue)
    text(0.062, 0.76, "RTX: prefix", 17, weight="bold")
    text(
        0.062,
        0.72,
        "Layers 0–17\nEmbedding + complete layer math\nKDA / DSA + local attention state\n"
        "Routing, shared and routed experts\nmHC state and FP32 reductions",
        12,
        linespacing=1.65,
    )
    text(0.062, 0.553, "All 288 experts in each owned\nMoE layer stay on the RTX.", 12, weight="bold")
    text(0.387, 0.76, "Ascend: middle block", 17, weight="bold")
    text(
        0.387,
        0.72,
        "Layers 18–32 • all W4\nOne card, two 310P3 chips (TP2)\n144 resident experts per chip/layer\n"
        "KDA / DSA state remains here\nHCCL stays within this stage",
        12,
        linespacing=1.65,
    )
    text(0.387, 0.553, "Whole-layer results return once\nafter the middle block finishes.", 12, weight="bold")
    text(0.712, 0.76, "RTX: suffix + head", 17, weight="bold")
    text(
        0.712,
        0.72,
        "Layers 33–44\nComplete W2 expert layers\nKDA / DSA + local attention state\n"
        "Final mHC contraction + norm\nLM head + sampling",
        12,
        linespacing=1.65,
    )
    text(0.712, 0.553, "Prefix and suffix use the SAME\nRTX and share its memory.", 12, weight="bold")
    arrow(0.315, 0.62, 0.363, 0.62, "1")
    arrow(0.64, 0.62, 0.688, 0.62, "2")

    box(0.045, 0.354, 0.91, 0.087, "#fff")
    text(0.063, 0.426, "Two complete deferred mHC handoffs per pass", 14, weight="bold")
    text(
        0.063,
        0.397,
        "Hidden output + 4-stream residual + post/comb factors + positions; preserve their exact dtypes.",
        12,
    )
    text(
        0.063,
        0.372,
        f"FP32 residual packet: {design['boundary_state']['bytes_per_token']:,} bytes/token  |  "
        f"Measured TWO crossings: {transport:.3f} ms decode / {bulk:.2f} ms at 512 rows (copy-only probe)",
        11,
    )

    gpu, npu = plan["gpu"], plan["npus"][1]
    box(0.045, 0.206, 0.435, 0.117, blue)
    box(0.515, 0.206, 0.44, 0.117, green)
    text(0.063, 0.31, "RTX memory ledger — both GPU blocks combined", 13, weight="bold")
    text(
        0.063,
        0.28,
        f"Weights {gpu['weights'] / GIB:.2f} GiB  +  cache 6 GiB  +  workspace 2 GiB\n"
        f"Reserve / rings included: {gpu['peak_budgeted_bytes'] / GIB:.2f} GiB total  |  "
        f"{gpu['headroom_bytes'] / GIB:.2f} GiB budget headroom",
        11,
        linespacing=1.6,
    )
    text(0.533, 0.31, "Ascend memory ledger — each chip", 13, weight="bold")
    text(
        0.533,
        0.28,
        f"Weights ≤{npu['weights'] / GIB:.2f} GiB  +  cache 4 GiB  +  workspace 2 GiB\n"
        f"Reserve / rings included: {npu['peak_budgeted_bytes'] / GIB:.2f} GiB total  |  "
        f"≥{npu['headroom_bytes'] / GIB:.2f} GiB budget headroom",
        11,
        linespacing=1.6,
    )

    text(
        0.045,
        0.173,
        "Execution rule: reduce the slow critical path; do not assume the next token can run early.",
        13,
        weight="bold",
    )
    text(
        0.045,
        0.14,
        "Single-token latency = GPU prefix + Ascend middle + GPU suffix + both handoffs.\n"
        "Prefill chunks / independent sequences can overlap; steady-state work is bounded by "
        "max(GPU prefix + suffix, Ascend).\n"
        "Double-buffered stage state, streamed startup loading, and local KV ownership "
        "prevent repeated weight / cache transfers.",
        11,
        linespacing=1.65,
    )
    text(
        0.045,
        0.052,
        "Cache/workspace allowances are provisional. GPU KDA/DSA and the 144-expert NPU stage "
        "need layer and model gates.\n"
        "No live GLM restart or measured end-to-end speedup. Existing W2/W3/W4 codes and FP16 scales "
        "retained; MTP/vision deferred.",
        10,
        color="#52667a",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    fig.savefig(output.with_suffix(".svg"))
    svg = output.with_suffix(".svg")
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    draw(json.loads(args.design.read_text()), json.loads(args.probe.read_text()), args.output)


if __name__ == "__main__":
    main()
