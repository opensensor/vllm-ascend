# SPDX-License-Identifier: Apache-2.0
"""Draw the measured expert-only prototype and its pending integration."""

import argparse
import json
from pathlib import Path


def draw(study, probes, output):
    """Render pending placement separately from measured component evidence."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    fig = plt.figure(figsize=(16, 10), facecolor="#f5f7fb")
    axis = fig.add_axes((0, 0, 1, 1))
    axis.set(xlim=(0, 1), ylim=(0, 1))
    axis.axis("off")
    dark, blue, green = "#20334f", "#e2edff", "#ddf3e8"

    def text(x, y, value, size=12, **kwargs):
        axis.text(x, y, value, va="top", fontsize=size, color=dark, **kwargs)

    def box(x, y, width, height, color):
        axis.add_patch(
            FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.009", facecolor=color, edgecolor="#b8c8dc")
        )

    def arrow(start, end):
        axis.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=18, color=dark, linewidth=1.7))

    text(0.045, 0.96, "GLM sidecard: Ascend as a resident cold-expert engine", 23, weight="bold")
    text(
        0.045,
        0.915,
        "CANDIDATE DESIGN  |  real expert math + transfers validated  |  full-model integration pending",
        12,
    )
    box(0.045, 0.80, 0.91, 0.075, "#fff0cc")
    text(0.063, 0.86, "Keep the backbone on RTX; send only selected cold-expert work to Ascend.", 15, weight="bold")
    text(
        0.063,
        0.827,
        "All selected outputs contribute to the token. "
        "Cold describes usage frequency; every required result must finish.",
        11,
    )

    box(0.045, 0.47, 0.43, 0.28, blue)
    text(0.064, 0.73, "RTX: all 45 layers' backbone + hot experts", 15, weight="bold")
    text(
        0.064,
        0.687,
        "Attention / KDA state / DSA cache\nmHC + routing + shared MLPs\n"
        "Resident hot expert weights\nFP32 hot + cold partial merge → next layer",
        13,
        linespacing=1.65,
    )
    text(0.064, 0.517, "Embedding, final norm, head and sampling stay here.", 11)

    box(0.57, 0.62, 0.385, 0.13, green)
    text(0.59, 0.73, "Ascend chip 2: resident cold expert bank", 14, weight="bold")
    text(
        0.59,
        0.69,
        "Clipped W4A4 projections + weighted FP32 partial\nPersistent independent worker; local native reduction",
        11,
        linespacing=1.65,
    )
    box(0.57, 0.45, 0.385, 0.13, green)
    text(0.59, 0.56, "Ascend chip 3: second cold expert bank", 14, weight="bold")
    text(
        0.59,
        0.52,
        "Same engine; disjoint resident ownership in the design\nGPU hot computation overlaps cold worker execution",
        11,
        linespacing=1.65,
    )
    arrow((0.48, 0.70), (0.565, 0.70))
    arrow((0.48, 0.60), (0.565, 0.51))
    arrow((0.565, 0.64), (0.48, 0.64))
    arrow((0.565, 0.46), (0.48, 0.50))
    text(0.486, 0.757, "Selected work", 9)
    text(
        0.048,
        0.442,
        "Pinned request: FP16 hidden + int32 IDs + original FP32 route weights\n"
        "Return: one weighted FP32 vector per token/chip. Backbone and residual state remain on RTX.",
        11,
        linespacing=1.6,
    )

    box(0.045, 0.145, 0.43, 0.21, "#fff")
    text(0.064, 0.34, "Measured expert-only token latency", 15, weight="bold")
    lines = []
    for cold in (0, 1, 2, 4, 8):
        values = [
            next(row["median_ms"] for row in probe["cases"] if row["rows"] == 1 and row["cold_experts"] == cold)
            for probe in probes
        ]
        label = "RTX all 8 experts" if cold == 0 else f"{cold} cold + {8 - cold} hot"
        lines.append(f"{label:<23} {min(values):.2f}–{max(values):.2f} ms")
    text(0.064, 0.297, "\n".join(lines), 12, family="monospace", linespacing=1.55)
    text(
        0.064,
        0.178,
        "Layer 18; synthetic inputs; each chip tested separately.\n"
        "20 gated configurations; original route weights; repeatable merge.",
        10,
        linespacing=1.45,
    )

    box(0.52, 0.145, 0.435, 0.21, "#fff")
    text(0.539, 0.34, "What the real-model trace tells us", 15, weight="bold")
    matched = study["workloads"][0]["profiled_component_median_ms"]
    text(
        0.539,
        0.298,
        "15 Ascend layers, synchronized diagnostic:\n"
        f"mHC {matched['mhc']:.1f} ms  |  attention {matched['attention']:.1f} ms\n"
        f"experts {matched['routed_experts']:.1f} ms  |  reductions {matched['collectives']:.1f} ms",
        12,
        linespacing=1.5,
    )
    admitted = study["ledger"]["uniform_hot_experts_per_layer"]
    text(
        0.539,
        0.224,
        "Retaining existing RTX banks + 85 GiB weight budget:\n"
        f"{admitted} hot experts/middle layer; ~15 cold waits/token\n"
        "on Python, reasoning and arithmetic completion fixtures.",
        11,
        linespacing=1.5,
    )
    text(0.539, 0.158, "Profile adds synchronization; these costs are not serving latency.", 9)
    text(
        0.045,
        0.105,
        "Next integration gate: complete GPU backbone, arbitrary expert ownership, "
        "two-chip overlap and long-context memory.",
        12,
        weight="bold",
    )
    text(
        0.045,
        0.07,
        "Prototype preserves clipping and original FP32 route weights/partials. "
        "GPU hot slots are padded; no HCCL or weight paging in this probe.\n"
        "Live service remains the qualified whole-layer layout. No full-model speedup claim for this candidate.",
        10,
        linespacing=1.5,
    )
    fig.savefig(output, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--probes", nargs=2, type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    draw(json.loads(args.study.read_bytes()), [json.loads(path.read_bytes()) for path in args.probes], args.output)


if __name__ == "__main__":
    main()
