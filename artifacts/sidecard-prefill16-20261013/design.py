# SPDX-License-Identifier: Apache-2.0
"""Render the qualified sidecard architecture and matched cold measurements."""

import json
from pathlib import Path

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


def main():
    """Plot paired compute times beside the data flow and overlapping worker spans."""
    root = Path(__file__).resolve().parent
    summary = json.loads((root / "summary.json").read_text())
    rows = 1024
    capacity = rows * 16384 // 1024**2
    groups = rows // 32
    figure = Figure(figsize=(13, 9), facecolor="#f8fafc")
    FigureCanvasAgg(figure)
    figure.text(
        0.5,
        0.965,
        f"GLM sidecard: {rows} cold rows with {capacity} MiB logical slots",
        ha="center",
        fontsize=21,
        weight="bold",
        color="#0f172a",
    )
    figure.text(
        0.5,
        0.92,
        "One RTX PRO 6000 + one Ascend 310P Duo • Same full Code prompt • Cold KV",
        ha="center",
        fontsize=12,
        color="#475569",
    )
    architecture = figure.add_axes([0.03, 0.46, 0.94, 0.42])
    architecture.set_xlim(0, 13)
    architecture.set_ylim(0, 5)
    architecture.axis("off")

    def box(x, y, width, height, label, color):
        architecture.add_patch(
            FancyBboxPatch(
                (x, y),
                width,
                height,
                boxstyle="round,pad=0.12",
                linewidth=1.5,
                edgecolor=color,
                facecolor="white",
            )
        )
        architecture.text(
            x + width / 2,
            y + height / 2,
            label,
            ha="center",
            va="center",
            fontsize=11,
            linespacing=1.5,
            color="#0f172a",
        )

    def arrow(start, end):
        architecture.add_patch(
            FancyArrowPatch(
                start,
                end,
                arrowstyle="-|>",
                mutation_scale=15,
                linewidth=1.7,
                color="#64748b",
            )
        )

    box(
        0.2,
        2.4,
        4.4,
        2,
        f"RTX PRO 6000 • 96 GB\n{groups} original 32-token CUDA chunks\n"
        "Attention, routing, hot/shared MLP\nRecurrent and KV state stays here",
        "#16a34a",
    )
    box(
        7.6,
        2.4,
        4.8,
        2,
        f"One Ascend 310P Duo\nOne {rows}-row batch per active chip\n"
        "Resident cold expert weights\nStateless MLP work, layers 18–32",
        "#0284c7",
    )
    arrow((4.8, 3.15), (7.4, 3.15))
    architecture.text(6.1, 3.45, "DMA: ≤8 MiB/piece\nFP16 inputs + routes", ha="center", fontsize=10, color="#475569")
    box(
        3.4,
        0.15,
        6.2,
        1.2,
        f"CUDA ordered hot + rank 0 + rank 1 merge\n"
        f"{capacity} MiB slots, two 8 MiB transfers, 64 KiB control\n"
        "Both pieces finish before sequence publication",
        "#7c3aed",
    )
    arrow((2.2, 2.25), (4.2, 1.5))
    arrow((10, 2.25), (8.4, 1.5))
    architecture.text(10.15, 1.8, "FP32 result pieces", fontsize=9, color="#475569")
    measurements = summary["measurements"]
    chart = figure.add_axes([0.085, 0.13, 0.54, 0.25])
    cases = ["2K", "8K", "full-code"]
    colors = ["#16a34a", "#7c3aed"]
    for column, window in enumerate((512, 1024)):
        values = [
            next(x["prefill_seconds"] for x in measurements if x["case"] == case and x["window_rows"] == window)
            for case in cases
        ]
        positions = [index + (column - 0.5) * 0.3 for index in range(len(cases))]
        chart.bar(positions, values, 0.3, color=colors[column], label=f"{window} cold rows")
        for position, value in zip(positions, values, strict=True):
            chart.text(position, value + 3, f"{value:.1f}", ha="center", fontsize=9)
    chart.set_xticks(list(range(3)), ["2,048 tokens", "8,192 tokens", "Code: 23,820 tokens"])
    chart.set_ylabel("Cold prefill compute seconds")
    chart.legend(frameon=False, loc="upper left")
    chart.spines[["top", "right"]].set_visible(False)
    chart.set_ylim(0, max(x["prefill_seconds"] for x in measurements) * 1.18)
    profiles = json.loads((root / "worker-span-summary.json").read_text())["samples"]
    spans = [
        next(x["native_microseconds_per_row"] for x in profiles if x["window_rows"] == window and x["rank"] == 0)
        for window in (512, 1024)
    ]
    profile_chart = figure.add_axes([0.73, 0.13, 0.23, 0.25])
    profile_chart.bar(["512", "1024"], spans, color=colors)
    profile_chart.set_ylabel("Mean native span per row, µs")
    profile_chart.set_xlabel("Cold rows per call")
    profile_chart.set_title("Worker 0: retained Code calls", fontsize=10)
    profile_chart.spines[["top", "right"]].set_visible(False)
    profile_chart.set_ylim(0, max(spans) * 1.2)
    for index, value in enumerate(spans):
        profile_chart.text(index, value + 3, f"{value:.1f}", ha="center", fontsize=10)
    figure.text(
        0.5,
        0.055,
        "One matched run per path/fixture. All vocabulary logits match bitwise; "
        "native spans include launch and synchronization.",
        ha="center",
        fontsize=9,
        color="#475569",
    )
    figure.text(
        0.5,
        0.025,
        "Host admits commands; activation/result payloads stay on devices. "
        "Ranks and CUDA overlap. Decode math is unchanged.",
        ha="center",
        fontsize=9,
        color="#475569",
    )
    figure.savefig(root / "design.png", dpi=170, facecolor=figure.get_facecolor())


if __name__ == "__main__":
    main()
