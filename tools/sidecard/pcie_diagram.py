# SPDX-License-Identifier: Apache-2.0
"""Draw the qualified first PCIe bus backend and its integration boundaries."""

import argparse


def draw(output):
    """Draw ownership, both DMA legs, the host copy, and measured limits."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    figure, axis = plt.subplots(figsize=(16, 9), dpi=150)
    figure.patch.set_facecolor("#f5f7fb")
    axis.set(xlim=(0, 16), ylim=(0, 9))
    axis.axis("off")

    def box(x, y, width, height, title, detail, color):
        axis.add_patch(
            FancyBboxPatch(
                (x, y), width, height, boxstyle="round,pad=0.16", linewidth=1.5, edgecolor=color, facecolor="white"
            )
        )
        axis.text(
            x + width / 2, y + height - 0.35, title, ha="center", va="center", fontsize=15, weight="bold", color=color
        )
        axis.text(
            x + width / 2,
            y + height / 2 - 0.12,
            detail,
            ha="center",
            va="center",
            fontsize=12,
            linespacing=1.6,
            color="#25324b",
        )

    def arrow(start, end, label, color="#3b526b", offset=0.22):
        axis.add_patch(FancyArrowPatch(start, end, arrowstyle="<->", mutation_scale=16, linewidth=2, color=color))
        axis.text(
            (start[0] + end[0]) / 2, (start[1] + end[1]) / 2 + offset, label, ha="center", fontsize=11, color=color
        )

    axis.text(0.4, 8.5, "Ascend ↔ CUDA PCIe bus · first qualified backend", fontsize=24, weight="bold", color="#13233a")
    axis.text(
        0.4,
        8.02,
        "CUDA IPC + native DMA streams + two bounded transfer slots per Ascend worker",
        fontsize=15,
        color="#455673",
    )
    box(
        0.45,
        4.55,
        4.2,
        2.8,
        "RTX / CUDA producer",
        "Dedicated device buffers\nExported CUDA IPC handles\nSource-ready IPC events\nBuffer lifetime held until ACK",
        "#276b46",
    )
    box(
        6.0,
        4.55,
        4.2,
        2.8,
        "Isolated bridge worker",
        "Imports CUDA buffers\nLoads CUDA + ACL native SDKs\nOwn stream / events per slot\nOne worker per Ascend chip",
        "#3452a2",
    )
    box(
        11.55,
        4.55,
        3.95,
        2.8,
        "Ascend 310P",
        "Assigned physical chips 2, 3\nDevice buffers / expert inputs\n"
        "Readiness and completion events\nOther card remains available",
        "#a34e29",
    )
    arrow((4.85, 6.45), (5.8, 6.45), "IPC", offset=0.3)
    arrow((10.4, 6.45), (11.35, 6.45), "ACL", offset=0.3)
    axis.text(
        7.9,
        4.02,
        "Socket carries control metadata and completion identity only",
        ha="center",
        fontsize=13,
        color="#3452a2",
    )
    box(0.5, 1.85, 3.9, 1.6, "CUDA-pinned host slot", "CUDA D2H / H2D\nNative completion event", "#276b46")
    box(
        5.95,
        1.85,
        4.25,
        1.6,
        "Local host-memory copy",
        "One CPU copy per direction\nAfter source DMA completes",
        "#626b79",
    )
    box(11.55, 1.85, 3.9, 1.6, "ACL-pinned host slot", "Ascend H2D / D2H\nNative completion event", "#a34e29")
    arrow((4.6, 2.52), (5.75, 2.52), "RAM", offset=0.25)
    arrow((10.4, 2.52), (11.35, 2.52), "RAM", offset=0.25)
    axis.text(
        0.55,
        1.25,
        "Measured: 2,400 byte-exact round trips · two 8 KB transfers: 1.37–1.38 ms (27–30% lower)",
        fontsize=13,
        color="#25324b",
    )
    axis.text(
        0.55,
        0.77,
        "Host-staged backend: direct cross-vendor peer DMA and shared dual registration are unqualified.",
        fontsize=12,
        color="#626b79",
    )
    axis.text(
        0.55,
        0.33,
        "Library is hardware-tested. GLM still uses its qualified socket transport; model speedup is not yet measured.",
        fontsize=12,
        color="#626b79",
    )
    figure.savefig(output, facecolor=figure.get_facecolor(), bbox_inches=None)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args()
    draw(arguments.output)


if __name__ == "__main__":
    main()
