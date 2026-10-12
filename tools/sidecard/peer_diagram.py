# SPDX-License-Identifier: Apache-2.0
"""Draw the hardware-qualified direct DMA path and its lifetime guard."""

import argparse


def draw(output):
    """Keep direct payload movement distinct from descriptor-only control."""
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
            x + width / 2, y + height - 0.36, title, ha="center", va="center", fontsize=15, weight="bold", color=color
        )
        axis.text(
            x + width / 2,
            y + height / 2 - 0.15,
            detail,
            ha="center",
            va="center",
            fontsize=12,
            linespacing=1.55,
            color="#25324b",
        )

    axis.text(0.45, 8.45, "Direct CUDA VRAM ↔ Ascend HBM DMA", fontsize=25, weight="bold", color="#13233a")
    axis.text(
        0.45,
        7.95,
        "Hardware-validated on RTX PRO 6000 + both chips of the assigned Ascend 310P card",
        fontsize=14,
        color="#455673",
    )
    box(
        0.55,
        4.7,
        4.2,
        2.7,
        "RTX / NVIDIA VRAM",
        "Dedicated CUDA allocations\nNVIDIA DMA-BUF fd export\n"
        "Mapped for the Ascend requester\nCUDA readiness / write visibility",
        "#276b46",
    )
    box(
        6.0,
        4.7,
        4.1,
        2.7,
        "Ascend DMA engine",
        "Drives both transfer directions\nPhysical scatter/gather nodes\n"
        "Uses mapped NVIDIA addresses\nNo payload bounce in host DRAM",
        "#3452a2",
    )
    box(
        11.4,
        4.7,
        4.0,
        2.7,
        "Ascend local HBM",
        "Fresh owned device buffers\nPhysical cache IDs 2 and 3\n"
        "Retained until DMA completes\nOther card remains available",
        "#a34e29",
    )
    for start, end, label in [((4.95, 6.0), (5.8, 6.0), "PCIe"), ((10.3, 6.0), (11.2, 6.0), "local")]:
        axis.add_patch(FancyArrowPatch(start, end, arrowstyle="<->", mutation_scale=17, linewidth=2.5, color="#3b526b"))
        axis.text((start[0] + end[0]) / 2, 6.37, label, ha="center", fontsize=12, color="#3b526b")
    axis.text(
        8.0,
        4.13,
        "GPU and Ascend are under different PCIe roots; no topology or IOMMU settings changed",
        ha="center",
        fontsize=12,
        color="#455673",
    )
    box(
        0.55,
        1.8,
        6.1,
        1.65,
        "CPU / kernel control",
        "Attach fd, retain ranges, submit DMA, wait completion\nFirst API uses a blocking ioctl; mappings are per call",
        "#626b79",
    )
    box(
        8.05,
        1.8,
        7.35,
        1.65,
        "HBM lifetime guard · metadata only",
        "ACL host guard stays poisoned throughout validation\nIts reference descriptor SQ is never submitted",
        "#626b79",
    )
    axis.text(
        0.55,
        1.2,
        "1,440 direct copies passed · 4 KB to 1 MiB · payloads, red zones, and poisoned guards verified",
        fontsize=13,
        color="#25324b",
    )
    axis.text(
        0.55,
        0.75,
        "8 KB: ~0.23–0.26 ms GPU → HBM; ~0.21–0.22 ms HBM → GPU, including setup and completion",
        fontsize=12,
        color="#25324b",
    )
    axis.text(
        0.55,
        0.32,
        "Experimental root-only kernel bridge. GLM remains on its qualified socket backend; "
        "model integration is pending.",
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
