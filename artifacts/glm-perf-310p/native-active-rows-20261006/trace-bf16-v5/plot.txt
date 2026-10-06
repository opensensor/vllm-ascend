# SPDX-License-Identifier: Apache-2.0
"""Plot measured native stages and overlapping CANN pipe activity."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main():
    plt.switch_backend("Agg")
    root = Path(__file__).resolve().parent
    report = json.loads((root / "attribution.json").read_text())
    figure, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    for index, (label, ranks) in enumerate(report["workloads"].items()):
        positions = np.arange(4) + (index - 0.5) * 0.32
        durations = [
            rank["steady_decode"]["categories"]["native_int4_expert"]["summed_ms"]
            / rank["steady_decode"]["model_steps"]
            for rank in ranks
        ]
        axes[0].bar(positions, durations, width=0.30, label=label)
    axes[0].set(
        xticks=range(4),
        xticklabels=["rank 0", "rank 1", "rank 2", "rank 3"],
        ylabel="Native gate/up + down task sum / step (ms)",
        title="Steady routed-expert projection stages",
    )
    axes[0].legend(loc="upper left")
    metrics = ["mac_ratio", "vec_ratio", "scalar_ratio", "mte2_ratio"]
    positions = np.arange(len(metrics))
    for index, (label, ranks) in enumerate(report["workloads"].items()):
        ratios = [
            100 * np.mean([rank["decode"]["pipeline_metrics"]["native_int4_expert"][key] for rank in ranks])
            for key in metrics
        ]
        axes[1].bar(positions + (index - 0.5) * 0.32, ratios, width=0.30, label=label)
    axes[1].set(
        xticks=positions,
        xticklabels=["MAC", "Vector", "Scalar", "MTE2"],
        ylabel="Reported pipe ratio (%) over full decode",
        title="Pipes overlap; bars are not additive",
    )
    axes[1].legend()
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
    figure.savefig(root / "native-decode-hotspots.png", dpi=160)


if __name__ == "__main__":
    main()
