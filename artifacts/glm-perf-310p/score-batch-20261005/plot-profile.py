# SPDX-License-Identifier: Apache-2.0
"""Render reproducible PNGs from saved trace and serving measurements."""

import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    root = Path(__file__).resolve().parent
    prior = root.parent / "prompt-profile-20261005"
    traces = json.loads((prior / "historical-prefill-attribution.json").read_text())
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    names = [item["name"] for item in traces[0]["top_tasks"][:10]]
    labels = [
        "Packed expert projections",
        "KDA (all 9 stages)",
        "Sparse attention",
        "Scatter / state updates",
        "Batch matmul",
        "Cast / copies",
        "Multiply",
        "Layout conversion",
        "Reduce sum",
        "Matmul zero fill",
    ]
    values = []
    for name in names:
        values.append(
            [next(item["summed_ms"] for item in t["top_tasks"] if item["name"] == name) / 1000 for t in traces]
        )
    means = np.mean(values, axis=1)
    fig, axes = plt.subplots(1, 2, figsize=(15, 7), gridspec_kw={"width_ratios": [1.4, 1]})
    y = np.arange(len(labels))
    axes[0].barh(y, means, color=["#c75445", "#326eac"] + ["#a6b5c4"] * 8)
    axes[0].errorbar(
        means,
        y,
        xerr=[means - np.min(values, axis=1), np.max(values, axis=1) - means],
        fmt="none",
        color="#263746",
        capsize=3,
    )
    axes[0].set_yticks(y, labels)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Summed NPU task duration per rank (seconds)")
    axes[0].set_xlim(0, 6.1)
    for pos, value in enumerate(means):
        axes[0].text(value + 0.06, pos, f"{value:.2f}s", va="center", fontsize=10)
    stage_labels = ["Stage 7: scores + solve", "Stage 8", "Remaining 7 stages"]
    stage_values = [[], [], []]
    for t in traces:
        stages = t["kda_stages_ordinal_only"].values()
        totals = {v["inferred_stage_id"]: v["summed_ms"] / 1000 for v in stages}
        for bucket, value in zip(
            stage_values, [totals[7], totals[8], sum(v for k, v in totals.items() if k not in (7, 8))]
        ):
            bucket.append(value)
    stage_means = [statistics.mean(v) for v in stage_values]
    axes[1].barh(stage_labels, stage_means, color=["#326eac", "#6d96bf", "#b8cbdc"])
    axes[1].invert_yaxis()
    axes[1].set_xlim(0, 3.4)
    axes[1].set_xlabel("Summed KDA stage duration (seconds)")
    for pos, value in enumerate(stage_means):
        axes[1].text(
            value + 0.04, pos, f"{value:.2f}s · {100 * value / sum(stage_means):.1f}%", va="center", fontsize=10
        )
    axes[1].set_title("81.5% of KDA time in score / solve stage")
    fig.suptitle("GLM prompt hotspots — historical CANN trace, 4 October 2026", fontsize=17, fontweight="bold")
    fig.text(
        0.04,
        0.045,
        "Two 640-token prefill chunks · 4 ranks · 13.43s wall window. Bars are task sums, which can overlap.\n"
        "Error bars show rank min/max. KDA stage mapping inferred from host launch order. "
        "This predates the column-cache change.",
        fontsize=10,
    )
    fig.tight_layout(rect=[0, 0.12, 1, 0.92])
    fig.savefig(root / "historical-prefill-hotspots.png", dpi=180)
    plt.close(fig)

    event_path = root / "events-measured.json"
    if event_path.exists():
        events = json.loads(event_path.read_text())
        groups = ["expert_projection", "moe_grouped_total", "moe_all_reduce", "moe_total", "kda_prefill"]
        labels = [
            "Expert projections",
            "Grouped MoE (includes projections)",
            "MoE all-reduce",
            "MoE total (includes above)",
            "KDA prefill",
        ]
        fig, axis = plt.subplots(figsize=(12, 6))
        for rank, worker in enumerate(events):
            rows = worker["prefill_events"]["records"]
            values = [sum(r["stream_ms"] for r in rows if r["label"] == group) / 1000 for group in groups]
            axis.barh(np.arange(len(groups)) + (rank - 1.5) * 0.16, values, height=0.15, label=f"Rank {worker['rank']}")
        axis.set_yticks(np.arange(len(groups)), labels)
        axis.invert_yaxis()
        axis.set_xlabel("Summed NPU stream event span (seconds)")
        axis.legend()
        fig.suptitle("Current GLM prefill — resident event sample, 6 October 2026", fontsize=16, fontweight="bold")
        fig.text(
            0.04,
            0.04,
            "One uncached 640-token request · resident column cache + fused Sinkhorn · same worker weights.\n"
            "Nested scopes overlap; do not add bars. Event spans include queued work and host dispatch gaps.\n"
            "5.83s request elapsed; diagnostic instrumentation overhead is included.",
            fontsize=10,
        )
        fig.tight_layout(rect=[0, 0.14, 1, 0.91])
        fig.savefig(root / "current-prefill-events.png", dpi=180)
        plt.close(fig)

    baseline = json.loads((prior / "cold-fused-baseline.json").read_text())
    candidate = json.loads((prior / "cold-score-cache.json").read_text())
    assert [v["nonce"] for v in baseline] == [v["nonce"] for v in candidate]
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    y = np.arange(len(baseline))
    for axis, key, label in zip(
        axes,
        ["ttft_s", "input_tokens_per_ttft_second"],
        ["Time to first token (seconds)", "Input tokens / TTFT second"],
    ):
        for offset, records, color, name in [
            (-0.18, baseline, "#a6b5c4", "Fused baseline"),
            (0.18, candidate, "#326eac", "Column cache"),
        ]:
            vals = [r[key] for r in records]
            axis.barh(y + offset, vals, height=0.32, color=color, label=name)
            for pos, v in enumerate(vals):
                axis.text(v + max(vals) * 0.01, pos + offset, f"{v:.2f}", va="center", fontsize=10)
        axis.set_yticks(y, [f"{r['input_tokens']:,} tokens" for r in baseline])
        axis.set_xlabel(label)
        axis.set_xlim(0, max(r[key] for r in baseline + candidate) * 1.15)
        axis.invert_yaxis()
    axes[0].legend(loc="lower right")
    fig.suptitle("Qualified column cache — cold prompt comparison", fontsize=17, fontweight="bold")
    fig.text(
        0.04,
        0.045,
        "Same prompt tokens / nonce; zero cached tokens and preemptions. "
        "One pair per length across a native-package restart.\n"
        "311K context, MTP1, full decode graphs, fused Sinkhorn. "
        "Baseline overlapped CPU compilation; no confidence interval.",
        fontsize=10,
    )
    fig.tight_layout(rect=[0, 0.12, 1, 0.92])
    fig.savefig(root / "qualified-cold-prefill.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
