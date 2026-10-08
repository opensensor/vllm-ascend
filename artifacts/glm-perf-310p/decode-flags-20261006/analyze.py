# SPDX-License-Identifier: Apache-2.0
"""Report all repetitions, acceptance metrics and answer regressions."""

import json
import re
import statistics
from pathlib import Path


def metrics(path):
    result = {}
    pattern = re.compile(r"^([^#{]+)(?:\{[^}]*\})? ([0-9.eE+-]+)$")
    for line in path.read_text().splitlines():
        match = pattern.match(line)
        if match:
            name, value = match.groups()
            if name in (
                "vllm:spec_decode_num_draft_tokens_total",
                "vllm:spec_decode_num_accepted_tokens_total",
                "vllm:num_preemptions_total",
            ):
                result[name] = result.get(name, 0) + float(value)
    return result


def main():
    root = Path(__file__).resolve().parent
    report = {}
    for label in ("baseline", "candidate", "rollback"):
        path = root / f"{label}-repeats.json"
        if not path.exists():
            continue
        repeats = json.loads(path.read_text())
        groups = {}
        for name in ("short_1", "short_4"):
            rates = [item["groups"][name]["aggregate_decode_tokens_per_s"] for item in repeats]
            median = statistics.median(rates)
            groups[name] = {
                "samples": rates,
                "median": median,
                "minimum": min(rates),
                "maximum": max(rates),
                "range_fraction": (max(rates) - min(rates)) / median,
                "cv_fraction": statistics.stdev(rates) / statistics.mean(rates) if len(rates) > 1 else None,
                "per_request_medians": [item["groups"][name]["per_request_decode_p50"] for item in repeats],
            }
        deltas = []
        for index in range(len(repeats)):
            before = metrics(root / f"{label}-{index}-metrics-before.txt")
            after = metrics(root / f"{label}-{index}-metrics-after.txt")
            delta = {name: after[name] - before.get(name, 0) for name in after}
            drafted = delta.get("vllm:spec_decode_num_draft_tokens_total", 0)
            delta["acceptance_fraction"] = (
                delta.get("vllm:spec_decode_num_accepted_tokens_total", 0) / drafted if drafted else None
            )
            deltas.append(delta)
        quality_path = root / f"{label}-quality-results.jsonl"
        quality = [json.loads(line) for line in quality_path.read_text().splitlines()] if quality_path.exists() else []
        report[label] = {
            "groups": groups,
            "metrics_deltas": deltas,
            "strict_pass_count": sum(row["passed"] for row in quality),
            "quality_count": len(quality),
            "quality_failures": [
                {key: row.get(key) for key in ("case_id", "content", "expected", "finish_reason", "valid")}
                for row in quality
                if not row["passed"]
            ],
        }
    if "baseline" in report and "candidate" in report:
        report["comparison"] = {
            name: {
                "median_gain_fraction": report["candidate"]["groups"][name]["median"]
                / report["baseline"]["groups"][name]["median"]
                - 1
            }
            for name in ("short_1", "short_4")
        }
        baseline_failed = {row["case_id"] for row in report["baseline"]["quality_failures"]}
        report["new_answer_failures"] = (
            [row for row in report["candidate"]["quality_failures"] if row["case_id"] not in baseline_failed]
            if report["baseline"]["quality_count"]
            else None
        )
    (root / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if "baseline" in report and "candidate" in report:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.style.use("seaborn-v0_8-whitegrid")
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
        for axis, name, title in zip(axes, ("short_1", "short_4"), ("c1 decode", "c4 aggregate decode")):
            for index, (label, color) in enumerate((("baseline", "#457b9d"), ("candidate", "#e76f51"))):
                group = report[label]["groups"][name]
                axis.scatter([index] * len(group["samples"]), group["samples"], color=color, s=55, zorder=3)
                axis.plot([index - 0.18, index + 0.18], [group["median"]] * 2, color=color, lw=3)
            axis.set_xticks([0, 1], ["Flags off", "Flags on"])
            axis.set_ylabel("Output tokens / second")
            gain = report["comparison"][name]["median_gain_fraction"] * 100
            axis.set_title(f"{title}: {gain:+.1f}% median")
        fig.suptitle(
            "GLM 5.3 Flash • 4 × 310P • 256 output tokens\nAfter warmup: one baseline run; three candidate runs"
        )
        fig.savefig(root / "decode-flags.png", dpi=180)
        plt.close(fig)


if __name__ == "__main__":
    main()
