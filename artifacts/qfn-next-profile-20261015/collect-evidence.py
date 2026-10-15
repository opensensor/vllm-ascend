"""Collect compact measurements from preserved resident experiment receipts."""

import json
import statistics
from pathlib import Path

ROOT = Path("/srv/ai/storage/artifacts/qfn-next-profile-20261015")
PROFILES = Path("/srv/ai/storage/artifacts/qfn-mixed-prefill-20261015")
LABELS = (
    "fresh-retained-native-20261015",
    "fresh-retained-late-native-20261015",
    "gdn-merge-shadow-v1",
    "gdn-merge-native-v1",
    "gdn-merge-off-a",
    "gdn-merge-on-a",
    "gdn-merge-off-b",
    "gdn-merge-on-b",
)


def main():
    result = dict(runs={}, gates={})
    for label in LABELS:
        path = PROFILES / label / "receipt.json"
        if not path.exists():
            continue
        receipt = json.loads(path.read_text())
        run = dict(source=str(path), unpaused=receipt.get("unpaused"), mode=receipt["mode"])
        for kind in ("decode", "prefill"):
            if kind not in receipt:
                continue
            row = receipt[kind]
            run[kind] = {k: row[k] for k in ("usage", "ttft_s", "max_content_gap_s", "median_content_gap_s")}
        run["ranks"] = []
        for path in sorted((PROFILES / label).glob("rank*-spans.json")):
            spans = json.loads(path.read_text())
            mixed = [
                r["duration_ns"] / 1e6
                for r in spans["records"]
                if r["label"] == "execute_model" and sorted(r["tokens"].values()) == [3, 512]
            ]
            run["ranks"].append(
                dict(
                    rank=spans["rank"],
                    mixed_steps=len(mixed),
                    median_execute_ms=statistics.median(mixed) if mixed else None,
                )
            )
        result["runs"][label] = run
    for label in ("schedule-5150-v1", "scatter-v1", "gather-v1", "merge-on-v1", "final-audit"):
        path = ROOT / label / "receipt.json"
        if path.exists():
            receipt = json.loads(path.read_text())
            result["gates"][label] = dict(source=str(path), result=receipt["applied"]["applied"])
    benchmarks = ROOT.parent / "npu-collective-qfn-20261014-024609/qsa-resident-v1"
    result["decode_benchmarks"] = {}
    for prefix, fields in (
        ("decode-benchmark", ("scope", "median_output_tokens_per_second")),
        ("concurrency", ("scope", "aggregate_http_output_tok_s", "makespan_s", "concurrency", "tokens_per_request")),
    ):
        path = benchmarks / f"{prefix}-mixed-gdn-merge-v1.json"
        if path.exists():
            row = json.loads(path.read_text())
            result["decode_benchmarks"][prefix] = dict(source=str(path), **{key: row[key] for key in fields})
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
