"""Summarize resident high-end profiling without copying large profiler traces."""

import json
from pathlib import Path

ROOT = Path("/srv/ai/storage/artifacts/qfn-device-targets-20261015")
PROFILES = ROOT.parent / "qfn-mixed-prefill-20261015"
LABELS = (
    "device-targets-native-v2",
    "qsa-table-large-only",
    "qsa-table-shadow-v1",
    "qsa-table-shadow-v2",
    "qsa-table-off-a",
    "qsa-table2-off-a",
    "qsa-table2-on-a",
    "qsa-table2-off-b",
    "qsa-table2-on-b",
)


def main():
    result = dict(runs={}, gates={})
    for label in LABELS:
        path = PROFILES / label / "receipt.json"
        if not path.exists():
            continue
        receipt = json.loads(path.read_text())
        row = dict(source=str(path), unpaused=receipt.get("unpaused"), mode=receipt["mode"])
        for kind in ("decode", "prefill"):
            if kind in receipt:
                row[kind] = {
                    key: receipt[kind][key] for key in ("usage", "ttft_s", "max_content_gap_s", "median_content_gap_s")
                }
        result["runs"][label] = row
    for name in (
        "shared-gate-v3",
        "shared-install",
        "shared-finalize",
        "qsa-gate",
        "compare2-off-a",
        "compare2-on-a",
        "compare2-off-b",
        "compare2-on-b",
        "final-audit",
        "final-audit-after-c4",
        "decode-check-off",
        "decode-check-on",
        "qsa-large-prefill",
        "final-audit-large-only",
        "final-audit-retained",
    ):
        path = ROOT / name / "receipt.json"
        if path.exists():
            receipt = json.loads(path.read_text())
            result["gates"][name] = dict(source=str(path), result=receipt.get("applied"), resume=receipt.get("resume"))
    benchmarks = ROOT.parent / "npu-collective-qfn-20261014-024609/qsa-resident-v1"
    result["decode_benchmarks"] = {}
    for prefix, fields in (
        ("decode-benchmark", ("scope", "median_output_tokens_per_second")),
        ("concurrency", ("scope", "aggregate_http_output_tok_s", "makespan_s", "concurrency", "tokens_per_request")),
    ):
        path = benchmarks / f"{prefix}-device-targets-v1.json"
        if path.exists():
            row = json.loads(path.read_text())
            if all(key in row for key in fields):
                result["decode_benchmarks"][prefix] = dict(source=str(path), **{key: row[key] for key in fields})
                if "trials" in row:
                    result["decode_benchmarks"][prefix]["trials"] = [
                        {key: trial[key] for key in ("trial", "seconds", "output_tokens_per_second")}
                        for trial in row["trials"]
                    ]
    result["c4_followup"] = {}
    for mode in ("off", "on", "large-only"):
        path = benchmarks / f"concurrency-device-targets-c4-{mode}.json"
        if path.exists():
            row = json.loads(path.read_text())
            result["c4_followup"][mode] = dict(
                source=str(path),
                aggregate_http_output_tok_s=row["aggregate_http_output_tok_s"],
                makespan_s=row["makespan_s"],
                median_client_decode_tok_s=row["median_client_decode_tok_s"],
            )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
