"""Summarize resident high-end profiling without copying large profiler traces."""

import json
from pathlib import Path

ROOT = Path("/srv/ai/storage/artifacts/qfn-high-end-20261015")
PROFILES = ROOT.parent / "qfn-mixed-prefill-20261015"
LABELS = (
    "high-end-retained-native-v1",
    "wy-graph-shadow-v1",
    "wy-graph-native-v1",
    "wy-graph-off-a",
    "wy-graph-on-a",
    "wy-graph-off-b",
    "wy-graph-on-b",
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
        "wy-probe-v1",
        "wy-shadow-status-v1",
        "gdn-output-install-v1",
        "final-audit",
        "final-audit-after-tail",
    ):
        path = ROOT / name / "receipt.json"
        if path.exists():
            receipt = json.loads(path.read_text())
            result["gates"][name] = dict(source=str(path), result=receipt.get("applied"), resume=receipt.get("resume"))
    for name in (
        "full-gdn-v2",
        "full-gdn-v3",
        "wy-diagnose-v1",
        "gdn-kernels-v1",
        "gdn-output-gate-v1",
        "gdn-output-qualified-v1",
        "gdn-output-stress-v1",
    ):
        result["gates"][name] = [
            dict(source=str(p), result=json.loads(p.read_text())) for p in sorted((ROOT / name).glob("rank*-gate.json"))
        ]
    benchmarks = ROOT.parent / "npu-collective-qfn-20261014-024609/qsa-resident-v1"
    result["decode_benchmarks"] = {}
    for prefix, fields in (
        ("decode-benchmark", ("scope", "median_output_tokens_per_second")),
        ("concurrency", ("scope", "aggregate_http_output_tok_s", "makespan_s", "concurrency", "tokens_per_request")),
    ):
        path = benchmarks / f"{prefix}-high-end-gdn-v1.json"
        if path.exists():
            row = json.loads(path.read_text())
            if all(key in row for key in fields):
                result["decode_benchmarks"][prefix] = dict(source=str(path), **{key: row[key] for key in fields})
                if "trials" in row:
                    result["decode_benchmarks"][prefix]["trials"] = [
                        {key: trial[key] for key in ("trial", "seconds", "output_tokens_per_second")}
                        for trial in row["trials"]
                    ]
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
