"""Collect compact receipts without copying resident control credentials."""

import hashlib
import json
from pathlib import Path

ROOT = Path("/srv/ai/storage/artifacts/qfn-local-pack-20261016")
BASE = Path("/srv/ai/storage/artifacts/npu-collective-qfn-20261014-024609")


def read(path):
    return json.loads(path.read_text())


def write(name, value):
    (ROOT / name).write_text(json.dumps(value, indent=2) + "\n")


def compact_benchmark(data):
    """Retain measurements and identity checks, omitting repeated runtime dumps."""
    identity_keys = ("rank", "pid", "weight_storage_digest", "graphs_dirty", "native_failed")
    for phase in ("before", "after"):
        data[phase] = [{k: row[k] for k in identity_keys} for row in data[phase]]
    for rank in data.get("graph_inventory", []):
        for graph in rank.get("graphs", []):
            graph.pop("segments", None)
    for row in data.get("trials", []) + data.get("requests", []):
        row.pop("response", None)
    return data


def compact_histograms(value):
    if isinstance(value, dict):
        return {k: compact_histograms(v) for k, v in value.items() if k != "expert_row_histogram"}
    if isinstance(value, list):
        return [compact_histograms(v) for v in value]
    return value


if __name__ == "__main__":
    gates = {}
    for name in ("initial-audit", "install", "local-shadow-status", "local-mixed-on", "final-audit", "delivery-audit"):
        result = read(ROOT / name / "receipt.json")["applied"]["applied"]
        gates[name] = result if name == "local-shadow-status" else compact_histograms(result)
    write("gates.json", gates)
    mixed = read(ROOT / "local-mixed-shadow/receipt.json")
    write(
        "mixed-shadow-result.json",
        {name: mixed[name] for name in ("decode", "prefill", "unpaused", "sensors_after", "profiler_status")},
    )
    result = {}
    for kind in ("decode-benchmark", "concurrency"):
        data = read(BASE / "qsa-resident-v1" / f"{kind}-localpack-v16.json")
        # Preserve measurements without duplicating benchmark-generated output.
        for row in data.get("trials", []) + data.get("requests", []):
            row.pop("response", None)
        result[kind] = compact_benchmark(data)
    write("decode-results.json", result)
    files = [
        ROOT / "native_activation_pack.cpp",
        ROOT / "native_activation_pack_local.bin",
        ROOT / "schedule-m16.cpp",
        ROOT / "schedule-m16.bin",
        Path(
            "/srv/ai/storage/artifacts/ple-host-gather-20261015/serving-source-v2/tools/qwen4exp/w4_activation_pack.py"
        ),
    ]
    write("source-hashes.json", {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files})

    comparisons = {}
    for action in ("off", "on"):
        comparisons[action] = {}
        for kind in ("decode-benchmark", "concurrency"):
            data = read(BASE / "qsa-resident-v1" / f"{kind}-localpack-{action}-v16.json")
            for row in data.get("trials", []) + data.get("requests", []):
                row.pop("response", None)
            comparisons[action][kind] = compact_benchmark(data)
        for phase in ("before", "after"):
            rows = read(ROOT / f"decode-{action}-{phase}" / "receipt.json")["applied"]["applied"]["workers"]
            comparisons[action][phase] = [
                dict(
                    rank=r["rank"],
                    **{
                        k: r["activation"][k] for k in ("calls", "local_calls", "checked", "mismatches", "local_routes")
                    },
                )
                for r in rows
            ]
    write("decode-comparison.json", comparisons)
