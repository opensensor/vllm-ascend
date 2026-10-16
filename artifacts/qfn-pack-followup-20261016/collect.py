"""Export compact recovery evidence without control tokens or weight manifests."""

import hashlib
import json
from pathlib import Path

ROOT = Path("/srv/ai/storage/artifacts/qfn-pack-followup-20261016")
BASE = Path("/srv/ai/storage/artifacts/npu-collective-qfn-20261014-024609")
TOOLS = Path("/srv/ai/storage/artifacts/ple-host-gather-20261015/serving-source-v2")


def read(path):
    return json.loads(path.read_text())


def save(name, value):
    (ROOT / name).write_text(json.dumps(value, indent=2) + "\n")


def main():
    gates = {}
    for path in sorted(ROOT.glob("*/receipt.json")):
        receipt = read(path)
        result = receipt.get("applied", {})
        gates[path.parent.name] = dict(
            prepared=receipt.get("prepared", {}).get("passed"),
            applied=result.get("applied"),
            failed=result.get("status", {}).get("failed"),
            resumed=receipt.get("resume"),
            preparation_error=receipt.get("prepared", {}).get("error"),
        )
    save("gates.json", gates)
    profiles = {}
    for label in ("retained-native-v16", "pack32-mixed-shadow"):
        path = ROOT / label / "receipt.json"
        if not path.exists():
            continue
        receipt = read(path)
        profiles[label] = {
            key: receipt.get(key)
            for key in ("start", "stop", "profiler_status", "unpaused", "sensors_before", "sensors_after")
        }
        for key in ("decode", "prefill"):
            if key in receipt:
                profiles[label][key] = {k: v for k, v in receipt[key].items() if k != "events"}
    save("profiles.json", profiles)
    hashes = {}
    for path in (
        ROOT / "native_activation_pack.cpp",
        ROOT / "native_activation_pack_v4.bin",
        ROOT / "native_activation_pack_final.bin",
        TOOLS / "tools/qwen4exp/w4_activation_pack.py",
        TOOLS / "tools/qwen4exp/resident_candidates/qfn_route_finalize.py",
        TOOLS / "tools/qwen4exp/queued_schedule.py",
        TOOLS / "tools/qwen4exp/mixed_prefill_budget.py",
    ):
        hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    save("source-hashes.json", hashes)
    decode = read(BASE / "qsa-resident-v1/decode-benchmark-pack32-v16.json")
    concurrent = read(BASE / "qsa-resident-v1/concurrency-pack32-v16.json")
    save(
        "decode-results.json",
        dict(
            single=dict(
                scope=decode["scope"],
                median_output_tokens_per_second=decode["median_output_tokens_per_second"],
                trials=[{k: v for k, v in row.items() if k != "response"} for row in decode["trials"]],
            ),
            concurrent={
                k: v
                for k, v in concurrent.items()
                if k not in ("before", "after", "requests", "graph_inventory", "metrics_before", "metrics_after")
            },
        ),
    )


if __name__ == "__main__":
    main()
