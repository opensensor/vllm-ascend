"""Collect compact evidence without control tokens, weight manifests, or raw traces."""

import hashlib
import json
from pathlib import Path

ROOT = Path("/srv/ai/storage/artifacts/qfn-hc-queue-20261015")
PROFILES = Path("/srv/ai/storage/artifacts/qfn-mixed-prefill-20261015")


def read(path):
    return json.loads(path.read_text())


def main():
    measurements = {}
    sensors = {}
    for pattern in ("hc-queue-*", "hc-shared-*"):
        for directory in sorted(PROFILES.glob(pattern)):
            streams = {}
            for name in ("prefill", "decode"):
                path = directory / f"{name}-stream.json"
                if path.exists():
                    streams[name] = {k: v for k, v in read(path).items() if k != "events"}
            if streams:
                measurements[directory.name] = streams
            receipt_path = directory / "receipt.json"
            if receipt_path.exists():
                receipt = read(receipt_path)
                sensors[directory.name] = {
                    key: receipt.get(key) for key in ("sensors_before", "sensors_after", "unpaused")
                }
    gates = {}
    for path in sorted(ROOT.glob("*/receipt.json")):
        receipt = read(path)
        prepared, applied = receipt.get("prepared"), receipt.get("applied")
        if prepared is not None:
            prepared.pop("revision", None)
        if applied is not None and "status" in applied:
            status = applied["status"]
            status["has_pending_revision"] = status.pop("pending", None) is not None
            status.pop("current", None)
        gates[path.parent.name] = dict(prepared=prepared, applied=applied, resumed=receipt.get("resume"))
    recovery = read(ROOT / "recovery.json")
    gates["private_gate_recovery"] = dict(
        recovered=recovery["recovered"],
        private=recovery["private"],
        retained_audit_passed=recovery["corrected_audit"]["passed"],
    )
    for name, result in (
        ("measurements.json", measurements),
        ("gates.json", gates),
        ("thermal-readings.json", sensors),
    ):
        (ROOT / name).write_text(json.dumps(result, indent=2) + "\n")
    provenance = {}
    for name in ("hc_epilogues.py", "queued_schedule.py", "mixed_prefill_budget.py"):
        path = ROOT / name
        if path.exists():
            provenance[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    (ROOT / "source-hashes.json").write_text(json.dumps(provenance, indent=2) + "\n")


if __name__ == "__main__":
    main()
