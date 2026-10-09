# SPDX-License-Identifier: Apache-2.0
"""Compare saved all-rank resident_status receipts, using host counters only."""

import argparse
import json
from pathlib import Path


def _workers(value):
    if not isinstance(value, list) or not value:
        raise ValueError("status receipt must be a nonempty list of worker results")
    workers = {}
    for worker in value:
        if "error" in worker or "rank" not in worker or "pid" not in worker or "transfer_audit" not in worker:
            raise ValueError("worker receipt is incomplete or failed")
        rank = worker["rank"]
        if rank in workers:
            raise ValueError("duplicate worker rank")
        workers[rank] = worker
    return workers


def _ledgers(worker):
    audit = worker["transfer_audit"]
    result = {f"prefix:{key}": value for key, value in audit["prefix_mamba"].items()}
    result.update({f"module:{key}": value for key, value in audit["modules"].items()})
    if audit["runner"] is not None:
        result["runner"] = audit["runner"]
    return result


def compare(before, after, *, expected_ranks=4):
    first, last = _workers(before), _workers(after)
    if set(first) != set(range(expected_ranks)) or set(last) != set(first):
        raise ValueError("all expected ranks must be present in both receipts")
    results = []
    for rank, worker in last.items():
        old_worker = first[rank]
        if worker["pid"] != old_worker["pid"]:
            raise ValueError("worker changed; transfer counters are not comparable")
        for field in ("generation", "candidate", "mode", "digest", "weight_storage_digest"):
            if worker.get(field) != old_worker.get(field):
                raise ValueError("runtime configuration changed during the observation interval")
        old_ledgers, new_ledgers = _ledgers(old_worker), _ledgers(worker)
        if set(old_ledgers) - set(new_ledgers):
            raise ValueError("a previously observed ledger disappeared")
        for name, ledger in new_ledgers.items():
            old = old_ledgers.get(name, {"counts": {}, "sequence": 0})
            if ledger["sequence"] < old["sequence"]:
                raise ValueError("ledger restarted; cannot calculate deltas")
            delta = {
                key: ledger["counts"].get(key, 0) - old["counts"].get(key, 0)
                for key in ledger["counts"].keys() | old["counts"].keys()
            }
            if any(value < 0 for value in delta.values()):
                raise ValueError("cumulative counters decreased")
            results.append(
                {
                    "rank": rank,
                    "pid": worker["pid"],
                    "ledger": name,
                    "delta": delta,
                    "recent_events": [event for event in ledger["events"] if event["sequence"] > old["sequence"]],
                    "retained_event_count": len(ledger["events"]),
                    "dropped_events": ledger["dropped_events"],
                }
            )
    return {
        "results": results,
        "measured_bus_bytes": False,
        "thermal_cause_established": False,
        "graph_replay_task_counts": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare(json.loads(args.before.read_text()), json.loads(args.after.read_text()))
    with args.output.open("x") as stream:
        stream.write(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
