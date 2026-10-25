# SPDX-License-Identifier: Apache-2.0
"""Activate a bitwise-qualified native kernel without discarding request caches."""

import argparse
import dataclasses
import fcntl
import json
import uuid
from pathlib import Path

from tools.glm_perf.resident_control import Control
from tools.glm_perf.resident_harness import ResidentClient
from tools.glm_perf.resident_native import NativeManifest
from tools.minimax_m3.resident import candidate_source

IDENTITY_FIELDS = ("rank", "pid", "weight_storage_digest", "generation", "digest")


def identities(workers):
    return sorted(tuple(row[key] for key in IDENTITY_FIELDS) for row in workers)


def validate_evidence(evidence, manifest, workers):
    if not evidence["passed"] or evidence["candidate"] != manifest.name:
        raise ValueError("candidate lacks passing evidence")
    if identities(evidence["after"]) != identities(workers):
        raise ValueError("evidence does not describe the current resident generation")
    if len(evidence["workers"]) != len(workers):
        raise ValueError("incomplete worker evidence")
    source = candidate_source(native_name=evidence["baseline"])
    restore = Control(workers[0]["generation"], "direct-target", workers[0]["candidate"], source)
    for row, measured in zip(
        sorted(workers, key=lambda w: w["rank"]), sorted(evidence["workers"], key=lambda w: w["rank"])
    ):
        proof = measured["benchmark"]
        native = row["native_loaded"].get(manifest.name, {})
        if (
            row["rank"] != measured["rank"]
            or row["native_failed"]
            or row["graphs_dirty"]
            or row["digest"] != restore.digest
            or native.get("native_digest") != manifest.digest
            or not native.get("validation", {}).get("passed")
            or not proof["passed"]
            or not proof["kv_storage_unchanged"]
            or not proof.get("byte_reuse", {}).get("passed")
            or not proof["cases"]
            or any(case["max_abs"] != 0 or not case["passed"] for case in proof["cases"])
        ):
            raise ValueError("native kernel lacks complete bitwise qualification on healthy workers")
    return restore


def activate(client, manifest, evidence):
    before = client.rpc("resident_status")
    restore = validate_evidence(evidence, manifest, before)
    if client.request("/is_paused", method="GET")["is_paused"]:
        raise ValueError("refusing to take ownership of an existing pause")
    control = Control(uuid.uuid4().hex, "direct-target", "m3_candidate", candidate_source(native_name=manifest.name))
    applied = False
    try:
        client.request("/pause?mode=wait&clear_cache=false")
        if identities(client.rpc("resident_status")) != identities(before):
            raise RuntimeError("resident generation changed while draining")
        staged = client.rpc("resident_prepare", json.dumps(dataclasses.asdict(control)))
        if any(row["digest"] != control.digest for row in staged):
            raise RuntimeError("workers disagree on candidate")
        applied = True
        client.rpc("resident_apply", control.generation)
        after = client.rpc("resident_status")
        if any(row["digest"] != control.digest or row["native_failed"] or row["graphs_dirty"] for row in after) or [
            r[:3] for r in identities(before)
        ] != [r[:3] for r in identities(after)]:
            raise RuntimeError("native activation did not preserve resident weights")
    except BaseException:
        if applied:
            client.rpc("resident_prepare", json.dumps(dataclasses.asdict(restore)))
            client.rpc("resident_apply", restore.generation)
        current = client.rpc("resident_status")
        if identities(current) == identities(before) and all(
            not w["native_failed"] and not w["graphs_dirty"] for w in current
        ):
            client.resume()
        raise
    client.resume()
    return {"before": before, "after": after, "native_digest": manifest.digest, "cache_reset": False, "passed": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    args = parser.parse_args()
    with Path("/tmp/minimax-m31-resident.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = activate(
            ResidentClient(args.url, 6),
            NativeManifest(json.loads(args.manifest.read_text())),
            json.loads(args.evidence.read_text()),
        )
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"passed": result["passed"], "cache_reset": False}))


if __name__ == "__main__":
    main()
