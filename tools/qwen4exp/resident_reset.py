# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Drain and clear scheduler and Qwen worker caches before a cold measurement."""

import argparse
import json

from tools.glm_perf.resident_harness import ResidentClient

IDENTITY_FIELDS = ("rank", "pid", "weight_storage_digest", "generation", "mode", "candidate", "digest", "graphs_dirty")
CHECKPOINT_FIELDS = ("resident_checkpoints", "archive_checkpoints", "host_checkpoints")


def reset_caches(client):
    before = client.rpc("resident_status")
    if any(not isinstance(worker.get("prefix_mamba"), dict) for worker in before):
        raise RuntimeError("Qwen workers must expose prefix_mamba status; update the extension before resetting")
    if any(worker.get("native_failed") or worker.get("graphs_dirty") is not False for worker in before):
        raise RuntimeError("workers require recovery before resetting caches")
    was_paused = client.request("/is_paused", method="GET").get("is_paused")
    if not isinstance(was_paused, bool):
        raise RuntimeError("server did not report its pause state")
    if client.request("/pause?mode=wait&clear_cache=true").get("status") != "paused":
        raise RuntimeError("server did not finish draining requests")
    try:
        after = client.rpc("resident_reset")
        if len({r.get("rank") for r in after}) != client.expected_workers:
            raise RuntimeError("worker ranks are not unique")
        if {tuple(r.get(field) for field in IDENTITY_FIELDS) for r in before} != {
            tuple(r.get(field) for field in IDENTITY_FIELDS) for r in after
        }:
            raise RuntimeError("resident workers, weights, dispatch or graphs changed during reset")
        for worker in after:
            if worker.get("native_failed") or not isinstance(worker.get("prefix_mamba"), dict):
                raise RuntimeError("worker did not report a healthy Mamba reset")
            if worker["prefix_mamba"].keys() != next(
                old["prefix_mamba"].keys() for old in before if old.get("rank") == worker.get("rank")
            ):
                raise RuntimeError("Mamba groups changed during reset")
            for tier in worker["prefix_mamba"].values():
                if any(tier.get(field) != 0 for field in CHECKPOINT_FIELDS):
                    raise RuntimeError("worker retained Mamba checkpoints after reset")
    except Exception as error:
        raise RuntimeError("Qwen cache reset failed; server remains paused") from error
    if not was_paused:
        client.resume()
    return {"before": before, "after": after, "paused": was_paused}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8001")
    parser.add_argument("--expected-workers", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=900)
    args = parser.parse_args()
    if args.expected_workers < 1 or args.timeout <= 0:
        parser.error("expected workers and timeout must be positive")
    print(json.dumps(reset_caches(ResidentClient(args.base_url, args.expected_workers, args.timeout)), indent=2))


if __name__ == "__main__":
    main()
