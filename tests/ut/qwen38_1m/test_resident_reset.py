# SPDX-License-Identifier: Apache-2.0
"""Cold measurements reset both scheduler caches and worker checkpoint tiers."""

from copy import deepcopy

import pytest

from tools.qwen4exp.resident_reset import reset_caches


class Client:
    expected_workers = 2

    def __init__(self, *, paused=False, failure=None):
        self.paused = paused
        self.failure = failure
        self.calls = []
        self.workers = [
            {
                "rank": rank,
                "pid": 100 + rank,
                "weight_storage_digest": f"weights-{rank}",
                "generation": "original",
                "mode": "graph",
                "candidate": "native_residual",
                "digest": "source",
                "graphs_dirty": False,
                "native_failed": False,
                "prefix_mamba": {
                    "1": {"resident_checkpoints": 5, "archive_checkpoints": 4, "host_checkpoints": 3, "spill_count": 3}
                },
            }
            for rank in range(2)
        ]

    def rpc(self, method):
        self.calls.append(method)
        if method == "resident_reset":
            assert self.paused
            if self.failure == "rpc":
                raise OSError("failed reset")
            for worker in self.workers:
                for tier in worker["prefix_mamba"].values():
                    for field in ("resident_checkpoints", "archive_checkpoints", "host_checkpoints"):
                        tier[field] = 0
            if self.failure == "identity":
                self.workers[0]["pid"] += 1000
            elif self.failure == "retained":
                self.workers[0]["prefix_mamba"]["1"]["host_checkpoints"] = 1
            elif self.failure == "groups":
                self.workers[0]["prefix_mamba"].clear()
            elif self.failure == "duplicate_rank":
                self.workers[1]["rank"] = 0
        return deepcopy(self.workers)

    def request(self, path, method="POST"):
        self.calls.append(path)
        if path == "/is_paused":
            assert method == "GET"
            return {"is_paused": self.paused}
        assert path == "/pause?mode=wait&clear_cache=true"
        self.paused = True
        return {"status": "paused"}

    def resume(self):
        self.calls.append("resume")
        self.paused = False


@pytest.mark.parametrize("paused", [False, True])
def test_reset_drains_and_verifies_all_worker_tiers(paused):
    client = Client(paused=paused)
    receipt = reset_caches(client)
    assert client.paused == paused and receipt["paused"] == paused
    assert client.calls == [
        "resident_status",
        "/is_paused",
        "/pause?mode=wait&clear_cache=true",
        "resident_reset",
        *([] if paused else ["resume"]),
    ]
    assert receipt["before"][0]["prefix_mamba"]["1"]["host_checkpoints"] == 3
    assert receipt["after"][0]["prefix_mamba"]["1"]["host_checkpoints"] == 0
    assert receipt["after"][0]["prefix_mamba"]["1"]["spill_count"] == 3


@pytest.mark.parametrize("failure", ["rpc", "identity", "retained", "groups", "duplicate_rank"])
def test_failed_reset_stays_paused(failure):
    client = Client(failure=failure)
    with pytest.raises(RuntimeError, match="remains paused"):
        reset_caches(client)
    assert client.paused and "resume" not in client.calls


@pytest.mark.parametrize("failure", ["old_extension", "graphs", "native"])
def test_unprepared_workers_are_rejected_before_mutation(failure):
    client = Client()
    if failure == "old_extension":
        client.workers[0].pop("prefix_mamba")
    elif failure == "graphs":
        client.workers[0]["graphs_dirty"] = True
    else:
        client.workers[0]["native_failed"] = True
    with pytest.raises(RuntimeError):
        reset_caches(client)
    assert client.calls == ["resident_status"]
    assert not client.paused
