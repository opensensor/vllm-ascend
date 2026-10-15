"""Record final engine, worker-owner, graph, and health status without resetting owners."""

import hashlib
import json
import subprocess
import urllib.request
import uuid
from pathlib import Path

from tools.sidecard.qfn_resident_server import QfnResidentClient

ROOT = Path("/srv/ai/storage/artifacts/npu-collective-qfn-20261014-024609")
SOURCE = """
def inspect_worker(worker):
    commit = worker._resident_session().native_resources['qfn_mamba_commit_v4']
    return dict(rank=worker.rank,
        transition=commit.device_transition_v1.status(),
        preparation=commit.decode_preparation_v1.status(),
        checkpoint=commit.checkpoint_preparation_v1.status(),
        host_commit_metadata=commit.host_commit_metadata_v1.status(),
        gdn_metadata=[owner.status() for owner in commit.gdn_metadata_graphs_v1],
        history=commit.ple_history_v2.status(),
        queued=commit.queued_schedule_v1.status(),
        observer=commit.output_observer_v1.status(),
        compact_tables=commit.compact_table_staging_v1.status(),
        worker_async=worker.model_runner.use_async_scheduling,
        profiler_absent=worker.profiler is None,
        native_profiler_absent=getattr(worker, '_qfn_native_profile_session', None) is None,
        cpu_profiler_absent=getattr(worker, '_qfn_cpu_profile', None) is None)
class Candidate:
    def validate(self, engine):
        if not engine._pause_complete() or engine.batch_queue or engine.scheduler.get_num_unfinished_requests():
            raise RuntimeError('drain before final owner inspection')
        return dict(passed=True)
    def apply(self, engine):
        self.validate(engine)
        return dict(passed=True, workers=engine.model_executor.collective_rpc(inspect_worker))
def prepare(engine):
    return Candidate()
"""


def main():
    output = ROOT / "resident-live-v7/final-checkpoint-status.json"
    if output.exists():
        raise FileExistsError(output)
    client = QfnResidentClient("http://127.0.0.1:8001", ROOT / "resident-live-v7/control.token")
    client.request("/pause?mode=wait&clear_cache=false")
    revision, digest = "status_" + uuid.uuid4().hex, hashlib.sha256(SOURCE.encode()).hexdigest()
    prepared = client.request(
        "/qfn_engine_control", dict(operation="prepare", revision=revision, sha256=digest, source=SOURCE)
    )
    assert prepared.get("passed"), prepared
    result = dict(
        owners=client.request("/qfn_engine_control", dict(operation="apply", revision=revision, sha256=digest))
    )
    output.write_text(json.dumps(result, indent=2) + "\n")
    assert result["owners"].get("passed"), result
    owners = result["owners"]["applied"]["workers"]
    assert len(owners) == 4
    for row in owners:
        assert row["checkpoint"]["enabled"] and row["checkpoint"]["lazy_readback"]
        assert not row["checkpoint"]["failed"] and not row["checkpoint"]["verify"]
        assert row["checkpoint"]["checked_blocks"] > 0 and row["checkpoint"]["avoided_syncs"] > 0
        assert row["host_commit_metadata"]["enabled"] and row["host_commit_metadata"]["calls"] > 0
        assert row["preparation"]["enabled"] and not row["preparation"]["verify"]
        assert not row["preparation"]["failed"] and row["preparation"]["verification_error"] is None
        assert row["preparation"]["checked"] > 0
        assert len(row["gdn_metadata"]) == 3
        assert all(o["enabled"] and not o["verify"] and not o["failed"] for o in row["gdn_metadata"])
        assert not row["worker_async"]
        assert row["profiler_absent"] and row["native_profiler_absent"] and row["cpu_profiler_absent"]
        for key in ("transition", "history", "observer"):
            assert row[key]["enabled"] and not row[key]["failed"]
        assert not row["transition"]["pending"] and not row["observer"]["pending"]
        assert row["history"]["early_readback"] and row["compact_tables"]["enabled"]
        assert row["queued"]["enabled"] and row["queued"]["device_drafts"]
        assert not row["queued"]["pending_device_inputs"] and not row["queued"]["pending_device_history"]
    result["workers"] = client.rpc("resident_status")
    assert all(not row["graphs_dirty"] and not row["native_failed"] for row in result["workers"])
    result["graphs"] = client.rpc("qfn_graph_inventory")
    client.resume()
    result["engine"] = client.request("/qfn_engine_control", dict(operation="status"))
    result["paused"] = client.request("/is_paused", method="GET")
    with urllib.request.urlopen(client.base_url + "/health", timeout=10) as response:
        result["health"] = response.status
    result["sensors"] = subprocess.run(
        ["npu-smi", "info"], capture_output=True, text=True, check=True, timeout=20
    ).stdout
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            dict(engine=result["engine"], paused=result["paused"], health=result["health"], owners=owners), indent=2
        )
    )


if __name__ == "__main__":
    main()
