"""Drain-controlled compact-table staging; preserves all graph owners."""

import argparse
import hashlib
import json
import uuid
from pathlib import Path

from tools.sidecard.qfn_resident_server import QfnResidentClient

ROOT = Path("/srv/ai/storage/artifacts/npu-collective-qfn-20261014-024609")
SOURCE = """
def refresh_worker(worker):
    import importlib
    import torch
    from tools.qwen4exp.resident_candidates import qfn_compact_table_staging
    from vllm_ascend._310p.model_runner_310p import NPUModelRunner310
    importlib.reload(qfn_compact_table_staging)
    commit = worker._resident_session().native_resources['qfn_mamba_commit_v4']
    if commit.device_transition_v1.pending is not None or worker.model_runner.execute_model_state is not None:
        raise RuntimeError('drain worker before changing compact-table uploads')
    torch.npu.synchronize()
    owner = getattr(commit, 'compact_table_staging_v1', None)
    if owner is None:
        owner = qfn_compact_table_staging.CompactTableStaging(worker.model_runner)
        commit.compact_table_staging_v1 = owner
    remap = qfn_compact_table_staging.build_remap(NPUModelRunner310._remap_compact_mamba_block_tables, owner)
    NPUModelRunner310._remap_compact_mamba_block_tables = remap
    owner.enabled = ENABLED
    return dict(rank=worker.rank, passed=True, **owner.status())

class Candidate:
    def validate(self, engine):
        if not engine._pause_complete() or engine.batch_queue or engine.scheduler.get_num_unfinished_requests():
            raise RuntimeError('drain engine and workers before changing compact-table uploads')
        return dict(passed=True)
    def apply(self, engine):
        self.validate(engine)
        rows = engine.model_executor.collective_rpc(refresh_worker)
        return dict(passed=len(rows) == 4 and all(row.get('passed') for row in rows), workers=rows)
def prepare(engine):
    return Candidate()
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("on", "off"))
    parser.add_argument("--label", required=True)
    args = parser.parse_args()
    output = ROOT / "resident-live-v9/requalification" / f"compact-tables-{args.label}.json"
    if Path(args.label).name != args.label or output.exists():
        raise ValueError("fresh plain label required")
    source_hash = hashlib.sha256(
        (ROOT / "resident-source-v1/tools/qwen4exp/resident_candidates/qfn_compact_table_staging.py").read_bytes()
    ).hexdigest()
    source = SOURCE.replace("ENABLED", repr(args.action == "on")) + "\n# source sha256: " + source_hash
    client = QfnResidentClient("http://127.0.0.1:8001", ROOT / "resident-live-v9/control.token")
    result = dict(source=source)

    def save():
        output.write_text(json.dumps(result, indent=2) + "\n")

    save()
    client.request("/pause?mode=wait&clear_cache=false")
    revision, digest = "tables_" + uuid.uuid4().hex, hashlib.sha256(source.encode()).hexdigest()
    result["prepared"] = client.request(
        "/qfn_engine_control", dict(operation="prepare", revision=revision, sha256=digest, source=source)
    )
    save()
    assert result["prepared"].get("passed")
    result["applied"] = client.request("/qfn_engine_control", dict(operation="apply", revision=revision, sha256=digest))
    save()
    assert result["applied"].get("passed")
    client.resume()
    print(json.dumps(result["applied"], indent=2))


if __name__ == "__main__":
    main()
