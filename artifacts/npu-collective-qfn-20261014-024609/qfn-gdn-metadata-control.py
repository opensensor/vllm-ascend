"""Cold qualification and controls for GDN metadata graphs."""

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
    from tools.qwen4exp import gdn_metadata_graph
    from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata
    importlib.reload(gdn_metadata_graph)
    commit = worker._resident_session().native_resources['qfn_mamba_commit_v4']
    if commit.device_transition_v1.pending is not None or worker.model_runner.execute_model_state is not None:
        raise RuntimeError('drain worker before changing metadata graph ownership')
    torch.npu.synchronize()
    owners = getattr(commit, 'gdn_metadata_graphs_v1', None)
    if ACTION == 'install':
        if owners is not None:
            raise RuntimeError('metadata graph owners already exist')
        owners = []
        commit.gdn_metadata_graphs_v1 = owners
        for gid, groups in enumerate(worker.model_runner.attn_groups):
            for group in groups:
                builder = group.get_metadata_builder(0)
                if type(builder).__name__ != 'GDNAttentionMetadataBuilder310':
                    continue
                owner = gdn_metadata_graph.GDNMetadataGraphs.__new__(gdn_metadata_graph.GDNMetadataGraphs)
                owners.append(owner)
                owner.__init__(worker.model_runner, gid, builder, GDNAttentionMetadata)
        if len(owners) != 3:
            raise RuntimeError('expected three retained GDN metadata owners')
        for owner in owners:
            owner.install()
            owner.enabled = owner.verify = True
    elif owners is None:
        raise RuntimeError('install metadata graph owners first')
    elif ACTION != 'status':
        for owner in owners:
            if owner.failed:
                raise RuntimeError('failed metadata owner must remain retained')
            owner.enabled = ACTION != 'off'
            owner.verify = ACTION == 'verify'
    return dict(rank=worker.rank, passed=True, owners=[owner.status() for owner in owners])

class Candidate:
    def validate(self, engine):
        if not engine._pause_complete() or engine.batch_queue or engine.scheduler.get_num_unfinished_requests():
            raise RuntimeError('drain engine and workers before changing metadata graphs')
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
    parser.add_argument("action", choices=("install", "on", "off", "verify", "status"))
    parser.add_argument("--label", required=True)
    args = parser.parse_args()
    output = ROOT / "resident-live-v6/requalification" / f"gdn-metadata-{args.label}.json"
    if Path(args.label).name != args.label or output.exists():
        raise ValueError("fresh plain label required")
    source_hash = hashlib.sha256(
        (ROOT / "resident-source-v1/tools/qwen4exp/gdn_metadata_graph.py").read_bytes()
    ).hexdigest()
    source = SOURCE.replace("ACTION", repr(args.action)) + "\n# source sha256: " + source_hash
    client = QfnResidentClient("http://127.0.0.1:8001", ROOT / "resident-live-v6/control.token")
    result = dict(source=source)

    def save():
        output.write_text(json.dumps(result, indent=2) + "\n")

    save()
    client.request("/pause?mode=wait&clear_cache=false")
    revision, digest = "metadata_" + uuid.uuid4().hex, hashlib.sha256(source.encode()).hexdigest()
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
