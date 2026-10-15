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
    from types import MethodType
    from tools.qwen4exp import decode_preparation_graph as mod
    owner = worker._resident_session().native_resources['qfn_mamba_commit_v4'].decode_preparation_v1
    runner = owner.runner
    if ACTION == 'verify':
        owner.admission_audit = []
        def audit(self, schedule, counts):
            if len(self.admission_audit) < 4 and self.queue.expected_device_inputs:
                self.admission_audit.append(dict(n=runner.input_batch.num_reqs,
                    async_spec=runner.use_async_spec_decode, multimodal=runner.is_multimodal_model,
                    embeds=bool(runner.input_batch.req_prompt_embeds),
                    queue_matches=self.queue.expected_device_inputs == tuple(runner.input_batch.req_ids),
                    can_draft=mod.can_use_device_drafts(runner,schedule,self.transition,self.history),
                    counts=counts.tolist(), computed=runner.input_batch.num_computed_tokens_cpu[:runner.input_batch.num_reqs].tolist(),
                    totals=[runner.requests[key].num_tokens for key in runner.input_batch.req_ids]))
            return mod.DecodePreparationGraphs.prepare(self,schedule,counts)
        owner.prepare = MethodType(audit, owner)
    else:
        owner.prepare = MethodType(mod.DecodePreparationGraphs.prepare,owner)
    return dict(rank=worker.rank, passed=True, audit=getattr(owner,'admission_audit',[]), owner=owner.status())

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
    output = ROOT / "resident-live-v6/requalification" / f"preparation-audit-{args.label}.json"
    if Path(args.label).name != args.label or output.exists():
        raise ValueError("fresh plain label required")
    source_hash = hashlib.sha256(
        (ROOT / "resident-source-v1/tools/qwen4exp/decode_preparation_graph.py").read_bytes()
    ).hexdigest()
    source = SOURCE.replace("ACTION", repr(args.action)) + "\n# source sha256: " + source_hash
    client = QfnResidentClient("http://127.0.0.1:8001", ROOT / "resident-live-v6/control.token")
    result = dict(source=source)

    def save():
        output.write_text(json.dumps(result, indent=2) + "\n")

    save()
    client.request("/pause?mode=wait&clear_cache=false")
    revision, digest = "preparation_" + uuid.uuid4().hex, hashlib.sha256(source.encode()).hexdigest()
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
