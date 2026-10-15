"""Restore qualified retained owners after the v6 worker failure."""
import dataclasses
import hashlib
import json
import uuid
from pathlib import Path
from tools.glm_perf.resident_control import Control
from tools.glm_perf.resident_native import NativeManifest
from tools.sidecard.qfn_resident_server import QfnResidentClient
ROOT=Path('/srv/ai/storage/artifacts/npu-collective-qfn-20261014-024609')
client=QfnResidentClient('http://127.0.0.1:8001',ROOT/'resident-live-v8/control.token')
output=ROOT/'resident-live-v8/requalification/restore-scheduling.json'
if output.exists(): raise FileExistsError(output)
result={'before':client.rpc('resident_status')}
manifest=json.loads((ROOT/'mamba-commit-live-v1/load-v1.json').read_text())['manifest']
result['native']=client.load_native(NativeManifest(manifest))
base=json.loads((ROOT/'resident-live-v6/requalification/observer-install-v1.json').read_text())['control']
control=Control(**(base|{'generation':uuid.uuid4().hex,'recapture':True}))
result['control']=dataclasses.asdict(control)
output.write_text(json.dumps(result,indent=2)+'\n')
result['installed']=client.switch(control)
output.write_text(json.dumps(result,indent=2)+'\n')
source="""
def operation(worker):
    commit=worker._resident_session().native_resources['qfn_mamba_commit_v4']
    worker.model_runner._qfn_acceptance_reuse=True
    commit.enabled=True
    commit.verify=commit.all_steps=False
    for owner in (commit.device_transition_v1,commit.ple_history_v2,commit.output_observer_v1):
        if owner.failed: raise RuntimeError('failed restored owner')
        owner.enabled=True
        owner.verify=False
    commit.ple_history_v2.early_readback=True
    return dict(rank=worker.rank,passed=True)
class Candidate:
    def validate(self,engine):
        if not engine._pause_complete() or engine.batch_queue or engine.scheduler.get_num_unfinished_requests():
            raise RuntimeError('drain before restoring controls')
        return dict(passed=True)
    def apply(self,engine):
        self.validate(engine)
        rows=engine.model_executor.collective_rpc(operation)
        return dict(passed=len(rows)==4 and all(r.get('passed') for r in rows),workers=rows)
def prepare(engine): return Candidate()
"""
client.request('/pause?mode=wait&clear_cache=false')
revision='restore_'+uuid.uuid4().hex; digest=hashlib.sha256(source.encode()).hexdigest()
result['prepared']=client.request('/qfn_engine_control',dict(operation='prepare',revision=revision,sha256=digest,source=source))
assert result['prepared']['passed']
result['applied']=client.request('/qfn_engine_control',dict(operation='apply',revision=revision,sha256=digest))
output.write_text(json.dumps(result,indent=2)+'\n')
assert result['applied']['passed']
client.resume()
print('Restored transition, PLE and output-observer owners',flush=True)
