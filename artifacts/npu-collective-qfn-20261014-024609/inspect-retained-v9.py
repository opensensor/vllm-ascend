"""Inspect retained controls through admission only; never apply a mutation."""
import hashlib
import json
import subprocess
import uuid
from pathlib import Path
from tools.sidecard.qfn_resident_server import QfnResidentClient

ROOT=Path('/srv/ai/storage/artifacts/npu-collective-qfn-20261014-024609')
RUN=ROOT/'resident-live-v9'
SOURCE='''
def inspect_worker(worker):
    commit=worker._resident_session().native_resources['qfn_mamba_commit_v4']
    names=('device_transition_v1','ple_history_v2','output_observer_v1',
           'queued_schedule_v1','compact_table_staging_v1','decode_preparation_v1',
           'host_commit_metadata_v1','checkpoint_preparation_v1','device_commit_inputs_v1')
    result={name:getattr(commit,name).status() for name in names}
    result['gdn_metadata_graphs_v1']=[o.status() for o in commit.gdn_metadata_graphs_v1]
    return dict(rank=worker.rank,owners=result)
class Candidate:
    def validate(self,engine):
        rows=engine.model_executor.collective_rpc(inspect_worker)
        return dict(passed=len(rows)==4,workers=rows)
    def apply(self,engine):
        raise RuntimeError('read-only inspection must never be applied')
def prepare(engine):return Candidate()
'''
client=QfnResidentClient('http://127.0.0.1:8001',RUN/'control.token')
output=RUN/'final-driver-restoration.json'
assert not output.exists()
revision='inspect_'+uuid.uuid4().hex
client.request('/pause?mode=wait&clear_cache=false')
try:
    prepared=client.request('/qfn_engine_control',dict(operation='prepare',revision=revision,
        sha256=hashlib.sha256(SOURCE.encode()).hexdigest(),source=SOURCE))
    if prepared.get('passed'):
        discarded=client.request('/qfn_engine_control',dict(operation='discard',revision=revision))
        assert discarded.get('passed')
    assert prepared.get('passed'),prepared
    result=dict(inspection=prepared,workers=client.rpc('resident_status'),graphs=client.rpc('qfn_graph_inventory'))
finally:
    client.resume()
result['engine']=client.request('/qfn_engine_control',dict(operation='status'))
result['paused']=client.request('/is_paused',method='GET')
result['models']=client.request('/v1/models',method='GET')
result['sensors']=subprocess.check_output(['npu-smi','info'],text=True,timeout=20)
output.write_text(json.dumps(result,indent=2)+'\n')
assert not result['engine']['failed'] and not result['paused']['is_paused']
assert result['engine']['batch_queue_size']==2
for worker in prepared['validation']['workers']:
    owners=worker['owners']
    for name,state in owners.items():
        if isinstance(state,list):
            assert all(o['enabled'] and not o['failed'] and not o['verify'] for o in state)
        else:
            assert state['enabled'] and not state.get('failed',False),(name,state)
    assert owners['checkpoint_preparation_v1']['backend']=='fused'
    assert owners['checkpoint_preparation_v1']['lazy_readback']
    assert owners['queued_schedule_v1']['device_drafts']
assert all(not row['graphs_dirty'] and not row['native_failed'] for row in result['workers'])
print(json.dumps(dict(passed=True,workers=len(result['workers']),batch_queue_size=result['engine']['batch_queue_size'],models=result['models'])),flush=True)
