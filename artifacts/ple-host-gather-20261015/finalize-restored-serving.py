AUDIT = "\ndef inspect_worker(worker):\n    commit = worker._resident_session().native_resources['qfn_mamba_commit_v4']\n    names = ('device_transition_v1','ple_history_v2','output_observer_v1',\n             'queued_schedule_v1','compact_table_staging_v1','decode_preparation_v1',\n             'host_commit_metadata_v1','checkpoint_preparation_v1','device_commit_inputs_v1')\n    owners = {name:getattr(commit,name).status() for name in names}\n    owners['gdn_metadata_graphs_v1']=[owner.status() for owner in commit.gdn_metadata_graphs_v1]\n    for name in names:\n        if not owners[name].get('enabled') or owners[name].get('failed',False):\n            raise RuntimeError('inactive or failed retained optimization: '+name)\n    assert commit.enabled and not commit.verify\n    assert all(owner['enabled'] and not owner.get('failed',False) for owner in owners['gdn_metadata_graphs_v1'])\n    assert owners['checkpoint_preparation_v1']['backend']=='fused'\n    assert owners['queued_schedule_v1']['device_drafts']\n    sizes=list(worker.model_runner.compilation_config.cudagraph_capture_sizes)\n    assert sorted(sizes)==[3,6,9,12]\n    import inspect, hashlib\n    from tools.qwen4exp import iteration_transition_runtime\n    fn = iteration_transition_runtime.admit_all_acceptances\n    source = inspect.getsource(fn)\n    assert 'np.arange(1, 4' in source and 'np.nonzero(copy)' in source\n    assert worker.model_runner._qfn_acceptance_reuse\n    assert owners['ple_history_v2']['early_readback']\n    assert owners['checkpoint_preparation_v1']['lazy_readback']\n    return dict(rank=worker.rank,passed=True,owners=owners,capture_sizes=sizes,\n                acceptance_reuse=True, consolidated_admission=dict(file=inspect.getsourcefile(fn),\n                sha256=hashlib.sha256(source.encode()).hexdigest(), names=list(fn.__code__.co_names)))\nclass Candidate:\n    def validate(self,engine):\n        assert engine._pause_complete() and not engine.batch_queue and not engine.scheduler.get_num_unfinished_requests()\n        return dict(passed=True)\n    def apply(self,engine):\n        self.validate(engine)\n        rows=engine.model_executor.collective_rpc(inspect_worker)\n        return dict(passed=len(rows)==4 and all(r['passed'] for r in rows),workers=rows,\n                    engine_scheduler=type(engine.scheduler).__name__,async_scheduling=engine.async_scheduling)\ndef prepare(engine):return Candidate()\n"
import concurrent.futures
import hashlib
import json
import subprocess
import sys
import uuid
from pathlib import Path
from tools.sidecard.qfn_resident_server import QfnResidentClient
ROOT = Path('/srv/ai/storage/artifacts/ple-host-gather-20261015')
RUN = Path('/srv/ai/storage/artifacts/npu-collective-qfn-20261014-024609/resident-live-v13')
client = QfnResidentClient('http://127.0.0.1:8001', RUN/'control.token', timeout=1800)
def control(action,label):
    subprocess.run([sys.executable,'-m','tools.qwen4exp.apply_ple_serving','--run-root',str(RUN),'--action',action,'--output',str(ROOT/label)],check=True,stdout=subprocess.DEVNULL)
control('verify','serving-v2-c2-verify')
def request(index):
    return client.request('/v1/chat/completions',dict(model='qwen38-flash-next',messages=[dict(role='user',content=f'Explain why {index+17} times 23 equals {(index+17)*23}.')],temperature=0,max_tokens=64,min_tokens=64,ignore_eos=True,chat_template_kwargs=dict(enable_thinking=False)))
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
    responses=list(pool.map(request,range(2)))
state=client.rpc('resident_status')
(ROOT/'serving-v2-c2-verified.json').write_text(json.dumps(dict(responses=responses,workers=state),indent=2))
assert all(w['native_ple']['checked']>25 and not w['native_ple']['failed'] for w in state)
control('on','serving-v2-final-on')
client.request('/pause?mode=wait&clear_cache=false')
revision='audit_'+uuid.uuid4().hex
digest=hashlib.sha256(AUDIT.encode()).hexdigest()
prepared=client.request('/qfn_engine_control',dict(operation='prepare',revision=revision,sha256=digest,source=AUDIT))
assert prepared.get('passed'),prepared
result=client.request('/qfn_engine_control',dict(operation='apply',revision=revision,sha256=digest))
(ROOT/'restored-delivery-audit.json').write_text(json.dumps(result,indent=2))
assert result.get('passed'),result
client.resume()
(ROOT/'serving-v2-delivery-status.json').write_text(json.dumps(client.rpc('resident_status'),indent=2))
print('C2 exact verification and complete CPU-reduction audit passed; serving resumed')
