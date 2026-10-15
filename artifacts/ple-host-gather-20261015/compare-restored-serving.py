"""Matched complete-stack PLE A/B; never reload graphs between treatments."""
import concurrent.futures
import hashlib
import json
import subprocess
import sys
import time
import uuid
from pathlib import Path

from tools.sidecard.qfn_resident_server import QfnResidentClient

OLD = Path('/srv/ai/storage/artifacts/npu-collective-qfn-20261014-024609')
ROOT = Path('/srv/ai/storage/artifacts/ple-host-gather-20261015')
RUN = OLD / 'resident-live-v13'
CLIENT = QfnResidentClient('http://127.0.0.1:8001', RUN / 'control.token', timeout=1800)
AUDIT = '''
def inspect_worker(worker):
    commit = worker._resident_session().native_resources['qfn_mamba_commit_v4']
    names = ('device_transition_v1','ple_history_v2','output_observer_v1',
             'queued_schedule_v1','compact_table_staging_v1','decode_preparation_v1',
             'host_commit_metadata_v1','checkpoint_preparation_v1','device_commit_inputs_v1')
    owners = {name:getattr(commit,name).status() for name in names}
    owners['gdn_metadata_graphs_v1']=[owner.status() for owner in commit.gdn_metadata_graphs_v1]
    for name in names:
        if not owners[name].get('enabled') or owners[name].get('failed',False):
            raise RuntimeError('inactive or failed retained optimization: '+name)
    assert commit.enabled and not commit.verify
    assert all(owner['enabled'] and not owner.get('failed',False) for owner in owners['gdn_metadata_graphs_v1'])
    assert owners['checkpoint_preparation_v1']['backend']=='fused'
    assert owners['queued_schedule_v1']['device_drafts']
    sizes=list(worker.model_runner.compilation_config.cudagraph_capture_sizes)
    assert sorted(sizes)==[3,6,9,12]
    return dict(rank=worker.rank,passed=True,owners=owners,capture_sizes=sizes)
class Candidate:
    def validate(self,engine):
        assert engine._pause_complete() and not engine.batch_queue and not engine.scheduler.get_num_unfinished_requests()
        return dict(passed=True)
    def apply(self,engine):
        self.validate(engine)
        rows=engine.model_executor.collective_rpc(inspect_worker)
        return dict(passed=len(rows)==4 and all(r['passed'] for r in rows),workers=rows,
                    engine_scheduler=type(engine.scheduler).__name__,async_scheduling=engine.async_scheduling)
def prepare(engine):return Candidate()
'''


def audit(label):
    CLIENT.request('/pause?mode=wait&clear_cache=false')
    revision='audit_'+uuid.uuid4().hex
    digest=hashlib.sha256(AUDIT.encode()).hexdigest()
    prepared=CLIENT.request('/qfn_engine_control',dict(operation='prepare',revision=revision,sha256=digest,source=AUDIT))
    assert prepared.get('passed'),prepared
    result=CLIENT.request('/qfn_engine_control',dict(operation='apply',revision=revision,sha256=digest))
    (ROOT/f'{label}-audit.json').write_text(json.dumps(result,indent=2))
    assert result.get('passed'),result
    CLIENT.resume()
    print('AUDIT',label,'passed',flush=True)


def run(*args):
    print('RUN',*map(str,args),flush=True)
    subprocess.run([sys.executable,*map(str,args)],check=True)


def benchmark(label):
    (ROOT/f'{label}-health.txt').write_text(subprocess.check_output(['npu-smi','info'],text=True,timeout=20))
    run(OLD/'qsa-decode-benchmark.py','--run-root',RUN,'--label',label,'--trials','3','--tokens','256')
    run(OLD/'qsa-concurrency-benchmark.py','--run-root',RUN,'--label',label,'--tokens','256','--concurrency','4')


def ple(action,label):
    args=['-m','tools.qwen4exp.apply_ple_serving','--run-root',RUN,'--output',ROOT/label,'--action',action]
    if action=='install':
        args += ['--table',ROOT/'serving-table-v1.json','--library',ROOT/'serving-source-v2/libple_host_gather.so',
                 '--selector-build',OLD/'ple-request-build-v1']
    run(*args)


for _ in range(240):
    log=(OLD/'restore-retained-v13-final.log').read_text()
    if 'ALL RETAINED OPTIMIZATIONS RESTORED AND VERIFIED' in log:break
    if 'Traceback' in log:raise RuntimeError('restoration failed; inspect log')
    time.sleep(5)
else:raise TimeoutError('restoration did not complete')
audit('restored-before-ple')
benchmark('ple-complete-stack-before-v1')
ple('install','serving-v2-install')
# Exercise C4 and its changing request membership while exact lookup checks run.
def verify(index):
    return CLIENT.request('/v1/chat/completions',dict(model='qwen38-flash-next',
        messages=[dict(role='user',content=f'Compute {17+index} times 23, then explain multiplication briefly.')],
        temperature=0,max_tokens=64,min_tokens=64,ignore_eos=True,chat_template_kwargs=dict(enable_thinking=False)))
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
    responses=list(executor.map(verify,range(4)))
state=CLIENT.rpc('resident_status')
(ROOT/'serving-v2-verified.json').write_text(json.dumps(dict(responses=responses,workers=state),indent=2))
assert all(w['native_ple']['checked'] and not w['native_ple']['failed'] for w in state)
ple('on','serving-v2-enable')
audit('restored-with-ple')
benchmark('ple-complete-stack-on-v1')
ple('off','serving-v2-off')
benchmark('ple-complete-stack-off-v1')
ple('on','serving-v2-on-repeat')
benchmark('ple-complete-stack-on-v2')
audit('restored-final')
(ROOT/'serving-v2-final-status.json').write_text(json.dumps(CLIENT.rpc('resident_status'),indent=2))
print('COMPLETE STACK AND MATCHED PLE A/B COMPLETE',flush=True)
