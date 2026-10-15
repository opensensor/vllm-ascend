"""Load/bind the qualified grouped commit while retaining service identity."""
import argparse
import dataclasses
import hashlib
import json
import uuid
from pathlib import Path
from tools.glm_perf.resident_control import Control
from tools.glm_perf.resident_native import NativeManifest
from tools.sidecard.qfn_resident_server import QfnResidentClient

ROOT=Path('/srv/ai/storage/artifacts/npu-collective-qfn-20261014-024609')
BUILD=ROOT/'mamba-commit-build-v4'
OUT=ROOT/'mamba-commit-live-v1'
NAME='qfn_mamba_commit_v4'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('load','install','off','verify','on','all'))
    parser.add_argument('--label', required=True)
    args=parser.parse_args()
    OUT.mkdir(exist_ok=True)
    path=OUT/f'{args.label}.json'
    if Path(args.label).name != args.label or path.exists():
        raise ValueError('use a new plain label')
    c=QfnResidentClient('http://127.0.0.1:8001',ROOT/'resident-live-v4/control.token')
    assert not c.request('/is_paused',method='GET')['is_paused']
    result={'before':c.rpc('resident_status')}
    path.write_text(json.dumps(result,indent=2)+'\n')
    if args.action=='load':
        provenance=json.loads((BUILD/'provenance.json').read_text())
        source='''
import gc
import torch
from tools.qwen4exp.mamba_commit_runtime import ResidentMambaCommit
from vllm_ascend._310p.worker_310p import NPUWorker310

def prepare():
    workers=[o for o in gc.get_objects() if type(o) is NPUWorker310]
    if len(workers)!=1 or workers[0].model_runner.execute_model_state is not None:
        raise RuntimeError('requires one drained resident worker')
    kernel=torch.classes.qwen_mamba_commit_v1.Kernel(
        '/srv/ai/storage/artifacts/npu-collective-qfn-20261014-024609/mamba-commit-build-v4/native_mamba_commit.bin',
        'qwen_mamba_commit_group_v1')
    return ResidentMambaCommit(workers[0].model_runner,kernel,torch.ops.qwen_mamba_commit_v1.launch)

def validate(resource):
    return dict(passed=not resource.failed and sorted(resource.records)==[1,2,3,4], **resource.status())
'''
        module=ROOT/'resident-source-v1/tools/qwen4exp/mamba_commit_runtime.py'
        manifest=NativeManifest(dict(name=NAME,libraries=[provenance['bridge']],
            assets=[*provenance['binaries'].values(),dict(path=str(module),sha256=hashlib.sha256(module.read_bytes()).hexdigest())],
            operators=['qwen_mamba_commit_v1::launch'],validation_source=source))
        result['manifest']=manifest.value
        path.write_text(json.dumps(result,indent=2)+'\n')
        result['loaded']=c.load_native(manifest)
    elif args.action=='install':
        base=json.loads((ROOT/'iteration-closure-v1/switch-install-v3.json').read_text())['control']
        assert all(r['digest']==Control(**base).digest for r in result['before'])
        hashes={name:hashlib.sha256((ROOT/'resident-source-v1/tools/qwen4exp'/name).read_bytes()).hexdigest()
                for name in ('mamba_commit.py','mamba_commit_runtime.py','resident_candidates/qfn_mamba_commit.py')}
        source=base['source']+'\n# commit module hashes: '+json.dumps(hashes,sort_keys=True)+'\n'+'''
commit_base_replacements = replacements
from tools.qwen4exp.resident_candidates import qfn_mamba_commit
importlib.reload(qfn_mamba_commit)
def replacements(native_resources):
    result=commit_base_replacements()
    resource=native_resources[qfn_mamba_commit.NATIVE_NAME]
    result.update(qfn_mamba_commit.make_replacements(result,resource))
    return result
'''
        control=Control(generation=uuid.uuid4().hex,candidate='qsa_combined_native_commit',source=source,recapture=True)
        result['control']=dataclasses.asdict(control)
        path.write_text(json.dumps(result,indent=2)+'\n')
        result['installed']=c.switch(control)
    else:
        request=dict(enabled=args.action!='off',verify=args.action=='verify',all_steps=args.action in ('verify','all'))
        c.request('/pause?mode=wait&clear_cache=false')
        result['ranks']=c.rpc('qfn_cpu_profile_stop',json.dumps(request))
        assert all(r.get('passed') for r in result['ranks'])
        c.resume()
    result['after']=c.rpc('resident_status')
    assert [(r['pid'],r['weight_storage_digest']) for r in result['before']]==[(r['pid'],r['weight_storage_digest']) for r in result['after']]
    assert all(not r['native_failed'] and not r['graphs_dirty'] for r in result['after'])
    path.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(action=args.action,ranks=result.get('ranks'),loaded=result.get('loaded'),pids=[r['pid'] for r in result['after']]),indent=2))

if __name__=='__main__':
    main()
