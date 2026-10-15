"""Install/control the device transition bridge without changing service identity."""

import argparse
import dataclasses
import hashlib
import json
import uuid
from pathlib import Path

from tools.glm_perf.resident_control import Control
from tools.sidecard.qfn_resident_server import QfnResidentClient

ROOT = Path('/srv/ai/storage/artifacts/npu-collective-qfn-20261014-024609')
OUT = ROOT / 'device-transition-live-v1'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('install', 'refresh', 'on', 'off', 'verify'))
    parser.add_argument('--label', required=True)
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)
    path = OUT / f'{args.label}.json'
    if Path(args.label).name != args.label or path.exists():
        raise ValueError('use a new plain label')
    client = QfnResidentClient('http://127.0.0.1:8001', ROOT / 'resident-live-v4/control.token')
    assert not client.request('/is_paused', method='GET')['is_paused']
    result = {'before': client.rpc('resident_status')}
    path.write_text(json.dumps(result, indent=2) + '\n')
    if args.action in ('install', 'refresh'):
        base = json.loads((ROOT / 'mamba-commit-live-v1/install-v1.json').read_text())['control']
        expected = base if args.action == 'install' else json.loads((OUT / 'install-v1.json').read_text())['control']
        assert all(rank['digest'] == Control(**expected).digest for rank in result['before'])
        files = ('iteration_transition.py', 'iteration_transition_runtime.py',
                 'resident_candidates/qfn_device_transition.py', 'resident_candidates/qfn_acceptance_snapshot.py')
        hashes = {name: hashlib.sha256((ROOT / 'resident-source-v1/tools/qwen4exp' / name).read_bytes()).hexdigest()
                  for name in files}
        source = base['source'] + '\n# transition module hashes: ' + json.dumps(hashes, sort_keys=True) + '\n' + '''
transition_base_replacements = replacements
from tools.qwen4exp import iteration_transition_runtime
from tools.qwen4exp.resident_candidates import qfn_device_transition
importlib.reload(qfn_device_transition)
def replacements(native_resources):
    result = transition_base_replacements(native_resources)
    commit = native_resources[qfn_mamba_commit.NATIVE_NAME]
    resource = getattr(commit, 'device_transition_v1', None)
    if resource is None:
        # Publish ownership before any native graph construction can fail.
        resource = iteration_transition_runtime.ResidentDeviceTransition.__new__(
            iteration_transition_runtime.ResidentDeviceTransition)
        commit.device_transition_v1 = resource
        resource.failed = True
        resource.__init__(commit)
        resource.construction_complete = True
    if not getattr(resource, 'construction_complete', False) or resource.failed:
        raise RuntimeError('device transition construction failed; retain owners')
    from vllm_ascend._310p.model_runner_310p import NPUModelRunner310
    result.update(qfn_device_transition.make_replacements(result, resource, NPUModelRunner310))
    return result
'''
        control = Control(generation=uuid.uuid4().hex, candidate='qsa_combined_device_transition',
                          source=source, recapture=True)
        result['control'] = dataclasses.asdict(control)
        path.write_text(json.dumps(result, indent=2) + '\n')
        result['installed'] = client.switch(control)
    else:
        payload = dict(transition=True, enabled=args.action != 'off', verify=args.action == 'verify')
        client.request('/pause?mode=wait&clear_cache=false')
        result['ranks'] = client.rpc('qfn_cpu_profile_stop', json.dumps(payload))
        if not all(rank.get('passed') for rank in result['ranks']):
            path.write_text(json.dumps(result, indent=2) + '\n')
            raise RuntimeError('control failed; retain drained service for inspection')
        client.resume()
    result['after'] = client.rpc('resident_status')
    assert [(r['pid'], r['weight_storage_digest']) for r in result['before']] == [
        (r['pid'], r['weight_storage_digest']) for r in result['after']]
    assert all(not r['native_failed'] and not r['graphs_dirty'] for r in result['after'])
    path.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(action=args.action, ranks=result.get('ranks'), pids=[r['pid'] for r in result['after']]), indent=2))


if __name__ == '__main__':
    main()
