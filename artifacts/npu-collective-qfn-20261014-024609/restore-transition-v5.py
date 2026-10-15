"""Restore the already-qualified worker transition after v5 base requalification.

Never launches or stops a service. Run only after qfn-recovery-v5.py has restored
and validated the combined QSA/W4 candidate and initialized compact mappings.
"""

import dataclasses
import json
import uuid
from pathlib import Path

from tools.glm_perf.resident_control import Control
from tools.glm_perf.resident_native import NativeManifest
from tools.sidecard.qfn_resident_server import QfnResidentClient

ROOT = Path('/srv/ai/storage/artifacts/npu-collective-qfn-20261014-024609')


def main():
    client = QfnResidentClient('http://127.0.0.1:8001', ROOT / 'resident-live-v5/control.token')
    path = ROOT / 'resident-live-v5/requalification/restore-device-transition.json'
    if path.exists():
        raise ValueError('preserve existing recovery evidence')
    result = dict(before=client.rpc('resident_status'))
    base = json.loads((ROOT / 'resident-live-v5/requalification/restore-w4-650-down_proj.json').read_text())['control']
    assert all(rank['digest'] == Control(**base).digest for rank in result['before'])
    assert client.request('/qfn_engine_control', {'operation': 'status'})['passed']
    manifest = json.loads((ROOT / 'mamba-commit-live-v1/load-v1.json').read_text())['manifest']
    result['native'] = client.load_native(NativeManifest(manifest))
    path.write_text(json.dumps(result, indent=2) + '\n')
    previous = json.loads((ROOT / 'device-transition-live-v1/install-v2.json').read_text())['control']
    control = Control(**(previous | {'generation': uuid.uuid4().hex, 'recapture': True}))
    result['control'] = dataclasses.asdict(control)
    result['installed'] = client.switch(control)
    client.request('/pause?mode=wait&clear_cache=false')
    result['acceptance'] = client.rpc('qfn_cpu_profile_stop', True, False)
    result['commit'] = client.rpc('qfn_cpu_profile_stop', json.dumps(dict(enabled=True, verify=False, all_steps=False)))
    result['transition'] = client.rpc('qfn_cpu_profile_stop', json.dumps(dict(transition=True, enabled=True, verify=True)))
    path.write_text(json.dumps(result, indent=2) + '\n')
    assert all(rank.get('passed') for key in ('acceptance', 'commit', 'transition') for rank in result[key])
    result['after'] = client.rpc('resident_status')
    assert all(not rank['native_failed'] and not rank['graphs_dirty'] for rank in result['after'])
    client.resume()
    path.write_text(json.dumps(result, indent=2) + '\n')
    print('Restored worker transition in verification mode; run model gates before disabling diagnostics.')


if __name__ == '__main__':
    main()
