"""Restore the qualified v7 optimizations after the dynamic DMA driver reload."""
import subprocess
import sys
from pathlib import Path

ROOT = Path('/srv/ai/storage/artifacts/npu-collective-qfn-20261014-024609')
RUN = ROOT / 'resident-live-v9'

def run(script, *args):
    print('RESTORE', script, *args, flush=True)
    subprocess.run([sys.executable, str(ROOT / script), *args], check=True)

def control(name, action, label):
    run(f'qfn-{name}-control-v9.py', action, '--label', label)

def validate(label):
    run('qsa-resident-validation.py', '--run-root', str(RUN), '--output',
        str(RUN / 'requalification' / f'{label}.json'), '--parallel-requests', '4')

run('restore-v9-driver.py')
control('decode-preparation', 'install', 'restore-install-v1')
validate('decode-preparation-verified')
control('decode-preparation', 'on', 'restore-on-v1')
control('host-commit-metadata', 'on', 'restore-on-v1')
control('checkpoint-preparation', 'install', 'restore-install-v1')
validate('checkpoint-preparation-verified')
control('checkpoint-preparation', 'on', 'restore-on-v1')
control('checkpoint-readback', 'on', 'restore-on-v1')
run('qfn-checkpoint-native-private-v9.py', 'install', '--label', 'restore-v1')
control('checkpoint-native', 'install', 'restore-install-v1')
validate('checkpoint-native-verified')
control('checkpoint-native', 'fused', 'restore-on-v1')
control('device-commit-inputs', 'install', 'restore-install-v1')
validate('device-commit-inputs-verified')
control('device-commit-inputs', 'on', 'restore-on-v1')
run('qfn-scheduling-serving-gate-v9.py', '--label', 'driver-restored-v1')
print('ALL RETAINED OPTIMIZATIONS RESTORED AND VERIFIED', flush=True)
