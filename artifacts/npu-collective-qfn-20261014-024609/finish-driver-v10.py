"""Verify the retained serving state once restoration has finished."""
import json
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path('/srv/ai/storage/artifacts/npu-collective-qfn-20261014-024609')
RUN = ROOT / 'resident-live-v10'

def cool():
    deadline = time.monotonic() + 1800
    while time.monotonic() < deadline:
        text = subprocess.check_output(['npu-smi', 'info'], text=True, timeout=15)
        temps = [int(x) for x in re.findall(r'310P3\s*\|\s*OK\s*\|\s*\S+\s+(\d+)', text)]
        if len(temps) != 4:
            raise RuntimeError('cannot verify four healthy NPU temperatures')
        if max(temps) <= 79:
            print('TEMPERATURE READY', temps, flush=True)
            return
        print('COOLING', temps, flush=True)
        time.sleep(10)
    raise RuntimeError('cooling deadline exceeded')

def run(name, *args):
    subprocess.run([sys.executable, str(ROOT / name), *args], check=True)

deadline = time.monotonic() + 1800
while 'ALL RETAINED OPTIMIZATIONS RESTORED AND VERIFIED' not in (ROOT / 'restore-retained-v10.log').read_text():
    if time.monotonic() > deadline:
        raise RuntimeError('restoration deadline exceeded')
    time.sleep(5)
run('inspect-retained-v10.py')
cool()
run('qsa-decode-benchmark.py', '--run-root', str(RUN), '--label', 'driver-source-v10-v1', '--trials', '3', '--tokens', '256')
cool()
run('qsa-concurrency-benchmark.py', '--run-root', str(RUN), '--label', 'driver-source-v10-v1', '--concurrency', '4', '--tokens', '256')
print('FINAL VERIFICATION COMPLETE', flush=True)
