"""Sequential metadata graph A/B; no overlapping benchmark clients."""
import subprocess
import sys
from pathlib import Path

ROOT = Path('/srv/ai/storage/artifacts/npu-collective-qfn-20261014-024609')
for enabled in (True, False):
    mode = 'on' if enabled else 'off'
    controlled = subprocess.run([sys.executable, str(ROOT/'qfn-commit-admission-control.py'), mode,
        '--label', f'ab-{mode}-v2'], capture_output=True, text=True)
    if controlled.returncode:
        print(controlled.stdout, controlled.stderr, flush=True)
        raise SystemExit(controlled.returncode)
    print('CONTROL', mode, flush=True)
    for script, extra in [('qsa-decode-benchmark.py', []), ('qsa-concurrency-benchmark.py', ['--concurrency','4','--tokens','256'])]:
        subprocess.run([sys.executable,str(ROOT/script),'--run-root',str(ROOT/'resident-live-v6'),
                       '--label',f'commit-admission-{mode}-v2',*extra],check=True)
