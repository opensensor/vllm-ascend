import subprocess, sys
from pathlib import Path
ROOT=Path('/srv/ai/storage/artifacts/npu-collective-qfn-20261014-024609')
RUN=ROOT/'resident-live-v7'
def run(script,*args):
    print(script,*args,flush=True)
    subprocess.run([sys.executable,str(ROOT/script),*args],check=True)
run('qsa-resident-validation.py','--run-root',str(RUN),'--output',str(RUN/'requalification/device-commit-inputs-fast-v1.json'),'--parallel-requests','4')
run('qfn-scheduling-serving-gate-v7.py','--label','device-commit-inputs-v1')
run('qfn-scheduling-cpu-profile-v7.py','--label','device-commit-inputs-v1')
for mode in ('off','on'):
    run('qfn-device-commit-inputs-control-v7.py',mode,'--label',f'benchmark-{mode}-v1')
    run('qsa-decode-benchmark.py','--run-root',str(RUN),'--label',f'device-commit-inputs-v7-{mode}-v1')
    run('qsa-concurrency-benchmark.py','--run-root',str(RUN),'--label',f'device-commit-inputs-v7-{mode}-v1','--concurrency','4','--tokens','256')
run('qfn-device-commit-inputs-control-v7.py','status','--label','final-v1')
print('DEVICE COMMIT INPUT BENCHMARKS COMPLETE',flush=True)
