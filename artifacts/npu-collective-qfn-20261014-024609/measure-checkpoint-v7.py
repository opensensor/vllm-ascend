import subprocess, sys
from pathlib import Path
ROOT=Path('/srv/ai/storage/artifacts/npu-collective-qfn-20261014-024609')
RUN=ROOT/'resident-live-v7'
def run(script,*args):
    print(script,*args,flush=True)
    subprocess.run([sys.executable,str(ROOT/script),*args],check=True)
run('qfn-scheduling-cpu-profile-v7.py','--label','checkpoint-lazy-v1')
for mode in ('off','on'):
    run('qfn-checkpoint-preparation-control-v7.py',mode,'--label',f'benchmark-{mode}-v1')
    run('qsa-decode-benchmark.py','--run-root',str(RUN),'--label',f'checkpoint-v7-{mode}-v1')
    run('qsa-concurrency-benchmark.py','--run-root',str(RUN),'--label',f'checkpoint-v7-{mode}-v1','--concurrency','4','--tokens','256')
run('qfn-checkpoint-readback-control-v7.py','status','--label','final-v1')
print('CHECKPOINT BENCHMARKS COMPLETE',flush=True)
