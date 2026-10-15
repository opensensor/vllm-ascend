import subprocess, sys
from pathlib import Path
ROOT=Path('/srv/ai/storage/artifacts/npu-collective-qfn-20261014-024609')
RUN=ROOT/'resident-live-v7'
def run(script,*args):
    print(script,*args,flush=True)
    subprocess.run([sys.executable,str(ROOT/script),*args],check=True)
run('qsa-resident-validation.py','--run-root',str(RUN),'--output',str(RUN/'requalification/checkpoint-native-fast-v1.json'),'--parallel-requests','4')
run('qfn-scheduling-cpu-profile-v7.py','--label','checkpoint-native-v1')
for mode in ('tensor','fused'):
    run('qfn-checkpoint-native-control-v7.py',mode,'--label',f'benchmark-{mode}-v1')
    run('qsa-decode-benchmark.py','--run-root',str(RUN),'--label',f'checkpoint-native-v7-{mode}-v1')
    run('qsa-concurrency-benchmark.py','--run-root',str(RUN),'--label',f'checkpoint-native-v7-{mode}-v1','--concurrency','4','--tokens','256')
run('qfn-checkpoint-native-control-v7.py','status','--label','final-v1')
print('FUSED CHECKPOINT BENCHMARKS COMPLETE',flush=True)
