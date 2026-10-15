import subprocess,sys,json
from pathlib import Path
ROOT=Path('/srv/ai/storage/artifacts/npu-collective-qfn-20261014-024609')
RUN=ROOT/'resident-live-v7'
def run(script,*args):
    print(script,*args,flush=True)
    subprocess.run([sys.executable,str(ROOT/script),*args],check=True)
run('qfn-decode-preparation-control-v7.py','install','--label','install-v2')
run('qsa-resident-validation.py','--run-root',str(RUN),'--output',str(RUN/'requalification/preparation-verified-v2.json'),'--parallel-requests','4')
run('qfn-decode-preparation-control-v7.py','on','--label','qualified-on-v2')
run('qsa-resident-validation.py','--run-root',str(RUN),'--output',str(RUN/'requalification/preparation-fast-v2.json'),'--parallel-requests','4')
run('qfn-scheduling-serving-gate-v7.py','--label','preparation-v2')
run('qfn-scheduling-cpu-profile-v7.py','--label','preparation-v2')
for mode in ('off','on'):
    run('qfn-decode-preparation-control-v7.py',mode,'--label',f'ab-{mode}-v2')
    run('qsa-decode-benchmark.py','--run-root',str(RUN),'--label',f'preparation-v7-{mode}-v2')
    run('qsa-concurrency-benchmark.py','--run-root',str(RUN),'--label',f'preparation-v7-{mode}-v2','--concurrency','4','--tokens','256')
run('qfn-preparation-final-status-v7.py')
print('PREPARATION QUALIFIED AND RETAINED',flush=True)
