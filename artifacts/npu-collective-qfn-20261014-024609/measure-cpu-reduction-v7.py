import subprocess,sys
from pathlib import Path
ROOT=Path('/srv/ai/storage/artifacts/npu-collective-qfn-20261014-024609');RUN=ROOT/'resident-live-v7'
for script,args in [
 ('qfn-scheduling-cpu-profile-v7.py',['--label','host-commit-metadata-v1']),
 ('qsa-decode-benchmark.py',['--run-root',str(RUN),'--label','cpu-reduction-v7-final-v1']),
 ('qsa-concurrency-benchmark.py',['--run-root',str(RUN),'--label','cpu-reduction-v7-final-v1','--concurrency','4','--tokens','256']),
 ('qfn-cpu-reduction-final-status-v7.py',[]),
]:
 print(script,flush=True)
 subprocess.run([sys.executable,str(ROOT/script),*args],check=True)
