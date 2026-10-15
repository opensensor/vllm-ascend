import subprocess,sys,time,urllib.request
from pathlib import Path
ROOT=Path('/srv/ai/storage/artifacts/npu-collective-qfn-20261014-024609')
for attempt in range(120):
    try:
        with urllib.request.urlopen('http://127.0.0.1:8001/health',timeout=2) as response:
            if response.status==200:break
    except OSError: pass
    time.sleep(5)
else:raise RuntimeError('startup health timeout')
print('SERVICE READY; restoring qualified owners',flush=True)
subprocess.run(['bash',str(ROOT/'requalify-engine-v8.sh')],check=True)
for script,args in [
 ('restore-scheduling-v8.py',[]),
 ('qfn-device-drafts-control-v8.py',['install','--label','restore-install-v1']),
 ('qfn-queued-control-v8.py',['on','--label','restore-on-v1']),
 ('qfn-device-drafts-control-v8.py',['on','--label','restore-on-v1']),
 ('qfn-compact-table-control-v8.py',['on','--label','restore-on-v1']),
 ('qfn-gdn-metadata-control-v8.py',['install','--label','restore-install-v1']),
 ('qsa-resident-validation.py',['--run-root',str(ROOT/'resident-live-v8'),'--output',str(ROOT/'resident-live-v8/requalification/restored-model-validation.json'),'--parallel-requests','4']),
 ('qfn-gdn-metadata-control-v8.py',['on','--label','qualified-on-v1']),
]:
    print(script,flush=True)
    subprocess.run([sys.executable,str(ROOT/script),*args],check=True)
print('RESTORED AND QUALIFIED',flush=True)
