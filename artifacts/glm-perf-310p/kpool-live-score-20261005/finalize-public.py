# SPDX-License-Identifier: Apache-2.0
"""Verify the public server and apply the same measured worker affinity."""
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

import psutil

from tools.glm_perf.suite import make_groups, run_groups, summarize
from tools.glm_perf.worker_affinity import apply_bindings, plan_bindings

root=Path(__file__).resolve().parent
pid=int((root/'public311k.pid').read_text())
base='http://192.168.53.187:8001'
for _ in range(900):
    if not psutil.pid_exists(pid):raise RuntimeError('public server exited')
    try:
        with urllib.request.urlopen(base+'/health',timeout=2) as response:
            if response.status==200:break
    except OSError:pass
    time.sleep(2)
else:raise TimeoutError('public startup')
workers=sorted((p for p in psutil.Process(pid).children(recursive=True) if 'Worker_TP' in p.name()),key=lambda p:p.name())
if len(workers)!=4:raise RuntimeError('expected four workers')
masks=[list(range(start,start+4))+list(range(start+32,start+36)) for start in (8,12,16,20)]
plan=plan_bindings(pid,{p.pid:set(cpus) for p,cpus in zip(workers,masks,strict=True)})
apply_bindings(plan)
(root/'affinity-public311k.json').write_text(json.dumps(plan,indent=2))
with urllib.request.urlopen(base+'/v1/models',timeout=10) as response:models=json.load(response)
(root/'public-models.json').write_text(json.dumps(models,indent=2))
try:
    request=urllib.request.Request(base+'/collective_rpc',data=b'{}',headers={'Content-Type':'application/json'})
    urllib.request.urlopen(request,timeout=10)
except urllib.error.HTTPError as exc:
    if exc.code!=404:raise
else:raise RuntimeError('development RPC exposed')
rows=run_groups(make_groups(['fault','fault4'],[]),base,'glm53-flash-selective-w3',42,timeout_s=180)
(root/'public-smoke.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
(root/'public-smoke-summary.json').write_text(json.dumps(summarize(rows),indent=2))
if not all(r.get('valid') and r.get('passed') for r in rows):raise RuntimeError('public smoke failed')
print(json.dumps(dict(pid=pid,models=models,smoke=summarize(rows)),indent=2),flush=True)
