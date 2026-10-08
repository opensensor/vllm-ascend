# SPDX-License-Identifier: Apache-2.0
"""Matched graph selector qualification with resident weight identity checks."""
import json
import time
import urllib.request
import uuid
from pathlib import Path

import psutil

from tools.glm_perf.resident_control import Control
from tools.glm_perf.resident_harness import ResidentClient
from tools.glm_perf.suite import load_tokenizer_json, make_groups, retrieval_case, run_groups, summarize
from tools.glm_perf.worker_affinity import apply_bindings, plan_bindings

root = Path(__file__).resolve().parent
base = "http://127.0.0.1:8001"
server_pid = int((root / "resident640.pid").read_text())
for _ in range(900):
    if not psutil.pid_exists(server_pid):
        raise RuntimeError("server exited during startup")
    try:
        with urllib.request.urlopen(base + "/health", timeout=2) as response:
            if response.status == 200:
                break
    except OSError:
        pass
    time.sleep(2)
else:
    raise TimeoutError("server startup")
workers = sorted((p for p in psutil.Process(server_pid).children(recursive=True) if "Worker_TP" in p.name()), key=lambda p:p.name())
if len(workers) != 4:
    raise RuntimeError(f"expected four workers, found {workers}")
masks = [list(range(start,start+4))+list(range(start+32,start+36)) for start in (8,12,16,20)]
plan = plan_bindings(server_pid, {p.pid:set(cpus) for p,cpus in zip(workers,masks,strict=True)})
apply_bindings(plan)
(root / "affinity-resident640.json").write_text(json.dumps(plan,indent=2))
client = ResidentClient(base,4,1200)
count,tokenizer = load_tokenizer_json(Path("/srv/ai/models/GLM-5.3-Flash-selective-W3-310p/tokenizer.json"))
identity = None
all_rows = []
with (root / "resident640-results.jsonl").open("w") as out:
    for index,name in enumerate(("baseline","native","baseline","native")):
        source = (root / "resident_candidate.py").read_text() if name == "native" else ""
        print(f"SWITCH {index} {name}",flush=True)
        receipts = client.switch(Control.from_dict(dict(generation=uuid.uuid4().hex, mode="graph",candidate=name,source=source)))
        current={(r['rank'],r['pid'],r['weight_storage_digest']) for r in receipts}
        if identity is not None and current != identity:
            raise RuntimeError("resident worker/weight identity changed")
        identity=current
        print(f"READY {index} {name}; weights unchanged",flush=True)
        # First pair uses full 256-token c1/c4; repeat pair checks drift with
        # 32-token streams. Quality and retrieval are measured on both paths.
        groups=make_groups(["short"] if index<2 else ["fault","fault4"],[])
        if index<2:
            groups += make_groups(["quality","tool"],[])
            retrieval=retrieval_case(8192,count_tokens=count)
            groups += [("cold_8k",[retrieval]),("warm_8k",[retrieval])]
        def record(row,name=name,index=index,receipts=receipts):
            row.update(candidate=name,round=index,workers=receipts)
            all_rows.append(row)
            out.write(json.dumps(row)+"\n");out.flush()
            print(json.dumps({k:row.get(k) for k in ('group_id','case_id','candidate','ttft_s','decode_tokens_per_s','passed','error')}),flush=True)
        rows=run_groups(groups,base,"glm53-flash-selective-w3",42,tokenizer_identity=tokenizer,on_result=record,timeout_s=1200)
        if any(r.get('error') for r in rows):raise RuntimeError("request error")
        (root/f"round-{index}-summary.json").write_text(json.dumps(summarize(rows),indent=2))
print("COMPARISON COMPLETE; native remains running",flush=True)
