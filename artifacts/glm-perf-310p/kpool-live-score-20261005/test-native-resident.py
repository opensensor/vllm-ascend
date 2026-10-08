# SPDX-License-Identifier: Apache-2.0
"""Switch once to live scoring and qualify it; do not restore baseline."""
import json
import uuid
from pathlib import Path
from tools.glm_perf.resident_control import Control
from tools.glm_perf.resident_harness import ResidentClient
from tools.glm_perf.suite import make_groups, run_groups, summarize, retrieval_case, load_tokenizer_json

root=Path(__file__).resolve().parent
client=ResidentClient("http://127.0.0.1:8001",4,1200)
control=Control.from_dict(dict(generation=uuid.uuid4().hex,mode="graph",candidate="native",source=(root/'resident_candidate.py').read_text()))
print("Switching to native selector",flush=True)
receipts=client.switch(control)
(root/'native-switch-receipts.json').write_text(json.dumps(receipts,indent=2))
print("NATIVE ACTIVE; worker and weight identities unchanged",flush=True)
count,tokenizer=load_tokenizer_json(Path('/srv/ai/models/GLM-5.3-Flash-selective-W3-310p/tokenizer.json'))
groups=make_groups(['short','quality','tool'],[])
case=retrieval_case(8192,count_tokens=count)
groups += [('cold_8k',[case]),('warm_8k',[case])]
rows=[]
with (root/'native640-results.jsonl').open('w') as out:
    def record(row):
        row.update(candidate='native',workers=receipts)
        rows.append(row)
        out.write(json.dumps(row)+'\n');out.flush()
        (root/'native640-summary.json').write_text(json.dumps(summarize(rows),indent=2))
        print(json.dumps({k:row.get(k) for k in ('group_id','case_id','ttft_s','decode_tokens_per_s','passed','error')}),flush=True)
    run_groups(groups,client.base_url,'glm53-flash-selective-w3',42,tokenizer_identity=tokenizer,on_result=record,timeout_s=1200)
print('NATIVE QUALIFICATION COMPLETE',flush=True)
