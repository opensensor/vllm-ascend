import json
import sys
import time
import uuid
from pathlib import Path
from tools.glm_perf.resident_control import Control
from tools.glm_perf.resident_harness import ResidentClient
from tools.glm_perf.suite import Case, run_request

candidate, output = map(Path, sys.argv[1:3])
mode = sys.argv[3] if len(sys.argv)>3 else 'graph'
client = ResidentClient('http://127.0.0.1:8001')
source = candidate.read_text() if str(candidate) != 'baseline' else ''
name = candidate.stem if source else 'baseline'
start=time.monotonic()
workers=client.switch(Control(uuid.uuid4().hex, mode, name, source))
print('SWITCH',round(time.monotonic()-start,3),workers,flush=True)
cases=[Case('subtract','quality','What is 100 - 37? Answer only the number.','63',12)]
with output.open('w') as log:
    for case in cases:
        row=run_request(case,client.base_url,'glm53-flash-selective-w3',42,name)
        row['resident']=workers
        log.write(json.dumps(row)+'\n'); log.flush()
        print(case.case_id,row.get('passed'),row.get('finish_reason'),repr(row.get('content')),flush=True)
        if not row['valid']: raise RuntimeError(row.get('error'))
