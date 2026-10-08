"""Compare short scored requests after independent resident resets."""
import json
import sys
import time
import uuid
from pathlib import Path

from tools.glm_perf.resident_control import Control, MODES
from tools.glm_perf.resident_harness import ResidentClient
from tools.glm_perf.suite import Case, run_request

client = ResidentClient('http://127.0.0.1:8001')
cases = [
    Case('sum', 'quality', 'What is 17 + 28? Answer only the number.', '45', 48),
    Case('subtract', 'quality', 'What is 100 - 37? Answer only the number.', '63', 48),
    Case('retrieve', 'quality', 'The access code is BLUE-ORCHID-7319. What is the access code? Reply with only the code.', 'BLUE-ORCHID-7319', 48),
]
initial = client.rpc('resident_status')
identities = {(r['rank'], r['pid'], r['weight_storage_digest']) for r in initial}
with Path(sys.argv[1]).open('w') as output:
    for mode in MODES:
        for case in cases:
            started = time.monotonic()
            workers = client.switch(Control(uuid.uuid4().hex, mode))
            assert {(r['rank'], r['pid'], r['weight_storage_digest']) for r in workers} == identities
            switch_s = time.monotonic() - started
            row = run_request(case, client.base_url, 'glm53-flash-selective-w3', 42, mode)
            row['resident'] = workers
            row['switch_s'] = switch_s
            output.write(json.dumps(row) + '\n')
            output.flush()
            print(mode, case.case_id, row.get('passed'), row.get('finish_reason'), repr(row.get('content')), flush=True)
            if not row['valid']:
                raise RuntimeError(row.get('error'))
    client.switch(Control(uuid.uuid4().hex, 'graph'))
