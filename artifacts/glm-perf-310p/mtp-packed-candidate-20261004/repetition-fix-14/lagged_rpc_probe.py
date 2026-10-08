"""Paused recovery after the executor left unequal reply queues on rank failure.

Every mutation is followed by four read-only status barriers, consuming any
older replies before inspecting the stable worker generation. This controller
is solely for the already affected diagnostic process, not the standard client.
"""
import dataclasses
import json
import sys
import uuid
from pathlib import Path
from tools.glm_perf.resident_control import Control
from tools.glm_perf.resident_harness import ResidentClient
from tools.glm_perf.suite import Case, run_request

client=ResidentClient('http://127.0.0.1:8001')
if not client.request('/is_paused',method='GET')['is_paused']:
    client.request('/pause?mode=wait&clear_cache=true')

def barrier(allow_old_errors=False):
    result=None
    for i in range(8 if allow_old_errors else 4):
        try:
            result=client.rpc('resident_status')
        except Exception as error:
            print('DRAIN',i,repr(error),flush=True)
            if not allow_old_errors: raise
    assert result is not None
    assert all('weight_storage_digest' in r for r in result)
    return result

before=barrier(True)
identities={(r['rank'],r['pid'],r['weight_storage_digest']) for r in before}
source=Path(sys.argv[1]).read_text()
setting=Control(uuid.uuid4().hex,sys.argv[3] if len(sys.argv)>3 else 'direct-both',Path(sys.argv[1]).stem,source)
for method,args in (
    ('resident_prepare',[json.dumps(dataclasses.asdict(setting))]),
    ('resident_reset',[]),
    ('resident_apply',[setting.generation]),
    ('resident_capture',[]),
    ('resident_reset',[]),
):
    receipt=client.rpc(method,*args)
    status=barrier()
    print(method,status,flush=True)
    assert {(r['rank'],r['pid'],r['weight_storage_digest']) for r in status}==identities
    if method in ('resident_apply','resident_capture'):
        assert all(r['generation']==setting.generation for r in status)
    if method=='resident_capture':
        assert all(r['graphs_dirty'] is False for r in status)
client.request('/resume')
case=Case('subtract','quality','What is 100 - 37? Answer only the number.','63',12)
row=run_request(case,client.base_url,'glm53-flash-selective-w3',42,setting.candidate)
Path(sys.argv[2]).write_text(json.dumps(row)+'\n')
print(row,flush=True)
