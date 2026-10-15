"""Read retained private probe state through a preparation-only diagnostic."""
import dataclasses
import json
import uuid
from pathlib import Path
from tools.glm_perf.resident_control import Control
from tools.sidecard.qfn_resident_server import QfnResidentClient

ROOT = Path('/srv/ai/storage/artifacts/npu-collective-qfn-20261014-024609')
source = '''
import gc
import json

def replacements():
    matches = []
    for obj in gc.get_objects():
        if type(obj).__name__ != "NPUWorker310":
            continue
        state = getattr(obj, "_qfn_prefill_collective_probe", None)
        if state is None:
            continue
        private = state.get("private")
        owners = []
        if private is not None:
            for key, owner in private.owners.items():
                owners.append(dict(schema=key, state=owner.state, calls=owner.calls,
                    allocations=len(owner.allocations), allocation_nonnull=[bool(p.value) for p in owner.allocations], plan_exists=bool(owner.plan.value),
                    edges=[dict(peer=peer, local=list(edge["local"]), exported=list(edge["exported"]),
                                imported=list(edge["imported"]), handle_exists=bool(edge["handle"].value))
                           for peer, edge in owner.edges.items()]))
        matches.append(dict(rank=obj.rank, state=state.get("state"), errors=state.get("errors"),
                            private_state=getattr(private,"state",None), owners=owners,
                            completed=state.get("completed")))
    raise RuntimeError("READ_ONLY_PRIVATE_STATE=" + json.dumps(matches))
'''
client = QfnResidentClient('http://127.0.0.1:8001', ROOT/'resident-live-v3/control.token')
control = Control(generation=uuid.uuid4().hex, candidate='private_state_read_only', source=source, recapture=False)
result = client.request('/collective_rpc',dict(method='resident_prepare',args=[json.dumps(dataclasses.asdict(control))],timeout=60))
(ROOT/'qsa-resident-v1/prefill-private-inspect.json').write_text(json.dumps(result,indent=2)+'\n')
for row in result['results']:
    message=row.get('error','')
    if 'READ_ONLY_PRIVATE_STATE=' in message:
        values=json.loads(message.split('READ_ONLY_PRIVATE_STATE=',1)[1])
        print(json.dumps([{key:value for key,value in item.items() if key!='completed'} for item in values]))
    else:
        print(json.dumps(row))
