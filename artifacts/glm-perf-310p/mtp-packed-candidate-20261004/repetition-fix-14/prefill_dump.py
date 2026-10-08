import ast
from pathlib import Path
import torch
from vllm_ascend.models.glm5next_w2 import kda_310
source=Path(kda_310.__file__).read_text()
node=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='_run_prefill')
scope=dict(vars(kda_310))
source=ast.get_source_segment(source,node).replace('recurrent_state[state_indices] = result[1].to(recurrent_state.dtype)', 'self_attn._prefill_raw_state = result[1]\n    recurrent_state[state_indices] = result[1].to(recurrent_state.dtype)')
exec(compile(source,'<dump_prefill>','exec'),scope)
original=scope['_run_prefill']
def prefill(self_attn,q,k,v,raw_gate,beta_raw,recurrent_state,state_indices,has_initial_state,chunk_metadata):
    out=original(self_attn,q,k,v,raw_gate,beta_raw,recurrent_state,state_indices,has_initial_state,chunk_metadata)
    if q.shape[1] == 25 and self_attn.prefix == 'model.layers.0.self_attn':
        torch.save({name:value.cpu() for name,value in dict(q=q,k=k,v=v,qn=kda_310._l2norm_310p(q),kn=kda_310._l2norm_310p(k),raw_gate=raw_gate,gates=kda_310._safe_gate_for_layer(self_attn,raw_gate),beta=beta_raw.float().sigmoid(),state=self_attn._prefill_raw_state,stored=recurrent_state[state_indices],out=out,a_log=self_attn.A_log,dt_bias=self_attn.dt_bias).items()}, '/home/matteius/experiments/glm-w3-20261004/repetition-fix-14/prefill-rank%d.pt'%torch.distributed.get_rank())
    return out

def replacements():
    return {'vllm_ascend.models.glm5next_w2.kda_310:_run_prefill':prefill}
