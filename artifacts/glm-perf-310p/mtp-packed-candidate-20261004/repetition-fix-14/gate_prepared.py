"""Apply production gate functions and prepare resident operands before capture."""
import ast
from pathlib import Path
import torch
from vllm_ascend.models.glm5next_w2 import kda_310
from vllm_ascend.models.glm5next.model import Glm5NextModel

source=Path(kda_310.__file__).read_text()
scope=dict(vars(kda_310))
for name in ('prepare_kda_gate_weights','_safe_gate_for_layer'):
    node=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name==name)
    exec(compile(ast.get_source_segment(source,node),'<production_gate>','exec'),scope)
prepare=scope['prepare_kda_gate_weights']
gate=scope['_safe_gate_for_layer']
original_forward=Glm5NextModel.forward

def forward(self,*args,**kwargs):
    for layer in self._active_layers:
        attn=layer.self_attn
        if hasattr(attn,'A_log') and not hasattr(attn,'_kda_gate_weights'):
            assert not torch.npu.is_current_stream_capturing(), 'gate preparation must precede capture'
            prepare(attn)
            if torch.distributed.get_rank()==0:
                print('GLM_GATE_PREPARED',attn.prefix,flush=True)
    return original_forward(self,*args,**kwargs)

def replacements():
    return {'vllm_ascend.models.glm5next_w2.kda_310:_safe_gate_for_layer':gate,
            'vllm_ascend.models.glm5next.model:Glm5NextModel.forward':forward}
