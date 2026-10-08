"""Recompute bounded gate from current weights; bypass stale graph-owned cache."""
from vllm_ascend.models.glm5next_w2 import kda_310

def gate(self_attn,raw_gate):
    return kda_310._safe_gate(raw_gate,self_attn.A_log,self_attn.dt_bias,float(self_attn.kda_lower_bound))

def replacements():
    return {'vllm_ascend.models.glm5next_w2.kda_310:_safe_gate_for_layer':gate}
