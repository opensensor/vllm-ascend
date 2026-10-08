"""Actual-input CPU prefill recurrence, including final carry comparison."""
import ast
from pathlib import Path
import torch
from vllm_ascend.models.glm5next_w2 import kda_310
source=Path(kda_310.__file__).read_text()
node=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='_run_prefill')
scope=dict(vars(kda_310))
exec(compile(ast.get_source_segment(source,node),'<original_prefill>','exec'),scope)
original=scope['_run_prefill']

def prefill(self_attn,q,k,v,raw_gate,beta_raw,recurrent_state,state_indices,has_initial_state,chunk_metadata):
    if torch.npu.is_current_stream_capturing():
        return original(self_attn,q,k,v,raw_gate,beta_raw,recurrent_state,state_indices,has_initial_state,chunk_metadata)
    cu=chunk_metadata.cu_seqlens_host if chunk_metadata.cu_seqlens_kern is None else chunk_metadata.cu_seqlens_kern
    bounds=cu.cpu().tolist() if isinstance(cu,torch.Tensor) else list(cu)
    keep=chunk_metadata.keep_meta
    slots=state_indices if keep is None else state_indices[keep]
    initial=has_initial_state if keep is None else has_initial_state[keep]
    carries=kda_310._prefill_initial_state(recurrent_state,slots,initial).cpu()
    qn=kda_310._l2norm_310p(q).squeeze(0).cpu().float()
    kn=kda_310._l2norm_310p(k).squeeze(0).cpu().float()
    val=v.squeeze(0).cpu().float()
    gates=kda_310._safe_gate_for_layer(self_attn,raw_gate).squeeze(0).cpu()
    beta=beta_raw.float().sigmoid().squeeze(0).cpu()
    expected=torch.zeros_like(val)
    for i,(start,end) in enumerate(zip(bounds,bounds[1:])):
        carry=carries[i]
        for t in range(start,end):
            carry=carry*gates[t].exp()[:,None,:]
            residual=(val[t]-(carry*kn[t,:,None,:]).sum(-1))*beta[t,:,None]
            carry=carry+residual[:,:,None]*kn[t,:,None,:]
            expected[t]=(carry*qn[t,:,None,:]).sum(-1)*self_attn.head_dim**-0.5
        carries[i]=carry
    out=original(self_attn,q,k,v,raw_gate,beta_raw,recurrent_state,state_indices,has_initial_state,chunk_metadata)
    got=out.squeeze(0).cpu().float()
    states=recurrent_state[slots].cpu().float()
    for name,actual,ref in [('output',got,expected),('state',states,carries)]:
        if not torch.allclose(actual,ref,rtol=.03,atol=.003):
            print('GLM_PREFILL_ERROR',torch.distributed.get_rank(),self_attn.prefix,name,'max',(actual-ref).abs().max().item(),'ref',ref.abs().max().item(),'mean',(actual-ref).abs().mean().item(),flush=True)
    recurrent_state[slots]=carries.to(recurrent_state.device,dtype=recurrent_state.dtype)
    return expected.unsqueeze(0).to(out.device,dtype=out.dtype)

def replacements():
    return {'vllm_ascend.models.glm5next_w2.kda_310:_run_prefill':prefill}
