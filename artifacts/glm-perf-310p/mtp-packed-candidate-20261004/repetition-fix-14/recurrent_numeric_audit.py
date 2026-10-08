"""Independent CPU recurrence on actual serving tensors, direct execution only."""
import json
import torch
from vllm_ascend.models.glm5next_w2 import kda_310
original = kda_310._run_recurrent


def recurrent(self_attn,q,k,v,raw_gate,beta_raw,recurrent_state,cu_seqlens,state_indices,*,num_sequences,num_accepted_tokens=None):
    if torch.npu.is_current_stream_capturing():
        return original(self_attn,q,k,v,raw_gate,beta_raw,recurrent_state,cu_seqlens,state_indices,
                        num_sequences=num_sequences,num_accepted_tokens=num_accepted_tokens)
    qn=kda_310._l2norm_310p(q).squeeze(0).half().cpu().float()
    kn=kda_310._l2norm_310p(k).squeeze(0).half().cpu().float()
    val=v.squeeze(0).half().cpu().float()
    gates=kda_310._safe_gate_for_layer(self_attn,raw_gate).squeeze(0).cpu().float()
    beta=beta_raw.float().sigmoid().squeeze(0).half().cpu().float()
    boundaries=cu_seqlens[:num_sequences+1].cpu().tolist()
    slots=state_indices[:num_sequences].cpu().tolist()
    accepted=num_accepted_tokens[:num_sequences].cpu().tolist() if num_accepted_tokens is not None else [1]*num_sequences
    expected=torch.zeros_like(val)
    for i in range(num_sequences):
        row=slots[i] if isinstance(slots[i],list) else [slots[i]]
        if boundaries[i+1] <= boundaries[i]: continue
        carry=recurrent_state[row[accepted[i]-1]].cpu().float()
        for t in range(boundaries[i],boundaries[i+1]):
            carry=carry*gates[t].exp()[:,None,:]
            residual=(val[t]-(carry*kn[t,:,None,:]).sum(-1))*beta[t,:,None]
            carry=carry+residual[:,:,None]*kn[t,:,None,:]
            expected[t]=(carry*qn[t,:,None,:]).sum(-1)*self_attn.head_dim**-0.5
    actual=original(self_attn,q,k,v,raw_gate,beta_raw,recurrent_state,cu_seqlens,state_indices,
                    num_sequences=num_sequences,num_accepted_tokens=num_accepted_tokens)
    got=actual.squeeze(0).cpu().float()
    error=(got-expected).abs()
    if not torch.allclose(got,expected,rtol=0.02,atol=0.002):
        print('GLM_RECURRENT_ERROR '+json.dumps(dict(rank=torch.distributed.get_rank(),layer=self_attn.prefix,
            maximum=error.max().item(),reference_max=expected.abs().max().item(),mean=error.mean().item(),
            slots=slots,accepted=accepted,boundaries=boundaries)),flush=True)
    return actual


def replacements():
    return {'vllm_ascend.models.glm5next_w2.kda_310:_run_recurrent':recurrent}
