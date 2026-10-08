"""Compare latent-cache contents with their known values between forwards."""
import json
import torch
from vllm_ascend._310p.attention.mla_v1_310 import AscendMLAImpl310
original = AscendMLAImpl310._exec_kv_mla_nope


def latent(self, kv_no_split, kv_cache, slots, is_prefill):
    if torch.npu.is_current_stream_capturing():
        return original(self,kv_no_split,kv_cache,slots,is_prefill)
    cache=kv_cache[0]
    audit={} if is_prefill else getattr(self,'_mtp_mla_audit',{})
    for slot, expected in audit.items():
        actual=cache[slot//cache.shape[2],:,slot%cache.shape[2],:].cpu().reshape(-1)
        if not torch.equal(actual,expected):
            print('GLM_MLA_CHANGED '+json.dumps(dict(rank=torch.distributed.get_rank(),layer=self.layer_name,
                slot=slot,max_error=(actual.float()-expected.float()).abs().max().item(),
                shape=list(cache.shape),stride=list(cache.stride()),offset=cache.storage_offset())),flush=True)
    result=original(self,kv_no_split,kv_cache,slots,is_prefill)
    expected=self.kv_a_layernorm(kv_no_split.reshape(-1,self.kv_lora_rank)).cpu()
    audit={}
    for row,slot in enumerate(slots.cpu().reshape(-1).tolist()):
        if slot<0: continue
        audit[slot]=expected[row].clone()
        actual=cache[slot//cache.shape[2],:,slot%cache.shape[2],:].cpu().reshape(-1)
        if not torch.equal(actual,expected[row]):
            print('GLM_MLA_WRITE '+json.dumps(dict(rank=torch.distributed.get_rank(),layer=self.layer_name,
                slot=slot,max_error=(actual.float()-expected[row].float()).abs().max().item())),flush=True)
    self._mtp_mla_audit=audit
    return result


def replacements():
    return {'vllm_ascend._310p.attention.mla_v1_310:AscendMLAImpl310._exec_kv_mla_nope':latent}
