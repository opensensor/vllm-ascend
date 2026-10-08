"""Short-context dense CPU attention reference for live latent decode queries."""
import json
import torch
from vllm_ascend._310p.attention import mla_v1_310 as mla
original = mla.AscendMLAImpl310._forward_decode_fused


def decode(self,query,key_cache,value_cache,attn_metadata):
    if torch.npu.is_current_stream_capturing():
        return original(self,query,key_cache,value_cache,attn_metadata)
    meta=attn_metadata.decode
    positions=meta.input_positions[:query.shape[0]].cpu().tolist()
    if max(positions,default=0)>=256:
        return original(self,query,key_cache,value_cache,attn_metadata)
    table=mla._qsa_cache_block_table(meta.block_table[:attn_metadata.num_decodes].to(torch.int32),key_cache.shape[2]).cpu().tolist()
    boundaries=attn_metadata.query_start_loc[:attn_metadata.num_decodes+1].cpu().tolist()
    q=query.cpu().float()
    expected=torch.zeros_like(q)
    for req in range(attn_metadata.num_decodes):
        for row in range(boundaries[req],min(boundaries[req+1],len(positions))):
            length=positions[row]+1
            chunks=[]
            for block in range((length+key_cache.shape[2]-1)//key_cache.shape[2]):
                keep=min(key_cache.shape[2],length-block*key_cache.shape[2])
                chunks.append(key_cache[table[req][block],:,:keep,:].cpu().permute(1,0,2).reshape(keep,-1).float())
            keys=torch.cat(chunks)
            weights=(q[row]@keys.T*self.scale).softmax(-1)
            expected[row]=weights@keys
    native=original(self,query,key_cache,value_cache,attn_metadata)
    reference=self._v_up_proj(expected.to(device=query.device,dtype=query.dtype).transpose(0,1).contiguous())
    got=native.cpu().float(); want=reference.cpu().float()
    if not torch.allclose(got,want,rtol=0.03,atol=0.01):
        print('GLM_MLA_NUMERIC '+json.dumps(dict(rank=torch.distributed.get_rank(),layer=self.layer_name,
            maximum=(got-want).abs().max().item(),mean=(got-want).abs().mean().item(),ref_max=want.abs().max().item(),positions=positions)),flush=True)
    return reference


def replacements():
    return {'vllm_ascend._310p.attention.mla_v1_310:AscendMLAImpl310._forward_decode_fused':decode}
