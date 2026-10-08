"""Record actual target input/position pairs and target greedy predictions."""
import json
import torch
from vllm_ascend.models.glm5next.model import Glm5NextModel,Glm5NextForCausalLM
original_forward=Glm5NextModel.forward
original_logits=Glm5NextForCausalLM.compute_logits

def forward(self,input_ids,positions,intermediate_tensors,inputs_embeds=None,**kw):
    if not torch.npu.is_current_stream_capturing() and torch.distributed.get_rank()==0:
        print('GLM_INPUT '+json.dumps({'ids':input_ids.cpu().tolist() if input_ids is not None else None,'positions':positions.cpu().tolist()}),flush=True)
    return original_forward(self,input_ids,positions,intermediate_tensors,inputs_embeds,**kw)

def logits(self,*a,**kw):
    result=original_logits(self,*a,**kw)
    if result is not None and not torch.npu.is_current_stream_capturing() and torch.distributed.get_rank()==0:
        print('GLM_LOGITS '+json.dumps(result.argmax(-1).cpu().tolist()),flush=True)
    return result

def replacements():
    return {'vllm_ascend.models.glm5next.model:Glm5NextModel.forward':forward,
            'vllm_ascend.models.glm5next.model:Glm5NextForCausalLM.compute_logits':logits}
