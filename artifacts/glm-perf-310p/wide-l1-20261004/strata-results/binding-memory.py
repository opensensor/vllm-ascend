import argparse, gc, json
import torch, torch_npu
from vllm_ascend.utils import enable_custom_op
from vllm_ascend.models.glm5next_w2.model import _pack_codes_nz_w3
p=argparse.ArgumentParser();p.add_argument('--binding',choices=['original','owned','release'],required=True);a=p.parse_args()
enable_custom_op()
if a.binding!='original':
 torch.ops.load_library(f'/srv/ai/src/glm-l1-wide-build-20261004/build-strata-bindings-{a.binding}-20261004/glm_moe_candidates.so')
torch.npu.set_device(0);torch_npu.npu.set_compile_mode(jit_compile=False)
k=4096;rows=10240
codes=_pack_codes_nz_w3(torch.zeros(k,k*3//8,dtype=torch.uint8),k).unsqueeze(0).view(torch.int8).npu()
scales=torch.ones(1,k//32,k//32,dtype=torch.float32,device='npu')
ends=torch.tensor([128],dtype=torch.int64,device='npu')
op=torch.ops._C_ascend.npu_w2_grouped_blocked_dequant_matmul_310
out=[]
for steps in [1,16,64]:
 torch.npu.synchronize();gc.collect();torch.npu.empty_cache();torch.npu.reset_peak_memory_stats()
 before=torch.npu.memory_allocated()
 x=torch.zeros(rows,k,dtype=torch.float16,device='npu')
 for _ in range(steps): x=op(x,codes,scales,ends,False)
 torch.npu.synchronize()
 peak=torch.npu.max_memory_allocated();after=torch.npu.memory_allocated()
 good=bool((x[:128].cpu()==0).all())
 del x;gc.collect();torch.npu.synchronize()
 out.append({'steps':steps,'before':before,'peak':peak,'after_sync':after,'after_delete':torch.npu.memory_allocated(),'zero_local_output':good})
print(json.dumps({'binding':a.binding,'cases':out}),flush=True)
