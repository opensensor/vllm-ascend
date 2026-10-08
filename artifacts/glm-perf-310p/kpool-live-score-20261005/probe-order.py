# SPDX-License-Identifier: Apache-2.0
import torch
import torch_npu
from test_glm_kpool_score_310 import inputs, op, reference
from vllm_ascend.models.glm5next.kpool_ops import select_kpool_groups
from vllm_ascend.utils import enable_custom_op

torch.npu.set_device(0)
torch_npu.npu.set_compile_mode(jit_compile=False)
enable_custom_op()
torch.ops.load_library('/home/matteius/experiments/glm-kpool-live-score-20261005/glm_kpool_score_candidate.so')
a = inputs(2, 77760, [64, 64])
q, w, storage, table, bounds, pos = a[:6]
pools, blocks, br, bs, rs, offset = a[6:]
cache = storage.as_strided((blocks,br,128),(bs,rs,1),offset)
new = op(a)
for row in range(2):
    ids = torch.arange(pools, device='npu')
    valid = ids < 64
    pages = torch.where(valid, table[row,ids//br].long(),0)
    keys = torch.where(valid[:,None],cache[pages,ids%br],0)
    old = ((q[row].float() @ keys.float().T).relu_() * w[row,:,None]).sum(0,keepdim=True)
    x = select_kpool_groups(old,pos[row:row+1],2048,4)[0].cpu()
    y = select_kpool_groups(new[row:row+1],pos[row:row+1],2048,4)[0].cpu()
    error=(old[0,:64]-new[row,:64]).cpu()
    print('row',row,'maxscoreerror',float(error.abs().max()))
    bad=(x!=y)
    print('different',bad.nonzero().tolist(),'old_ids',x[bad].tolist(),'new_ids',y[bad].tolist())
    print('oldscore',old[0,x[bad].clamp_min(0).npu().long()].cpu().tolist())
    print('newscore',new[row,y[bad].clamp_min(0).npu().long()].cpu().tolist())
