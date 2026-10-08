# SPDX-License-Identifier: Apache-2.0
import torch
import torch_npu
from vllm_ascend.models.glm5next.kpool_ops import select_kpool_groups, expand_kpool_groups

torch.npu.set_device(0)
torch_npu.npu.set_compile_mode(jit_compile=False)
for pools in (8192, 32768, 77760):
    logits = torch.zeros(1, pools, device='npu')
    logits[:, :64] = torch.randn(1,64,device='npu')
    pos = torch.tensor([255],device='npu',dtype=torch.int32)
    for i in range(3):
        selected, _, start, count = select_kpool_groups(logits,pos,2048,4)
        scpu = selected.cpu()
        output = expand_kpool_groups(selected,start,count,4).cpu()
        print(pools,i,'selected nonnegative',int((scpu>=0).sum()),'selected bad',scpu[(scpu < -1)|(scpu>=64)].tolist(),
              'expanded nonnegative',int((output>=0).sum()),'expanded bad',output[(output < -1)|(output>=256)].tolist(),flush=True)
