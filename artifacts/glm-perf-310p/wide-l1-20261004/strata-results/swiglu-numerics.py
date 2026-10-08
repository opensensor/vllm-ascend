import json, torch, torch_npu
from vllm_ascend.utils import enable_custom_op
torch.npu.set_device(0)
torch_npu.npu.set_compile_mode(jit_compile=False)
enable_custom_op()
torch.ops.load_library('/srv/ai/src/glm-l1-wide-build-20261004/build-strata-bindings-owned-20261004/glm_moe_candidates.so')
gate=torch.arange(65536,dtype=torch.int32).to(torch.int16).view(torch.float16)
gate=torch.where(torch.isfinite(gate),gate,0).reshape(-1,32)
rows=[]
for up_value in [1.,1.3,-2.7]:
 up=torch.full_like(gate,up_value)
 data=torch.cat((gate,up),dim=1).npu()
 g,u=data.chunk(2,-1);g=g.float();u=u.float()
 ref=(torch.nn.functional.silu(g)*u).half().cpu()
 actual=torch.ops._C_ascend.npu_w2_swiglu_310(data).cpu()
 div=(g/(1+torch.exp(-g))*u).half().cpu()
 mul=(g*torch.sigmoid(g)*u).half().cpu()
 # Framework half conversion is also the actual model's final rounding.
 item={'up':up_value,'count':ref.numel(),'custom_mismatches':int((actual!=ref).sum()),'div_mismatches':int((div!=ref).sum()),'mul_mismatches':int((mul!=ref).sum()),'max_error':float((actual.float()-ref.float()).abs().max())}
 bad=(actual!=ref).flatten().nonzero().flatten()[:16]
 item['examples']=[{'gate':float(gate.flatten()[i]),'ref':float(ref.flatten()[i]),'custom':float(actual.flatten()[i]),'div':float(div.flatten()[i]),'mul':float(mul.flatten()[i])} for i in bad]
 rows.append(item)
print(json.dumps(rows),flush=True)
