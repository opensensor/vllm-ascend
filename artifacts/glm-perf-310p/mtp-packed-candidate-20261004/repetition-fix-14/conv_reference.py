"""Independent actual-input convolution oracle outside capture."""
import ast
from pathlib import Path
import torch
from vllm_ascend.models.glm5next_w2 import kda_310

native = torch.ops._C_ascend.npu_causal_conv1d_310

def conv(x, weight, **kw):
    if torch.npu.is_current_stream_capturing():
        return native(x, weight, **kw)
    state = kw['conv_states']
    bounds = kw['query_start_loc'].cpu().tolist()
    slots = kw['cache_indices'].cpu().tolist()
    accepted = kw['num_accepted_tokens']
    accepted = accepted.cpu().tolist() if accepted is not None else None
    initial = kw['initial_state_mode']
    initial = initial.cpu().tolist() if initial is not None else None
    xc, wc = x.cpu().float(), weight.cpu().float()
    expected = torch.zeros_like(xc)
    for i, (start, end) in enumerate(zip(bounds, bounds[1:])):
        slot = slots[i][0] if isinstance(slots[i], list) else slots[i]
        if end <= start or slot < 0: continue
        has_init = initial[i] if initial is not None else kw['run_mode'] == 1
        offset = accepted[i] - 1 if accepted is not None else 0
        history = state[slot,offset:offset+wc.shape[0]-1].cpu().float() if has_init else torch.zeros(wc.shape[0]-1,xc.shape[-1])
        for t in range(start,end):
            window = torch.cat((history,xc[t:t+1]),dim=0)
            out = (window*wc).sum(0)
            if kw['bias'] is not None: out += kw['bias'].cpu().float()
            expected[t] = torch.nn.functional.silu(out)
            history = window[1:]
    actual = native(x,weight,**kw)
    got=actual.cpu().float()
    if not torch.allclose(got,expected,rtol=.02,atol=.002):
        print('GLM_CONV_ERROR',torch.distributed.get_rank(),kw['run_mode'],bounds,accepted,'max',(got-expected).abs().max().item(),flush=True)
    return expected.to(device=x.device,dtype=x.dtype)

def replacements():
    source=Path(kda_310.__file__).read_text()
    tree=ast.parse(source)
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run_stateful_kda_310')
    source=ast.get_source_segment(source,fn).replace('torch.ops._C_ascend.npu_causal_conv1d_310(', 'conv(')
    scope=dict(vars(kda_310),conv=conv)
    exec(compile(source,'<conv_reference>','exec'),scope)
    return {'vllm_ascend.models.glm5next_w2.kda_310:run_stateful_kda_310':scope['run_stateful_kda_310']}
