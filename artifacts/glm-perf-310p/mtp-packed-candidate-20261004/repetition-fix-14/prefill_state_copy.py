"""Diagnostic: write each prefill carry through a dense physical page view."""
import inspect
from vllm_ascend.models.glm5next_w2 import kda_310


def replacements():
    source = inspect.getsource(kda_310._run_prefill)
    old = '    recurrent_state[state_indices] = result[1].to(recurrent_state.dtype)'
    assert old in source
    source = source.replace(old, '''    final = result[1].to(recurrent_state.dtype)
    for row, index in enumerate(state_indices.cpu().tolist()):
        recurrent_state[index].copy_(final[row])''')
    scope = dict(kda_310.__dict__)
    exec(compile(source, '<prefill_state_copy>', 'exec'), scope)
    return {'vllm_ascend.models.glm5next_w2.kda_310:_run_prefill': scope['_run_prefill']}
