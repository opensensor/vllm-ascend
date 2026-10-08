"""Trace recurrent carry between prefill and successive decode calls."""
import ast
from pathlib import Path
import json
import torch
from vllm_ascend.models.glm5next_w2 import kda_310

original_recurrent = kda_310._run_recurrent


def recurrent(self_attn, q, k, v, raw_gate, beta_raw, recurrent_state, cu_seqlens,
              state_indices, *, num_sequences, num_accepted_tokens=None):
    audit = getattr(self_attn, '_mtp_state_audit', None)
    active = audit is not None and not torch.npu.is_current_stream_capturing()
    rows = None
    if active:
        rows = state_indices[:num_sequences].cpu().tolist()
        counts = num_accepted_tokens[:num_sequences].cpu().tolist() if num_accepted_tokens is not None else [1]*num_sequences
        for row, count in zip(rows, counts):
            row = row if isinstance(row, list) else [row]
            if count < 1 or count > len(row): continue
            slot = row[count-1]
            if slot in audit:
                before = recurrent_state[slot].cpu()
                if not torch.equal(before, audit[slot]):
                    print('GLM_STATE_CHANGED ' + json.dumps(dict(rank=torch.distributed.get_rank(),layer=self_attn.prefix,
                        slot=slot, accepted=count, max_error=(before.float()-audit[slot].float()).abs().max().item(),
                        shape=list(recurrent_state.shape),stride=list(recurrent_state.stride()),offset=recurrent_state.storage_offset())),flush=True)
    result = original_recurrent(self_attn,q,k,v,raw_gate,beta_raw,recurrent_state,cu_seqlens,state_indices,
                                num_sequences=num_sequences,num_accepted_tokens=num_accepted_tokens)
    if active:
        for row in rows:
            for slot in row if isinstance(row,list) else [row]:
                if slot >= 0: audit[slot] = recurrent_state[slot].cpu()
    return result


def replacements():
    module_source = Path(kda_310.__file__).read_text()
    node = next(n for n in ast.parse(module_source).body if isinstance(n, ast.FunctionDef) and n.name == '_run_prefill')
    source = ast.get_source_segment(module_source, node)
    old = '    recurrent_state[state_indices] = result[1].to(recurrent_state.dtype)'
    assert old in source
    source = source.replace(old, old + '''
    final = result[1].to(recurrent_state.dtype).cpu()
    self_attn._mtp_state_audit = {slot: final[row].clone() for row, slot in enumerate(state_indices.cpu().tolist())}''')
    scope = dict(kda_310.__dict__)
    exec(compile(source, '<state_lifetime_audit>', 'exec'), scope)
    return {'vllm_ascend.models.glm5next_w2.kda_310:_run_prefill':scope['_run_prefill'],
            'vllm_ascend.models.glm5next_w2.kda_310:_run_recurrent':recurrent}
