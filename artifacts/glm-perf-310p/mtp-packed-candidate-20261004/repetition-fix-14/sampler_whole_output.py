"""Check and replace column-wise MTP1 sampling writes with one dense copy."""
import json
import torch
from vllm_ascend.sample.rejection_sampler import rejection_greedy_sample_spec_len_1_pytorch as original


def sample(output_token_ids, draft_token_ids, target_argmax, bonus_token_ids,
           uniform_probs=None, synthetic_conditional_rates=None, synthetic_mode=False):
    if synthetic_mode:
        accepted = (uniform_probs < synthetic_conditional_rates[0]) & (draft_token_ids >= 0)
        first = torch.where(accepted, draft_token_ids, target_argmax)
    else:
        accepted = draft_token_ids == target_argmax
        first = target_argmax
    second = torch.where(accepted, bonus_token_ids.reshape(-1), output_token_ids[:, 1])
    expected = torch.stack((first, second), dim=1).to(output_token_ids.dtype)
    original(output_token_ids, draft_token_ids, target_argmax, bonus_token_ids,
             uniform_probs, synthetic_conditional_rates, synthetic_mode)
    if not torch.equal(output_token_ids, expected):
        print('GLM_SAMPLER_MISMATCH ' + json.dumps(dict(
            rank=torch.distributed.get_rank(), actual=output_token_ids.cpu().tolist(),
            expected=expected.cpu().tolist(), draft=draft_token_ids.cpu().tolist(),
            target=target_argmax.cpu().tolist(), bonus=bonus_token_ids.cpu().tolist())), flush=True)
    output_token_ids.copy_(expected)


def replacements():
    return {'vllm_ascend.sample.rejection_sampler:rejection_greedy_sample_spec_len_1_pytorch': sample}
