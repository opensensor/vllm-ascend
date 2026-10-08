"""Diagnostic: emit only the first target token on every verification step."""
import torch
from vllm_ascend.sample.rejection_sampler import rejection_sample as original


def sample(draft_token_ids, num_draft_tokens, max_spec_len, cu_num_draft_tokens,
           draft_probs, target_logits, bonus_token_ids, sampling_metadata, **kwargs):
    if sampling_metadata.all_greedy and max_spec_len == 1 and all(n == 1 for n in num_draft_tokens):
        first = target_logits.argmax(dim=-1).to(torch.int32)
        return torch.stack((first, torch.full_like(first, -1)), dim=1)
    return original(draft_token_ids, num_draft_tokens, max_spec_len, cu_num_draft_tokens,
                    draft_probs, target_logits, bonus_token_ids, sampling_metadata, **kwargs)


def replacements():
    return {'vllm_ascend.sample.rejection_sampler:rejection_sample': sample,
            'vllm.v1.sample.rejection_sampler:rejection_sample': sample}
