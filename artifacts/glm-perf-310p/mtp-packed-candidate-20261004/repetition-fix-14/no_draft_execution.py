"""Diagnostic: retain MTP allocations but omit draft execution and verification."""
import torch
from vllm_ascend.worker.model_runner_v1 import NPUModelRunner
original_take = NPUModelRunner.take_draft_token_ids


def take(self):
    result = original_take(self)
    if result is not None:
        result.draft_token_ids = [[] for _ in result.req_ids]
    return result


def propose(self, *args, **kwargs):
    self._draft_probs = None
    self._draft_prob_req_ids = None
    return torch.zeros((self.input_batch.num_reqs, 1), dtype=torch.int64, device=self.device)


def replacements():
    return {
        'vllm_ascend.worker.model_runner_v1:NPUModelRunner.take_draft_token_ids': take,
        'vllm_ascend.worker.model_runner_v1:NPUModelRunner.propose_draft_token_ids': propose,
    }
