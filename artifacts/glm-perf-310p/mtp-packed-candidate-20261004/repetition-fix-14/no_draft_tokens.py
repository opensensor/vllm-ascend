"""Diagnostic: keep the draft model resident/running but schedule no draft tokens."""
from vllm_ascend.worker.model_runner_v1 import NPUModelRunner
original = NPUModelRunner.take_draft_token_ids


def take(self):
    result = original(self)
    if result is not None:
        result.draft_token_ids = [[] for _ in result.req_ids]
    return result


def replacements():
    return {'vllm_ascend.worker.model_runner_v1:NPUModelRunner.take_draft_token_ids': take}
