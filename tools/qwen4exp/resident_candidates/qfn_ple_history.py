# SPDX-License-Identifier: Apache-2.0
"""Cold-controlled PLE history closure, retaining host lookup and scheduling."""

import json

from tools.qwen4exp.resident_candidates.qfn_acceptance_snapshot import RUNNER, original_base
from tools.qwen4exp.resident_candidates.qfn_device_transition import INPUT_TARGET
from tools.qwen4exp.resident_candidates.qfn_mamba_commit import RPC

STAGE_TARGET = "vllm_ascend._310p.model_runner_310p:NPUModelRunner310._stage_qwen4exp_ple_inputs"


def make_replacements(base, history, runner_class):
    transition = history.transition
    stage_base = original_base(runner_class._stage_qwen4exp_ple_inputs)
    input_base = original_base(runner_class._prepare_input_ids)
    draft_base = base[RUNNER + ".propose_draft_token_ids"]
    control_base = base[RPC]

    def inputs(runner, scheduler, num_reqs, total, cu_tokens):
        used = transition.take_inputs(scheduler, num_reqs, total)
        history.input_result(used)
        if not used:
            return input_base(runner, scheduler, num_reqs, total, cu_tokens)

    def stage(runner, padded):
        if runner is not history.runner:
            raise RuntimeError("PLE history belongs to another runner")
        return history.stage(padded, stage_base)

    def draft(runner, *args, **kwargs):
        result = draft_base(runner, *args, **kwargs)
        history.advance()
        return result

    def control(worker, enabled, verify=False):
        if isinstance(enabled, str):
            try:
                payload = json.loads(enabled)
            except (ValueError, TypeError):
                return dict(rank=worker.rank, passed=False, error="invalid control JSON")
            if isinstance(payload, dict) and payload.get("ple_history") is True:
                try:
                    if set(payload) != {"ple_history", "enabled", "verify"} or any(
                        type(value) is not bool for value in payload.values()
                    ):
                        raise ValueError("PLE history control requires boolean enabled/verify")
                    if worker.model_runner.execute_model_state is not None or transition.pending is not None:
                        raise RuntimeError("drain inference before PLE history controls")
                    if history.failed or transition.failed:
                        raise RuntimeError("failed device transition retains owners")
                    if payload["enabled"] and not transition.enabled:
                        raise ValueError("enable device transition before PLE history")
                    history.enabled, history.verify = payload["enabled"], payload["verify"]
                    history.pending = history.continuation = history.staged = None
                    return dict(rank=worker.rank, passed=True, **history.status())
                except Exception as error:
                    return dict(rank=worker.rank, passed=False, error=str(error))
        return control_base(worker, enabled, verify)

    for wrapper, original in ((inputs, input_base), (stage, stage_base), (draft, draft_base)):
        wrapper._qfn_transition_base = original
    return {INPUT_TARGET: inputs, STAGE_TARGET: stage, RUNNER + ".propose_draft_token_ids": draft, RPC: control}
