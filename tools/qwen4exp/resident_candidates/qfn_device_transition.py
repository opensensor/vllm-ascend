# SPDX-License-Identifier: Apache-2.0
"""Cold-controlled bridge for device acceptance, copies, and next input IDs.

The engine scheduler remains synchronous. Controls must explicitly report that
limitation; enabling this bridge is not a host-independent execution window.
"""

import json

from tools.qwen4exp.resident_candidates.qfn_acceptance_snapshot import RUNNER, original_base
from tools.qwen4exp.resident_candidates.qfn_mamba_commit import RPC

INPUT_TARGET = "vllm_ascend._310p.model_runner_310p:NPUModelRunner310._prepare_input_ids"


def make_replacements(base, resource, runner_class):
    bookkeeping_base = base[RUNNER + "._bookkeeping_sync"]
    bookkeeping_device = original_base(bookkeeping_base)
    update_base = base[RUNNER + "._update_states_after_model_execute"]
    draft_base = original_base(runner_class.propose_draft_token_ids)
    input_base = original_base(runner_class._prepare_input_ids)
    previous_control = base[RPC]

    def bookkeeping(runner, scheduler_output, sampler_output, *args, **kwargs):
        if resource.begin(sampler_output.sampled_token_ids, scheduler_output):
            return bookkeeping_device(runner, scheduler_output, sampler_output, *args, **kwargs)
        return bookkeeping_base(runner, scheduler_output, sampler_output, *args, **kwargs)

    def update(runner, output_token_ids, scheduler_output):
        if resource.retire_commit(output_token_ids):
            return None
        return update_base(runner, output_token_ids, scheduler_output)

    def draft(runner, *args, **kwargs):
        result = draft_base(runner, *args, **kwargs)
        resource.make_next(result)
        return result

    def inputs(runner, scheduler_output, num_reqs, total_num_scheduled_tokens, cu_num_tokens):
        if resource.take_inputs(scheduler_output, num_reqs, total_num_scheduled_tokens):
            return None
        return input_base(runner, scheduler_output, num_reqs, total_num_scheduled_tokens, cu_num_tokens)

    def control(worker, enabled, verify=False):
        if isinstance(enabled, str):
            try:
                payload = json.loads(enabled)
            except (ValueError, TypeError):
                return dict(rank=worker.rank, passed=False, error="invalid control JSON")
            if isinstance(payload, dict) and payload.get("transition") is True:
                try:
                    if set(payload) != {"transition", "enabled", "verify"} or any(
                        type(value) is not bool for value in payload.values()
                    ):
                        raise ValueError("transition control requires boolean enabled/verify")
                    if worker.model_runner.execute_model_state is not None or resource.pending is not None:
                        raise RuntimeError("drain inference before transition controls")
                    if resource.failed:
                        raise RuntimeError("failed transition retains owners")
                    resource.enabled, resource.verify = payload["enabled"], payload["verify"]
                    resource.next_pending = None
                    return dict(rank=worker.rank, passed=True, host_scheduler_retained=True, **resource.status())
                except Exception as error:
                    return dict(rank=worker.rank, passed=False, error=str(error))
        return previous_control(worker, enabled, verify)

    for wrapper, original in (
        (bookkeeping, bookkeeping_base),
        (update, update_base),
        (draft, draft_base),
        (inputs, input_base),
    ):
        wrapper._qfn_transition_base = original
    return {
        RUNNER + "._bookkeeping_sync": bookkeeping,
        RUNNER + "._update_states_after_model_execute": update,
        RUNNER + ".propose_draft_token_ids": draft,
        INPUT_TARGET: inputs,
        RPC: control,
    }
