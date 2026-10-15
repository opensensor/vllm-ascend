# SPDX-License-Identifier: Apache-2.0
"""Cold-controlled worker schedule materialization for the queued engine."""

import json

from tools.qwen4exp.ple_history_runtime import uniform_history_rows
from tools.qwen4exp.resident_candidates.qfn_acceptance_snapshot import RUNNER
from tools.qwen4exp.resident_candidates.qfn_device_transition import INPUT_TARGET
from tools.qwen4exp.resident_candidates.qfn_mamba_commit import RPC
from tools.qwen4exp.resident_candidates.qfn_ple_history import STAGE_TARGET


def make_replacements(base, resource, runner_class):
    original = runner_class.execute_model
    original = getattr(original, "_qfn_queued_base", original)
    previous_control = base[RPC]
    previous_inputs = base[INPUT_TARGET]
    previous_stage = base[STAGE_TARGET]

    def execute(runner, scheduler_output, *args, **kwargs):
        if runner is not resource.runner:
            raise RuntimeError("queued schedule owner belongs to a different worker")
        return original(runner, resource.resolve(scheduler_output), *args, **kwargs)

    execute._qfn_queued_base = original

    def inputs(runner, *args, **kwargs):
        result = previous_inputs(runner, *args, **kwargs)
        required = resource.expected_device_inputs
        if required is not None:
            if resource.history.continuation != required or tuple(runner.input_batch.req_ids) != required:
                raise RuntimeError("reserved CPU drafts cannot replace a missing device input packet")
            resource.expected_device_inputs = None
            resource.expected_device_history = required
        return result

    def stage(runner, padded):
        required = resource.expected_device_history
        if required is not None and (
            resource.history.continuation != required or not uniform_history_rows(runner, padded)
        ):
            raise RuntimeError("reserved CPU drafts cannot enter a fallback PLE hash")
        result = previous_stage(runner, padded)
        resource.expected_device_history = None
        return result

    inputs._qfn_transition_base = previous_inputs
    stage._qfn_transition_base = previous_stage

    def control(worker, enabled, verify=False):
        if isinstance(enabled, str):
            try:
                payload = json.loads(enabled)
            except (ValueError, TypeError):
                return dict(rank=worker.rank, passed=False, error="invalid control JSON")
            if isinstance(payload, dict) and payload.get("queued_schedule") is True:
                try:
                    if payload == {"queued_schedule": True, "status": True}:
                        return dict(rank=worker.rank, passed=True, **resource.status())
                    if set(payload) == {"queued_schedule", "device_drafts"} and type(payload["device_drafts"]) is bool:
                        if (
                            resource.runner.execute_model_state is not None
                            or resource.expected_device_inputs is not None
                            or resource.expected_device_history is not None
                        ):
                            raise RuntimeError("drain device input/history packets before changing draft controls")
                        if payload["device_drafts"] and not resource.enabled:
                            raise RuntimeError("enable queued scheduling before device draft consumption")
                        resource.device_drafts = payload["device_drafts"]
                        return dict(rank=worker.rank, passed=True, **resource.status())
                    if set(payload) != {"queued_schedule", "enabled"} or type(payload["enabled"]) is not bool:
                        raise ValueError("queued schedule control requires boolean enabled")
                    if resource.runner.execute_model_state is not None or resource.runner.use_async_scheduling:
                        raise RuntimeError("drain synchronous worker execution before changing queued controls")
                    resource.enabled = payload["enabled"]
                    return dict(rank=worker.rank, passed=True, **resource.status())
                except Exception as error:
                    return dict(rank=worker.rank, passed=False, error=str(error))
        return previous_control(worker, enabled, verify)

    return {RUNNER + ".execute_model": execute, INPUT_TARGET: inputs, STAGE_TARGET: stage, RPC: control}
