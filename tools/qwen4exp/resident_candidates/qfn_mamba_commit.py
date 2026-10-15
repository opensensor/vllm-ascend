# SPDX-License-Identifier: Apache-2.0
"""Guarded postprocess replacement and cold A/B controls for grouped commits."""

import json

from tools.qwen4exp.resident_candidates.qfn_acceptance_snapshot import MAMBA

NATIVE_NAME = "qfn_mamba_commit_v4"
RPC = "vllm_ascend._310p.worker_310p:NPUWorker310.qfn_cpu_profile_stop"


def make_replacements(base, resource):
    original = base[MAMBA]
    previous_control = base[RPC]

    def postprocess(**kwargs):
        if resource.enabled and resource.execute(kwargs):
            return None
        return original(**kwargs)

    def control(worker, enabled, verify=False):
        if type(enabled) is bool:
            return previous_control(worker, enabled, verify)
        try:
            request = json.loads(enabled)
            if set(request) != {"enabled", "verify", "all_steps"} or any(type(v) is not bool for v in request.values()):
                raise ValueError("commit controls require enabled/verify/all_steps booleans")
            runner = worker.model_runner
            if (
                runner is not resource.runner
                or runner.execute_model_state is not None
                or getattr(runner, "_qfn_acceptance_snapshot", None) is not None
                or hasattr(runner.input_batch, "_qfn_commit_counts")
            ):
                raise RuntimeError("drain inference before commit controls")
            if resource.failed:
                raise RuntimeError("failed native commit retains its owners")
            if request["enabled"] and not getattr(runner, "_qfn_acceptance_reuse", False):
                raise RuntimeError("commit requires one-step acceptance reuse")
            resource.enabled, resource.verify, resource.all_steps = (
                request["enabled"],
                request["verify"],
                request["all_steps"],
            )
            return dict(rank=worker.rank, passed=True, **resource.status())
        except Exception as error:
            return dict(rank=worker.rank, passed=False, error=str(error))

    return {MAMBA: postprocess, RPC: control}
