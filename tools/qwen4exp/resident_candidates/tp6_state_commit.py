# SPDX-License-Identifier: Apache-2.0
"""Use the cold 310P acceptance snapshot for guarded TP6 state commits."""

import hashlib
import inspect
import json

from tools.qwen4exp.resident_candidates.qfn_mamba_commit import MAMBA, NATIVE_NAME, RPC

QUALIFIED_FALLBACK_SHA256 = "bb51af267b65cc544f4f1495bfbca7c0d704153e4b3f37e076ba6d1bc7665c59"


def make_replacements(original, previous_control, resource):
    resource.verified_shapes = set()

    def postprocess(**kwargs):
        batch = kwargs["input_batch"]
        counts = getattr(batch, "_mamba_accepted_counts_snapshot", None)
        if not resource.enabled or counts is None:
            return original(**kwargs)
        if hasattr(batch, "_qfn_commit_counts"):
            raise RuntimeError("another state-commit owner already holds this step")
        batch._qfn_commit_counts = counts
        try:
            committed = resource.execute(kwargs)
            if committed and resource.verify:
                resource.verified_shapes.add(kwargs["num_reqs"])
        finally:
            del batch._qfn_commit_counts
        if not committed:
            return original(**kwargs)
        return None

    def control(worker, enabled, verify=False):
        if not isinstance(enabled, str):
            return previous_control(worker, enabled, verify)
        try:
            options = json.loads(enabled)
            if set(options) != {"tp6_commit", "enabled", "verify"} or any(
                type(v) is not bool for v in options.values()
            ):
                raise ValueError("TP6 state controls require boolean tp6_commit/enabled/verify")
            runner = worker.model_runner
            if (
                not options["tp6_commit"]
                or runner is not resource.runner
                or runner.execute_model_state is not None
                or getattr(runner, "_mamba_sample_snapshot", None) is not None
                or hasattr(runner.input_batch, "_qfn_commit_counts")
                or hasattr(runner.input_batch, "_mamba_accepted_counts_snapshot")
            ):
                raise RuntimeError("drain TP6 acceptance snapshots before changing state commits")
            if resource.failed:
                raise RuntimeError("failed state commit retains its owners")
            if options["verify"] and not options["enabled"]:
                raise ValueError("verification requires enabled commits")
            if (
                options["enabled"]
                and not options["verify"]
                and (6 not in resource.verified_shapes or resource.checked_blocks == 0)
            ):
                raise RuntimeError("verify six request rows and real boundary copies before enabling fast mode")
            resource.enabled, resource.verify = options["enabled"], options["verify"]
            resource.all_steps = resource.verify
            return dict(
                rank=worker.rank, passed=True, verified_shapes=sorted(resource.verified_shapes), **resource.status()
            )
        except Exception as error:
            return dict(rank=worker.rank, passed=False, error=str(error))

    return {MAMBA: postprocess, RPC: control}


def replacements(native_resources):
    # Import the worker-owned objects only in a drained resident transaction.
    from vllm.v1.worker import mamba_utils

    from vllm_ascend._310p.worker_310p import NPUWorker310
    from vllm_ascend.patch.worker.patch_mamba_utils import _postprocess_mamba_align_gpu_cpu_fallback

    original = mamba_utils.postprocess_mamba_align_gpu
    if original is not _postprocess_mamba_align_gpu_cpu_fallback:
        raise RuntimeError("TP6 state commit requires the cold CPU fallback")
    if hashlib.sha256(inspect.getsource(original).encode()).hexdigest() != QUALIFIED_FALLBACK_SHA256:
        raise RuntimeError("TP6 state fallback source differs from its qualification")
    return make_replacements(original, NPUWorker310.qfn_cpu_profile_stop, native_resources[NATIVE_NAME])
