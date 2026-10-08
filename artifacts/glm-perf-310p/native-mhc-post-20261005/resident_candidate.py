# SPDX-License-Identifier: Apache-2.0
"""Select the qualified native prefill mixer without changing model weights."""

from types import SimpleNamespace

import torch

from vllm_ascend.models.glm5next.ops.mhc_native import use_native_mhc_post


def select_native(op, *args):
    selection = SimpleNamespace(
        use_310p_native_mhc_post=True, use_310p_fp16_mhc_state=getattr(op, "use_310p_fp16_mhc_state", False)
    )
    selected = use_native_mhc_post(selection, *args)
    if selected and not getattr(op, "_native_mhc_logged", False):
        if torch.distributed.get_rank() == 0:
            print(f"GLM_NATIVE_MHC_PREFILL rows={args[0].shape[0]} width={args[0].shape[1]}", flush=True)
        op._native_mhc_logged = True
    return selected


def replacements():
    return {"vllm_ascend.patch.worker.patch_mhc_norm:use_native_mhc_post": select_native}
