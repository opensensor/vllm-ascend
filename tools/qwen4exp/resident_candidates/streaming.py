# SPDX-License-Identifier: Apache-2.0
"""One admitted streaming candidate; no global default or unguarded loader."""


def replacements(native_resources):
    candidates = [
        value["candidate"] for value in native_resources.values() if isinstance(value, dict) and "candidate" in value
    ]
    if len(candidates) != 1:
        raise ValueError("exactly one admitted streaming candidate must be provided")
    candidate = candidates[0]
    candidate.require_admission()

    import torch

    from vllm_ascend._310p.model_runner_310p import NPUModelRunner310
    from vllm_ascend.models.qwen4_exp.model import _GDNAttention, _PLEInjection
    from vllm_ascend.models.qwen4_exp.w4_moe import W4SparseMoE

    return candidate.replacements(
        moe_original=W4SparseMoE.forward,
        capturing=lambda tensor: tensor.device.type != "npu" or torch.npu.is_current_stream_capturing(),
        gdn_original=_GDNAttention._native_delta_rule,
        prefix_update_original=NPUModelRunner310._update_states,
        prefix_remap_original=NPUModelRunner310._remap_compact_mamba_block_tables,
        ple_original=_PLEInjection._forward_eager,
    )
