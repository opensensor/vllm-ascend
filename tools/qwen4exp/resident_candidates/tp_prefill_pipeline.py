# SPDX-License-Identifier: Apache-2.0
"""Opt-in bounded prefill reduction pipeline; default and decode stay intact."""

CHUNK_TOKENS = 1024


def replacements():
    import torch

    from tools.qwen4exp.tp_prefill_pipeline import npu_pipelined_prefill
    from vllm_ascend.models.qwen4_exp.w4_moe import W4SparseMoE, route_topk

    original = W4SparseMoE.forward
    original = getattr(original, "_qwen_w4_forward_base", original)

    def forward(self, inputs):
        if inputs.device.type != "npu" or inputs.shape[0] <= CHUNK_TOKENS or torch.npu.is_current_stream_capturing():
            return original(self, inputs)
        if not self.native_int4 or self.shared_expert_execution != "tp_sharded":
            raise ValueError("TP pipeline requires native INT4 with tp_sharded shared experts")
        return npu_pipelined_prefill(self, inputs, route_topk, chunk_tokens=CHUNK_TOKENS)

    forward._qwen_w4_forward_base = original
    return {"vllm_ascend.models.qwen4_exp.w4_moe:W4SparseMoE.forward": forward}
