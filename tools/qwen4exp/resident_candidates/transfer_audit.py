# SPDX-License-Identifier: Apache-2.0
"""Bounded prefix/barrier records and W4 TP payloads without device readbacks.

Graph records reflect Python capture/dispatch, not replayed device task counts.
Use the NPU profiler for graph memcpy/kernel/HCCL execution measurements.
"""

from time import perf_counter_ns

from vllm_ascend._310p.transfer_audit import TransferLedger


def replacements():
    from vllm_ascend._310p.model_runner_310p import NPUModelRunner310
    from vllm_ascend.models.qwen4_exp.w4_moe import W4SparseMoE

    update = NPUModelRunner310._update_states
    update = getattr(update, "_qwen_prefix_base", update)
    forward = W4SparseMoE.forward
    forward = getattr(forward, "_qwen_w4_forward_base", forward)

    def states(self, output):
        for tier in getattr(self, "_prefix_mamba_tiers", {}).values():
            tier.transfer_ledger.enable_events()
        before = perf_counter_ns()
        result = update(self, output)
        ledger = getattr(self, "_transfer_ledger", None)
        if ledger is None:
            ledger = self._transfer_ledger = TransferLedger(event_limit=256)
        ledger.record("host_phase", "runner_update_states", elapsed_ns=perf_counter_ns() - before)
        return result

    def moe(self, inputs):
        if not hasattr(self, "_transfer_ledger"):
            self._transfer_ledger = TransferLedger(event_limit=256)
        # Preserve function ownership and restore even if the collective fails.
        original_reduce = self._tp_reduce
        if original_reduce is None:
            return forward(self, inputs)

        def reduce(tensor):
            result = original_reduce(tensor)
            self._transfer_ledger.record("collective", "w4_all_reduce", nbytes=tensor.numel() * tensor.element_size())
            return result

        self._tp_reduce = reduce
        try:
            return forward(self, inputs)
        finally:
            self._tp_reduce = original_reduce

    states._qwen_prefix_base = update
    moe._qwen_w4_forward_base = forward
    return {
        "vllm_ascend._310p.model_runner_310p:NPUModelRunner310._update_states": states,
        "vllm_ascend.models.qwen4_exp.w4_moe:W4SparseMoE.forward": moe,
    }
