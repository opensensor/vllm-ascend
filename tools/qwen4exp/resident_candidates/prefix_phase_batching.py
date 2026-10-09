# SPDX-License-Identifier: Apache-2.0
"""Batch prefix updates/admissions, preserving barriers between phases."""


def replacements():
    from vllm_ascend._310p.model_runner_310p import NPUModelRunner310

    def wrap(original):
        original = getattr(original, "_qwen_prefix_base", original)

        def call(self, *args, **kwargs):
            existed = hasattr(self, "_prefix_phase_batching")
            previous = getattr(self, "_prefix_phase_batching", False)
            self._prefix_phase_batching = True
            try:
                return original(self, *args, **kwargs)
            finally:
                if existed:
                    self._prefix_phase_batching = previous
                else:
                    del self._prefix_phase_batching

        call._qwen_prefix_base = original
        return call

    return {
        "vllm_ascend._310p.model_runner_310p:NPUModelRunner310._update_states": wrap(NPUModelRunner310._update_states),
        "vllm_ascend._310p.model_runner_310p:NPUModelRunner310._remap_compact_mamba_block_tables": wrap(
            NPUModelRunner310._remap_compact_mamba_block_tables
        ),
    }
