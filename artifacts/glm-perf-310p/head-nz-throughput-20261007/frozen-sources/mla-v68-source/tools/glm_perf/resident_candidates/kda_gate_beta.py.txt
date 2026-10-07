# SPDX-License-Identifier: Apache-2.0
"""Independent resident candidate: gate/beta fusion, without Q/K batching."""

from tools.glm_perf.resident_candidates.kda_input_preparation import native_gate_replacements


def replacements(native_resources):
    return native_gate_replacements(native_resources)
