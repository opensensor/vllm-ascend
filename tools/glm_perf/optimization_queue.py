# SPDX-License-Identifier: Apache-2.0
"""Print the next GLM test queue. This module never contacts a server or NPU."""

import json
from dataclasses import asdict, dataclass

CONTEXT_TOKENS = 311040
PREFILL_PAGE_TOKENS = 640
ROUTES_PER_TOKEN = 8
DEFAULT_ROUTE_CAPACITY = 6144
MAX_ROUTE_CAPACITY = 32768


@dataclass(frozen=True)
class ExpertBatch:
    scheduler_tokens: int
    grouped_max_routes: int
    context_tokens: int = CONTEXT_TOKENS

    def __post_init__(self):
        if self.scheduler_tokens not in (640, 1280, 2560):
            raise ValueError("use the isolated 640, 1280, or conditional 2560 token profile")
        if self.scheduler_tokens % PREFILL_PAGE_TOKENS:
            raise ValueError("scheduler batch must be a whole prefill page")
        if not DEFAULT_ROUTE_CAPACITY <= self.grouped_max_routes <= MAX_ROUTE_CAPACITY:
            raise ValueError("route capacity is outside the compiled operator range")
        if self.scheduler_tokens * ROUTES_PER_TOKEN > self.grouped_max_routes:
            raise ValueError("route cap would split the expert batch")
        if self.context_tokens != CONTEXT_TOKENS:
            raise ValueError("keep configured context fixed for the comparison")

    def plan(self):
        return {
            **asdict(self),
            "compiler_define": f"GLM_W2_GROUPED_MAX_ROUTES={self.grouped_max_routes}",
            "hf_override": {"ascend_glm_grouped_max_routes": self.grouped_max_routes},
            "requires_runner_reallocation": True,
            "verify_iteration_context_tokens": self.scheduler_tokens,
        }


def queue():
    return {
        "hardware_status": "deferred_to_qwen",
        "common": {"context": CONTEXT_TOKENS, "mtp": 1, "graph_sizes": [2, 8], "max_seqs": 4, "port": 8001},
        "experiments": [
            {
                "id": "expert_batch",
                "profiles": [
                    ExpertBatch(640, 6144).plan(),
                    ExpertBatch(1280, 10240).plan(),
                    ExpertBatch(2560, 20480).plan(),
                ],
                "gate": "1280 first; 2560 only after measured memory fit; preserve decoder and expert kernel schedule",
                "readiness": "configuration staged; native route-cap build and launch required",
            },
            {
                "id": "adaptive_expert_teams",
                "lanes": [2, 4],
                "small_expert_rows_below": 128,
                "gate": "mixed/empty/uniform/hot routes; all cores for >=128 expert rows and <=64 total routes",
                "readiness": "C++ ownership helper and integration patch staged; native build pending",
            },
            {
                "id": "kda_batched_qk",
                "factory": "tools.glm_perf.resident_candidates.kda_input_preparation:replacements",
                "gate": "FP16 Q/K parity, MTP accepted-state arguments, graph rows 2 and 8, c1/c4",
                "readiness": "Python resident candidate; NPU parity and recapture pending",
            },
            {
                "id": "kpool_decode_epilogue",
                "factory": "tools.glm_perf.resident_candidates.kpool_decode_epilogue:replacements",
                "gate": "exact indices including ties and padded rows at configured 311K capacity; c1/c4",
                "readiness": "Python resident candidate; NPU parity and recapture pending",
            },
            {
                "id": "kda_gate_beta",
                "factory": "tools.glm_perf.resident_candidates.kda_gate_beta:replacements",
                "resource": "kda_gate_beta_v1",
                "gate": "FP32 gate and FP16 beta parity, extremes, recurrent carry, MTP rejection/repetition",
                "readiness": "native kernel source and wrapper staged; native compilation pending",
            },
            {
                "id": "moe_half_unpermute",
                "factory": "tools.glm_perf.resident_candidates.moe_half_unpermute:replacements",
                "gate": "exact outputs; initialized peer rows; verify fallback dispatch; c1/c4 and cold prefill",
                "readiness": "Python resident candidate; NPU parity and recapture pending",
            },
        ],
    }


if __name__ == "__main__":
    print(json.dumps(queue(), indent=2))
