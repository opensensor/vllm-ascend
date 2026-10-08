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
        "hardware_status": "completed_pool_candidate_serving_on_8001; leave_running",
        "completed_run": "artifacts/glm-perf-310p/next-queue-20261005/hardware/README.md",
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
                "readiness": "route-cap parity passed; two memory-admission failures; final launch cancelled",
            },
            {
                "id": "adaptive_expert_teams",
                "lanes": [2, 4],
                "small_expert_rows_below": 128,
                "gate": "mixed/empty/uniform/hot routes; all cores for >=128 expert rows and <=64 total routes",
                "readiness": "compiled; both variants 20/20 exact NPU cases; projection timing flat; parked",
            },
            {
                "id": "kda_batched_qk",
                "factory": "tools.glm_perf.resident_candidates.kda_input_preparation:replacements",
                "gate": "FP16 Q/K parity, MTP accepted-state arguments, graph rows 2 and 8, c1/c4",
                "readiness": "NPU parity and serving completed; no established gain; parked",
            },
            {
                "id": "kpool_decode_epilogue",
                "factory": "tools.glm_perf.resident_candidates.kpool_decode_epilogue:replacements",
                "gate": "exact indices including ties and padded rows at configured 311K capacity; c1/c4",
                "readiness": "NPU parity and serving completed; c4 gain within baseline variability; parked",
            },
            {
                "id": "kda_gate_beta",
                "factory": "tools.glm_perf.resident_candidates.kda_gate_beta:replacements",
                "resource": "kda_gate_beta_v1",
                "gate": "FP32 gate and FP16 beta parity, extremes, recurrent carry, MTP rejection/repetition",
                "readiness": "compiled; NPU parity and serving completed; no gain; parked",
            },
            {
                "id": "moe_half_unpermute",
                "factory": "tools.glm_perf.resident_candidates.moe_half_unpermute:replacements",
                "gate": "exact outputs; initialized peer rows; verify fallback dispatch; c1/c4 and cold prefill",
                "readiness": "NPU parity and serving completed; c4 gain within baseline variability; parked",
            },
            {
                "id": "indexer_projection",
                "factory": "tools.glm_perf.resident_candidates.indexer_projection:replacements",
                "extra_weight_budget_bytes": 64 * 1024 * 1024,
                "gate": "real-weight projection/pool-index parity; target/MTP; memory fit; c1/c4 and cold prefill",
                "readiness": "real-weight exact projection gate failed; not admitted to serving",
            },
            {
                "id": "direct_route_tokens",
                "factory": "tools.glm_perf.resident_candidates.direct_route_tokens:replacements",
                "gate": "exact metadata/gather parity; INT32 division/gather; graphs 2/8; c1/c4 and cold prefill",
                "readiness": "division variant passes capture and 9 replay tests; serving timing flat; parked",
            },
            {
                "id": "kda_prepare_head",
                "compiler_define": "GLM_KDA_GATE_PREPARE_HEAD",
                "gate": "exact gate cumsum and KDA state/output parity; varlen tails; native ordering; cold prefill",
                "readiness": "safe-gate output/carry parity passed; whole KDA timing flat; parked",
            },
            {
                "id": "kda_skip_safe_score_cube",
                "compiler_define": "GLM_KDA_SKIP_SAFE_SCORE_CUBE",
                "gate": "exact KDA outputs/carry; varlen replay; dependency ordering; native timing and cold TTFT",
                "readiness": "safe-gate output/carry parity passed; whole KDA timing flat; parked",
            },
            {
                "id": "kpool_completed_prefill",
                "factory": "tools.glm_perf.resident_candidates.kpool_completed_prefill:replacements",
                "gate": "exact cache writes and MTP1 tails; realistic cache strides; cold prefill; decode fallback",
                "readiness": "82 CPU tests; 24 NPU parity cases; 7 serving checks passed; active on 8001",
                "evidence": "artifacts/glm-perf-310p/next-queue-20261005/completed-pools/README.md",
            },
        ],
    }


if __name__ == "__main__":
    print(json.dumps(queue(), indent=2))
