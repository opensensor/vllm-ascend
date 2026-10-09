# SPDX-License-Identifier: Apache-2.0
"""CPU-only admission arithmetic and conservative GLM prefill memory checks.

The default geometry describes the archived TP4 GLM + MTP1 deployment, not
every GLM model. Cache admission is necessary but does not qualify execution.
No imports of torch/vLLM, device probes, configuration edits, or launches.
"""

import argparse
import json
from dataclasses import asdict, dataclass, replace
from decimal import Decimal
from pathlib import Path

GIB_BYTES = 1 << 30


def ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def require_positive(**values: int) -> None:
    if any(type(value) is not int or value <= 0 for value in values.values()):
        raise ValueError(f"positive integers required: {values}")


@dataclass(frozen=True)
class CacheGeometry:
    attention_block_tokens: int = 640
    main_slots: int = 12
    main_page_bytes: int = 655360
    small_slots: int = 12
    small_page_bytes: int = 40960
    compressor_state_block_tokens: int = 4
    kda_groups: int = 3

    def __post_init__(self):
        require_positive(**asdict(self))

    @property
    def bytes_per_global_id(self) -> int:
        return self.main_slots * self.main_page_bytes + self.small_slots * self.small_page_bytes


DEFAULT_CACHE_GEOMETRY = CacheGeometry()


@dataclass(frozen=True)
class PrefillConfig:
    chunk_tokens: int
    context_tokens: int
    concurrent_batches: int = 1
    speculative_tokens: int = 1
    speculative_kda_blocks: int = 1
    kda_prefill_checkpoint_blocks: int = 0
    extra_retained_tokens: int = 0

    def __post_init__(self):
        require_positive(
            chunk_tokens=self.chunk_tokens,
            context_tokens=self.context_tokens,
            concurrent_batches=self.concurrent_batches,
        )
        for name in (
            "speculative_tokens",
            "speculative_kda_blocks",
            "kda_prefill_checkpoint_blocks",
            "extra_retained_tokens",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} requires a nonnegative integer")


def cache_admission(config: PrefillConfig, geometry: CacheGeometry = DEFAULT_CACHE_GEOMETRY) -> dict:
    """Match align-mode shared-pool admission for ONE maximum-length request.

    The alternative retains all scheduler-held tail pages, but removes their
    main-page backing. It requires new physical address translation / pools;
    it is a design estimate, not an available runtime option.
    """
    full = ceil_div(config.context_tokens, geometry.attention_block_tokens)
    window_minus_one = geometry.compressor_state_block_tokens - 1 + config.speculative_tokens
    held_tokens = min(
        config.context_tokens,
        window_minus_one + config.extra_retained_tokens + config.chunk_tokens * config.concurrent_batches,
    )
    tail = ceil_div(held_tokens, geometry.compressor_state_block_tokens) + 1
    kda = geometry.kda_groups * (2 + config.speculative_kda_blocks + config.kda_prefill_checkpoint_blocks)
    total_ids = full + tail + kda
    current = total_ids * geometry.bytes_per_global_id
    removable_tail_main_bytes = tail * geometry.main_slots * geometry.main_page_bytes
    return {
        "full_attention_ids": full,
        "compressor_tail_ids": tail,
        "kda_ids": kda,
        "total_ids": total_ids,
        "bytes_per_global_id": geometry.bytes_per_global_id,
        "minimum_cache_bytes": current,
        "minimum_cache_gib": current / GIB_BYTES,
        "separate_tail_backing_design_bytes": current - removable_tail_main_bytes,
        "separate_tail_backing_design_savings_bytes": removable_tail_main_bytes,
        "scope": "one-request startup admission; excludes null block, retained prefixes and concurrency reserve",
    }


def coarser_state_page_design(config: PrefillConfig, geometry: CacheGeometry = DEFAULT_CACHE_GEOMETRY) -> dict:
    """Offline proposal: eight four-token pools per 32-token state page.

    Compression stays at four tokens; the conservative sliding window becomes
    state_block_tokens + speculative_tokens. All raw state rows fit inside the
    existing 40KiB padded page. The implemented candidate requires explicit
    startup opt-in and hardware qualification before deployment.
    """
    state_block_tokens, compression_tokens, state_row_bytes = 32, 4, 256 * 4
    raw_bytes = state_block_tokens * state_row_bytes
    if raw_bytes > geometry.small_page_bytes:
        raise ValueError("coarser compressor state would inflate the shared small-page class")
    proposed = replace(geometry, compressor_state_block_tokens=state_block_tokens)
    result = cache_admission(config, proposed)
    result.update(
        implemented=True,
        hardware_validated=False,
        compression_tokens=compression_tokens,
        state_block_tokens=state_block_tokens,
        sliding_window_tokens=state_block_tokens + config.speculative_tokens,
        raw_state_page_bytes=raw_bytes,
        padded_state_page_bytes=geometry.small_page_bytes,
        savings_vs_current_bytes=cache_admission(config, geometry)["minimum_cache_bytes"]
        - result["minimum_cache_bytes"],
    )
    return result


def proposed_state_pool_offsets(block_id: int, offset_in_block: int) -> tuple[int, tuple[int, ...]]:
    """FP32 element offsets for a proposed 32-token state page padded to 40KiB.

    Return the row write offset and all four pool-member read offsets. This
    is an addressing worksheet, not a runtime writer or metadata replacement.
    """
    if type(block_id) is not int or block_id < 0 or type(offset_in_block) is not int or not 0 <= offset_in_block < 32:
        raise ValueError("valid block ID and token offset required; sentinels must stay masked")
    page_offset = block_id * (40960 // 4)
    pool_start = offset_in_block // 4 * 4
    return page_offset + offset_in_block * 256, tuple(page_offset + (pool_start + member) * 256 for member in range(4))


def max_admitted_context(chunk_tokens: int, cache_bytes: int, geometry: CacheGeometry = DEFAULT_CACHE_GEOMETRY) -> int:
    """Exact monotonic inversion of default MTP1/align admission, not runtime capacity."""
    if type(cache_bytes) is not int or cache_bytes < 0:
        raise ValueError("cache_bytes must be a nonnegative integer")
    require_positive(chunk_tokens=chunk_tokens)
    lower, upper = 0, (cache_bytes // geometry.bytes_per_global_id) * geometry.attention_block_tokens
    while lower < upper:
        midpoint = (lower + upper + 1) // 2
        required = cache_admission(PrefillConfig(chunk_tokens, midpoint), geometry)["minimum_cache_bytes"]
        if required <= cache_bytes:
            lower = midpoint
        else:
            upper = midpoint - 1
    return lower


def moe_scratch_bytes(
    tokens: int, hidden: int = 4096, intermediate: int = 2048, top_k: int = 8, experts: int = 72
) -> dict:
    """Nine shared storages with the archived A4/packed-down/raw-scale flags.

    Includes FP16 routed output workspace and routed input packing. Excludes
    returned FP32 output, routing temporaries and all other operator buffers.
    This is ONE shared geometry allocation, not a per-layer multiplier.
    """
    require_positive(tokens=tokens, hidden=hidden, intermediate=intermediate, top_k=top_k, experts=experts)
    if (
        tokens <= 16
        or tokens * top_k > 65536
        or hidden > 4096
        or intermediate > 4096
        or experts > 288
        or hidden % 64
        or intermediate % 128
    ):
        raise ValueError("bulk scratch requires complete packing groups")
    routes = tokens * top_k
    slots = ceil_div(routes, 31) + experts
    buffers = {
        "input_low": tokens * hidden,
        "input_high_sentinel": 1,
        "input_scales": tokens * (hidden // 32) * 4,
        "hidden_low": slots * (intermediate // 64) * 2048,
        "hidden_high_sentinel": 1,
        "hidden_scales": slots * (intermediate // 128) * 32 * 4 * 4,
        "route_workspace": routes * hidden * 2,
        "routed_input": slots * (hidden // 64) * 2048,
        "routed_input_scales": slots * 32 * 128 * 4,
    }
    return {"buffers": buffers, "total_bytes": sum(buffers.values()), "route_slots": slots}


@dataclass(frozen=True)
class RankEnvelope:
    """Non-overlapping upper bounds for a specified candidate and rank.

    resident_excluding_cache_graphs_bytes includes weights and all persistent
    ordinary buffers, including MoE scratch. Graph bytes include graph-owned
    live allocations. transient_peak_bytes excludes these persistent costs.
    non_torch_peak_bytes is all external/operator allocation at peak, not just
    a startup sample. fragmentation_reserve_bytes bounds allocator slack.
    Never add allocated and reserved counters, or lifetime maxima from
    different phases. None means an unknown bound, which blocks feasibility.
    """

    rank: int
    usable_capacity_bytes: int
    cache_allocation_bytes: int
    candidate_signature: str
    candidate_config: PrefillConfig | None = None
    candidate_geometry: CacheGeometry = DEFAULT_CACHE_GEOMETRY
    bounds_qualified_for_candidate: bool = False
    resident_excluding_cache_graphs_bytes: int | None = None
    graph_pool_peak_bytes: int | None = None
    transient_peak_bytes: int | None = None
    non_torch_peak_bytes: int | None = None
    fragmentation_reserve_bytes: int | None = None
    safety_margin_bytes: int | None = None

    def __post_init__(self):
        if isinstance(self.candidate_config, dict):
            object.__setattr__(self, "candidate_config", PrefillConfig(**self.candidate_config))
        if isinstance(self.candidate_geometry, dict):
            object.__setattr__(self, "candidate_geometry", CacheGeometry(**self.candidate_geometry))
        require_positive(
            usable_capacity_bytes=self.usable_capacity_bytes, cache_allocation_bytes=self.cache_allocation_bytes
        )
        if type(self.rank) is not int or self.rank < 0 or not self.candidate_signature:
            raise ValueError("rank and candidate signature are required")
        if type(self.bounds_qualified_for_candidate) is not bool:
            raise ValueError("bounds qualification must be boolean")
        for name, value in asdict(self).items():
            if name.endswith("_bytes") and value is not None and (type(value) is not int or value < 0):
                raise ValueError(f"{name} must be nonnegative integer bytes or unknown")


def assess_candidate(
    config: PrefillConfig,
    ranks: list[RankEnvelope],
    candidate_signature: str,
    geometry: CacheGeometry = DEFAULT_CACHE_GEOMETRY,
) -> dict:
    if not ranks or len({r.rank for r in ranks}) != len(ranks):
        raise ValueError("provide a nonempty set of distinct rank envelopes")
    admission = cache_admission(config, geometry)
    results = []
    for rank in ranks:
        data = asdict(rank)
        costs = [value for name, value in data.items() if name.endswith("_bytes") and name != "usable_capacity_bytes"]
        unknown = [name for name, value in data.items() if name.endswith("_bytes") and value is None]
        admission_fits = rank.cache_allocation_bytes >= admission["minimum_cache_bytes"]
        qualified = (
            rank.bounds_qualified_for_candidate
            and rank.candidate_signature == candidate_signature
            and rank.candidate_config == config
            and rank.candidate_geometry == geometry
        )
        peak = sum(costs) if not unknown else None
        fits = admission_fits and qualified and peak is not None and peak <= rank.usable_capacity_bytes
        results.append(
            {
                "rank": rank.rank,
                "cache_admission_fits": admission_fits,
                "cache_admission_gap_bytes": max(0, admission["minimum_cache_bytes"] - rank.cache_allocation_bytes),
                "unknown_bounds": unknown,
                "bounds_match_candidate": qualified,
                "peak_upper_bound_bytes": peak,
                "headroom_after_reserves_bytes": None if peak is None else rank.usable_capacity_bytes - peak,
                "offline_feasible": fits,
            }
        )
    return {
        "config": asdict(config),
        "geometry": asdict(geometry),
        "cache_admission": admission,
        "ranks": results,
        "offline_feasible": all(r["offline_feasible"] for r in results),
        "hardware_validated": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunk-tokens", type=int, required=True)
    parser.add_argument("--context-tokens", type=int, required=True)
    parser.add_argument("--state-block-tokens", type=int, choices=(4, 32), default=4)
    parser.add_argument("--cache-gib", type=Decimal)
    parser.add_argument("--envelope", type=Path, help="JSON with candidate_signature and ranks (exclusive bounds)")
    args = parser.parse_args()
    config = PrefillConfig(args.chunk_tokens, args.context_tokens)
    geometry = replace(DEFAULT_CACHE_GEOMETRY, compressor_state_block_tokens=args.state_block_tokens)
    result = {
        "config": asdict(config),
        "geometry": asdict(geometry),
        "cache_admission": cache_admission(config, geometry),
        "shared_moe_scratch": moe_scratch_bytes(config.chunk_tokens),
        "coarser_state_page_design": coarser_state_page_design(config),
        "offline_feasible": False,
    }
    if args.cache_gib is not None:
        cache_bytes = int(args.cache_gib * GIB_BYTES)
        result["max_admitted_context_at_given_cache"] = max_admitted_context(args.chunk_tokens, cache_bytes, geometry)
    if args.envelope:
        envelope = json.loads(args.envelope.read_text())
        result = assess_candidate(
            config, [RankEnvelope(**rank) for rank in envelope["ranks"]], envelope["candidate_signature"], geometry
        )
    else:
        result["blocked_reason"] = (
            "Candidate-specific per-rank graph, transient, external workspace and allocator bounds required"
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
