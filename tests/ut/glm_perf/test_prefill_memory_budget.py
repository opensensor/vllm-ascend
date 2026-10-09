# SPDX-License-Identifier: Apache-2.0
"""CPU admission regressions, production-source conformance and refusal gates."""

import ast
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.glm_perf.prefill_memory_budget import (
    GIB_BYTES,
    CacheGeometry,
    PrefillConfig,
    RankEnvelope,
    assess_candidate,
    cache_admission,
    ceil_div,
    coarser_state_page_design,
    max_admitted_context,
    moe_scratch_bytes,
    proposed_state_pool_offsets,
)

ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize("context,rounded", [(311040, "6.36"), (131072, "4.17"), (65536, "3.38")])
def test_reproduces_archived_1280_startup_requirements(context, rounded):
    result = cache_admission(PrefillConfig(1280, context))
    assert f"{result['minimum_cache_gib']:.2f}" == rounded
    assert result["compressor_tail_ids"] == 322


def test_chunk_growth_and_separate_tail_design_preserve_page_lifetimes():
    old = cache_admission(PrefillConfig(640, 131072))
    new = cache_admission(PrefillConfig(1280, 131072))
    assert old["compressor_tail_ids"] == 162
    assert new["minimum_cache_bytes"] - old["minimum_cache_bytes"] == 1336934400
    assert new["separate_tail_backing_design_savings_bytes"] == 2532311040
    assert new["separate_tail_backing_design_bytes"] == 1946419200


def test_32_token_state_page_fits_existing_small_page_class():
    candidate = coarser_state_page_design(PrefillConfig(1280, 131072))
    assert candidate["compressor_tail_ids"] == 42
    assert candidate["minimum_cache_bytes"] == 256 * 8355840
    assert candidate["raw_state_page_bytes"] == 32768
    assert candidate["padded_state_page_bytes"] == 40960
    assert candidate["compression_tokens"] == 4
    assert candidate["sliding_window_tokens"] == 33
    assert candidate["implemented"] is True and candidate["hardware_validated"] is False
    with pytest.raises(ValueError, match="inflate"):
        coarser_state_page_design(PrefillConfig(1280, 131072), CacheGeometry(small_page_bytes=16384))


def test_proposed_state_addresses_keep_eight_pools_distinct_and_padding_untouched():
    writes = set()
    for block_id in (0, 7, 123):
        for offset in range(32):
            write, reads = proposed_state_pool_offsets(block_id, offset)
            assert write in reads
            assert write not in writes
            writes.add(write)
            assert all(block_id * 10240 <= read < block_id * 10240 + 8192 for read in reads)
        assert proposed_state_pool_offsets(block_id, 31)[1] != proposed_state_pool_offsets(block_id, 0)[1]


@pytest.mark.parametrize("final_position", [3, 4, 7, 8, 30, 31, 32, 33, 63, 64, 1279, 1280])
def test_proposed_page_holds_both_pools_needed_for_mtp1_rejection(final_position):
    # Compare the exact logical pools retained by the current writer. Use
    # deliberately nonconsecutive physical IDs to exercise padded page strides.
    first_retained = (final_position - 1) // 4 * 4
    addresses = {}
    for position in range(first_retained, final_position + 1):
        physical_id = 17 + (position // 32) * 3
        write, reads = proposed_state_pool_offsets(physical_id, position % 32)
        assert write in reads
        addresses[position] = write
    assert len(set(addresses.values())) == len(addresses)
    for rejected in (False, True):
        accepted_last = final_position - int(rejected)
        pool_start = accepted_last // 4 * 4
        assert all(p in addresses for p in range(max(first_retained, pool_start), accepted_last + 1))


@pytest.mark.parametrize("chunk", [640, 1280, 2560])
@pytest.mark.parametrize("capacity", [0, 100, 3 * GIB_BYTES, int(3.72 * GIB_BYTES), 6 * GIB_BYTES])
def test_max_context_inverts_full_admission_including_short_contexts(chunk, capacity):
    maximum = max_admitted_context(chunk, capacity)
    if maximum:
        assert cache_admission(PrefillConfig(chunk, maximum))["minimum_cache_bytes"] <= capacity
    assert cache_admission(PrefillConfig(chunk, maximum + 1))["minimum_cache_bytes"] > capacity


def test_concurrent_batches_speculation_and_rollover_are_explicit():
    assert cache_admission(PrefillConfig(1280, 131072, concurrent_batches=2))["compressor_tail_ids"] == 642
    plain = cache_admission(PrefillConfig(1280, 131072, speculative_tokens=0, speculative_kda_blocks=0))
    assert plain["kda_ids"] == 6
    short = cache_admission(PrefillConfig(1280, 3))
    assert short["compressor_tail_ids"] == 2
    assert short["full_attention_ids"] == 1


def test_production_required_scheduler_blocks_matches_offline_budget():
    """Run actual allocator functions without importing vLLM or probing devices."""
    source = ast.parse((ROOT / "vllm_ascend/models/glm5next/cache_config.py").read_text())
    names = {"_required_scheduler_blocks", "get_glm5_next_pool_bytes_per_block", "get_glm5_next_max_memory_usage"}
    functions = [node for node in source.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert len(functions) == len(names)
    for tokens in (640, 1280, 2560):
        for context in (65536, 131072, 311040):
            config = PrefillConfig(tokens, context)
            expected = cache_admission(config)

            def group(ids, page):
                return SimpleNamespace(
                    kv_cache_spec=SimpleNamespace(page_size_bytes=page, max_memory_usage_bytes=lambda _: ids * page)
                )

            layout = SimpleNamespace(
                full_group=group(ceil_div(context, 640), 655360),
                state_group=group(ceil_div(min(context, tokens + 4), 4) + 1, 40960),
                mamba_groups=[group(3, 655360) for _ in range(3)],
                compact_mamba=False,
                main_slot_count=12,
                main_page_size=655360,
                small_slot_count=12,
                small_page_size=40960,
            )
            namespace = {
                "cdiv": ceil_div,
                "_get_glm5_next_cache_layout": lambda _, current=layout: current,
                "_compact_mamba_pool_bytes": lambda *_: 0,
                "_host_hot_bytes": lambda *_: 0,
                "envs": SimpleNamespace(VLLM_ASCEND_310P_GLM_HOST_KV=False),
            }
            module = ast.Module(
                body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *functions],
                type_ignores=[],
            )
            exec(compile(ast.fix_missing_locations(module), "production-cache-budget", "exec"), namespace)
            assert namespace["get_glm5_next_max_memory_usage"](config, []) == expected["minimum_cache_bytes"]


def test_scratch_matches_archived_nine_storages_and_production_allocator_shapes():
    source = ast.parse((ROOT / "tools/glm_perf/glm_fused_moe.py").read_text())
    names = {"route_input_shapes", "route_down_shape", "route_down_scale_shape"}
    functions = [n for n in source.body if isinstance(n, ast.FunctionDef) and n.name in names]
    cls = next(n for n in source.body if isinstance(n, ast.ClassDef) and n.name == "NativeFusedMoE")
    methods = {
        "input_scale_shape",
        "hidden_code_shape",
        "hidden_scale_shape",
        "allocate_scratch",
        "route_workspace_dtype",
    }
    selected = ast.ClassDef(
        name="NativeFusedMoE",
        bases=[],
        keywords=[],
        body=[n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in methods],
        decorator_list=[],
    )

    def empty(shape, dtype, device):
        size = 1
        for extent in shape:
            size *= extent
        return size * dtype

    namespace = {
        "torch": SimpleNamespace(empty=empty, int8=1, float16=2, float32=4),
        "FUSED_REDUCTION_TOKENS": 16,
        "ROUTE_INPUT_ROWS": 32,
        "ROUTE_INPUT_MAX_GROUPS": 128,
        "ROUTE_INPUT_PAIR_BYTES": 2048,
    }
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[*functions, selected], type_ignores=[])),
            "production-scratch",
            "exec",
        ),
        namespace,
    )
    native = namespace["NativeFusedMoE"]()
    native.device = "cpu"
    native.activation_bits = 4
    native.fp16_route_workspace = True
    native.route_packed_down = native.route_compact_down_scales = True
    native.raw_hidden_scales = native.raw_input_scales = True
    for tokens in (640, 1280, 2560):
        geometry = SimpleNamespace(
            tokens=tokens, top_k=8, experts=72, hidden=4096, intermediate=2048, activation_bits=4
        )
        ordinary = native.allocate_scratch(geometry)
        route_input = namespace["route_input_shapes"](geometry)
        total = sum(ordinary) + empty(route_input[0], 1, "cpu") + empty(route_input[1], 4, "cpu")
        assert total == moe_scratch_bytes(tokens)["total_bytes"]
    assert moe_scratch_bytes(640)["total_bytes"] == 97533954


def qualified_rank(config, rank=0):
    return RankEnvelope(
        rank=rank,
        usable_capacity_bytes=46 * GIB_BYTES,
        cache_allocation_bytes=5 * GIB_BYTES,
        candidate_signature="frozen-source-and-binaries",
        candidate_config=config,
        bounds_qualified_for_candidate=True,
        resident_excluding_cache_graphs_bytes=35 * GIB_BYTES,
        graph_pool_peak_bytes=2 * GIB_BYTES,
        transient_peak_bytes=GIB_BYTES,
        non_torch_peak_bytes=GIB_BYTES,
        fragmentation_reserve_bytes=GIB_BYTES,
        safety_margin_bytes=GIB_BYTES,
    )


def assess(config, ranks):
    return assess_candidate(config, ranks, "frozen-source-and-binaries")


def test_unknown_bounds_fail_closed_even_when_cache_fits():
    config = PrefillConfig(1280, 131072)
    result = assess(config, [replace(qualified_rank(config), graph_pool_peak_bytes=None)])
    assert result["ranks"][0]["cache_admission_fits"]
    assert not result["offline_feasible"]
    assert result["ranks"][0]["unknown_bounds"] == ["graph_pool_peak_bytes"]


def test_one_small_rank_blocks_all_ranks_and_safety_margin_is_counted():
    config = PrefillConfig(1280, 131072)
    rank = qualified_rank(config)
    assert assess(config, [rank])["offline_feasible"]
    assert not assess(config, [rank, replace(rank, rank=1, usable_capacity_bytes=45 * GIB_BYTES)])["offline_feasible"]


@pytest.mark.parametrize(
    "change",
    [
        {"candidate_signature": "stale"},
        {"bounds_qualified_for_candidate": False},
        {"candidate_config": PrefillConfig(640, 131072)},
    ],
)
def test_stale_or_unqualified_bounds_never_authorize_candidate(change):
    config = PrefillConfig(1280, 131072)
    assert not assess(config, [replace(qualified_rank(config), **change)])["offline_feasible"]


def test_packed_geometry_cannot_reuse_old_geometry_envelope():
    config = PrefillConfig(1280, 131072)
    geometry = CacheGeometry(compressor_state_block_tokens=32)
    rank = qualified_rank(config)
    result = assess_candidate(config, [rank], "frozen-source-and-binaries", geometry)
    assert result["cache_admission"]["minimum_cache_bytes"] == 2139095040
    assert not result["offline_feasible"]
    assert assess_candidate(
        config, [replace(rank, candidate_geometry=geometry)], "frozen-source-and-binaries", geometry
    )["offline_feasible"]


@pytest.mark.parametrize("bad", [0, -1, True, 1280.0])
def test_invalid_geometry_and_budgets(bad):
    with pytest.raises(ValueError):
        PrefillConfig(bad, 131072)
    with pytest.raises(ValueError):
        CacheGeometry(main_slots=bad)
