# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for GLM-Next cache grouping, physical layout, and capacity accounting."""

from types import SimpleNamespace

import pytest
import torch
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_utils import (
    generate_scheduler_kv_cache_config,
    get_request_block_hasher,
    init_none_hash,
)
from vllm.v1.core.single_type_kv_cache_manager import (
    register_all_kvcache_specs,
)
from vllm.v1.kv_cache_interface import (
    KVCacheGroupSpec,
    KVCacheTensor,
    MambaSpec,
    MLAAttentionSpec,
)
from vllm.v1.request import Request

from vllm_ascend.core.kv_cache_interface import AscendIndexerKPoolTailSpec, is_prefix_cacheable
from vllm_ascend.models.glm5next.cache_config import (
    _get_glm5_next_cache_layout,
    get_glm5_next_fixed_pool_bytes,
    get_glm5_next_kv_cache_config,
    get_glm5_next_kv_cache_groups,
    get_glm5_next_max_memory_usage,
    get_glm5_next_pool_bytes_per_block,
)
from vllm_ascend.patch.platform import patch_kv_cache_coordinator
from vllm_ascend.utils import get_kv_cache_tensor_layers, vllm_version_is


def _ratio_kwargs(ratio: int) -> dict[str, int]:
    return {"tokens_per_state": ratio}


def make_config(*, retention_interval: int | None = 0):
    return SimpleNamespace(
        model_config=SimpleNamespace(max_model_len=2048),
        parallel_config=SimpleNamespace(
            decode_context_parallel_size=1,
            prefill_context_parallel_size=1,
        ),
        scheduler_config=SimpleNamespace(disable_hybrid_kv_cache_manager=False, max_num_seqs=4),
        max_in_flight_tokens=32,
        cache_config=SimpleNamespace(
            num_gpu_blocks_override=None,
            mamba_cache_mode="none",
            enable_prefix_caching=False,
            prefix_cache_retention_interval=retention_interval,
        ),
    )


def make_specs(pool: int = 16, block_size: int = 512):
    specs = {
        "model.layers.3.attn": MLAAttentionSpec(
            block_size=block_size,
            num_kv_heads=1,
            head_size=512,
            dtype=torch.bfloat16,
            model_version="glm5_next",
        ),
        "model.layers.3.indexer.k_cache": MLAAttentionSpec(
            block_size=block_size,
            num_kv_heads=1,
            head_size=128,
            dtype=torch.bfloat16,
            model_version="glm5_next",
            **_ratio_kwargs(pool),
        ),
        "model.layers.3.indexer.tail_cache": AscendIndexerKPoolTailSpec(
            block_size=pool,
            sliding_window=pool,
            compress_ratio=pool,
            num_kv_heads=1,
            head_size=128,
            dtype=torch.float32,
            model_version="glm5_next",
            indexes_kv_by_block_stride=True,
        ),
    }
    for layer_idx in range(3):
        specs[f"model.layers.{layer_idx}.linear_attn"] = MambaSpec(
            block_size=block_size,
            shapes=((3, 16), (1, 16, 16)),
            dtypes=(torch.bfloat16, torch.float32),
        )
    return specs


@pytest.fixture(scope="module", autouse=True)
def register_cache_specs():
    # Match production: vLLM registers built-in specs before the Ascend hook.
    register_all_kvcache_specs(None)


@pytest.mark.parametrize("pool", [4, 16])
def test_groups_share_block_ids_and_pack_two_page_classes(pool):
    config = make_config()
    specs = make_specs(pool)
    groups = get_glm5_next_kv_cache_groups(config, dict(reversed(list(specs.items()))))
    layout = _get_glm5_next_cache_layout(groups)
    assert layout is not None
    assert groups[0].layer_names == [
        "model.layers.3.attn",
        "model.layers.3.indexer.k_cache",
    ]
    assert len(groups) == 5  # full, state, and three interleaved KDA groups
    assert layout.main_slot_count == layout.small_slot_count == 1

    bytes_per_block = layout.main_page_size + layout.small_page_size
    fixed_bytes = sum(
        len(group.layer_names) * group.kv_cache_spec.page_size_bytes * config.scheduler_config.max_num_seqs
        for group in layout.mamba_groups
    )
    budget = 20 * bytes_per_block + fixed_bytes
    plan = get_glm5_next_kv_cache_config(config, groups, budget)
    assert plan.num_blocks == 20
    assert len(plan.kv_cache_tensors) == 5
    assert all(isinstance(tensor, KVCacheTensor) for tensor in plan.kv_cache_tensors)
    assert {tensor.size for tensor in plan.kv_cache_tensors} == {
        20 * layout.main_page_size,
        20 * layout.small_page_size,
        4 * layout.mamba_groups[0].kv_cache_spec.page_size_bytes,
    }
    if vllm_version_is("0.28.0"):
        assert all(tensor.block_stride == 0 for tensor in plan.kv_cache_tensors)
    else:
        assert all(tensor.offset == 0 and tensor.layer_stride == 0 for tensor in plan.kv_cache_tensors)

    placements = {
        layer_name: tensor for tensor in plan.kv_cache_tensors for layer_name in get_kv_cache_tensor_layers(tensor)
    }
    main = placements[layout.mla_names[0]]
    indexer = placements[layout.indexer_names[0]]
    state = placements[layout.state_names[0]]
    assert main.offset == 0
    assert indexer.offset == 0
    assert state is indexer
    assert get_kv_cache_tensor_layers(main) == [layout.mla_names[0]]
    assert all(
        placements[group.layer_names[0]].size
        == config.scheduler_config.max_num_seqs * group.kv_cache_spec.page_size_bytes
        for group in layout.mamba_groups
    )
    assert set(get_kv_cache_tensor_layers(indexer)) == {
        layout.indexer_names[0],
        layout.state_names[0],
    }

    # Scheduler groups consume disjoint IDs from the shared global BlockPool.
    required_blocks = sum(
        (group.kv_cache_spec.max_memory_usage_bytes(config) + group.kv_cache_spec.page_size_bytes - 1)
        // group.kv_cache_spec.page_size_bytes
        for group in groups
    ) + len(layout.mamba_groups)
    assert get_glm5_next_pool_bytes_per_block(groups) == bytes_per_block
    assert get_glm5_next_max_memory_usage(config, groups) == required_blocks * bytes_per_block + fixed_bytes


def test_standalone_mtp_layout_has_no_mamba_groups():
    specs = {name: spec for name, spec in make_specs().items() if not isinstance(spec, MambaSpec)}
    groups = get_glm5_next_kv_cache_groups(make_config(), specs)
    layout = _get_glm5_next_cache_layout(groups)
    assert len(groups) == 2
    assert layout is not None
    assert layout.mamba_groups == ()


@pytest.mark.parametrize("retention_interval", [None, 0, 4096])
def test_kv_cache_config_preserves_retention_interval(retention_interval):
    config = make_config(retention_interval=retention_interval)
    groups = get_glm5_next_kv_cache_groups(config, make_specs())
    layout = _get_glm5_next_cache_layout(groups)
    assert layout is not None

    budget = 10 * (layout.main_page_size + layout.small_page_size)
    plan = get_glm5_next_kv_cache_config(config, groups, budget)

    assert plan.prefix_cache_retention_interval == retention_interval


def test_worker_block_counts_retain_fixed_kda_reserve(monkeypatch):
    from vllm_ascend.patch.platform import patch_kv_cache_utils

    config = make_config()
    groups = get_glm5_next_kv_cache_groups(config, make_specs())
    fixed_bytes = get_glm5_next_fixed_pool_bytes(config, groups)
    block_bytes = get_glm5_next_pool_bytes_per_block(groups)
    worker_memory = [fixed_bytes + 100 * block_bytes, fixed_bytes + 99 * block_bytes]
    limiting = get_glm5_next_kv_cache_config(config, groups, worker_memory[1])
    # vLLM's generic shrinker omits the fixed live-state reserve here.
    broken = get_glm5_next_kv_cache_config(config, groups, limiting.num_blocks * block_bytes)
    assert broken.num_blocks < limiting.num_blocks
    monkeypatch.setattr(patch_kv_cache_utils, "_orig_get_kv_cache_configs", lambda *_: [broken, limiting])

    configs = patch_kv_cache_utils._ascend_get_kv_cache_configs(config, [make_specs(), make_specs()], worker_memory)

    assert [worker.num_blocks for worker in configs] == [99, 99]
    assert generate_scheduler_kv_cache_config(configs).num_blocks == 99
    assert all(
        sum(tensor.size for tensor in worker.kv_cache_tensors) <= available_memory
        for worker, available_memory in zip(configs, worker_memory)
    )


def test_pipeline_projection_supports_a_mamba_only_worker():
    config = make_config()
    groups = get_glm5_next_kv_cache_groups(config, make_specs())
    local_mamba_name = groups[2].layer_names[0]
    projected_groups = [
        KVCacheGroupSpec(
            [local_mamba_name] if group is groups[2] else [],
            group.kv_cache_spec,
        )
        for group in groups
    ]

    layout = _get_glm5_next_cache_layout(projected_groups)
    assert layout is not None
    assert layout.mla_names == layout.indexer_names == layout.state_names == ()
    assert layout.main_slot_count == 0
    assert layout.small_slot_count == 0

    budget = config.scheduler_config.max_num_seqs * groups[2].kv_cache_spec.page_size_bytes
    plan = get_glm5_next_kv_cache_config(config, projected_groups, available_memory=budget)
    assert plan.num_blocks == 1 + config.scheduler_config.max_num_seqs * len(projected_groups)
    assert len(plan.kv_cache_tensors) == 1
    assert get_kv_cache_tensor_layers(plan.kv_cache_tensors[0]) == [local_mamba_name]


def test_worker_block_counts_include_mamba_only_pipeline_rank(monkeypatch):
    from vllm_ascend.patch.platform import patch_kv_cache_utils

    config = make_config()
    groups = get_glm5_next_kv_cache_groups(config, make_specs())
    projected_groups = [
        KVCacheGroupSpec([groups[2].layer_names[0]] if group is groups[2] else [], group.kv_cache_spec)
        for group in groups
    ]
    full_memory = get_glm5_next_fixed_pool_bytes(config, groups) + 100 * get_glm5_next_pool_bytes_per_block(groups)
    mamba_memory = get_glm5_next_fixed_pool_bytes(config, projected_groups)
    full = get_glm5_next_kv_cache_config(config, groups, full_memory)
    mamba = get_glm5_next_kv_cache_config(config, projected_groups, mamba_memory)
    assert full.num_blocks != mamba.num_blocks
    monkeypatch.setattr(patch_kv_cache_utils, "_orig_get_kv_cache_configs", lambda *_: [full, mamba])

    configs = patch_kv_cache_utils._ascend_get_kv_cache_configs(
        config, [make_specs(), make_specs()], [full_memory, mamba_memory]
    )

    assert [worker.num_blocks for worker in configs] == [100, 100]
    assert generate_scheduler_kv_cache_config(configs).num_blocks == 100
    assert sum(tensor.size for tensor in configs[1].kv_cache_tensors) == mamba_memory


def test_long_context_keeps_kda_state_fixed_per_request():
    config = make_config()
    config.model_config.max_model_len = 262_144
    groups = get_glm5_next_kv_cache_groups(config, make_specs())
    layout = _get_glm5_next_cache_layout(groups)
    assert layout is not None and layout.compact_mamba
    assert layout.main_page_size == 512 * 512 * 2

    required = get_glm5_next_max_memory_usage(config, groups)
    plan = get_glm5_next_kv_cache_config(config, groups, required)
    mamba_names = {name for group in layout.mamba_groups for name in group.layer_names}
    mamba_tensors = [tensor for tensor in plan.kv_cache_tensors if get_kv_cache_tensor_layers(tensor)[0] in mamba_names]
    assert len(mamba_tensors) == len(mamba_names)
    assert all(
        tensor.size == config.scheduler_config.max_num_seqs * layout.mamba_groups[0].kv_cache_spec.page_size_bytes
        for tensor in mamba_tensors
    )
    assert plan.num_blocks >= config.model_config.max_model_len // 512


def test_host_mla_planner_reserves_logical_history_and_bounded_hot_cache(monkeypatch):
    monkeypatch.setenv("VLLM_ASCEND_310P_GLM_HOST_KV", "1")
    monkeypatch.setenv("VLLM_ASCEND_310P_ENABLE_MLA", "1")
    config = make_config()
    config.model_config.max_model_len = 262_144
    config.model_config.hf_text_config = SimpleNamespace(index_topk=2048, index_kpool=4)
    groups = get_glm5_next_kv_cache_groups(config, make_specs())
    layout = _get_glm5_next_cache_layout(groups)
    assert layout is not None
    required = get_glm5_next_max_memory_usage(config, groups)
    plan = get_glm5_next_kv_cache_config(config, groups, required)
    placements = {name: tensor for tensor in plan.kv_cache_tensors for name in get_kv_cache_tensor_layers(tensor)}
    main = placements[layout.mla_names[0]]
    indexer = placements[layout.indexer_names[0]]
    assert getattr(main, "glm_host_hot", False)
    assert not getattr(indexer, "glm_host_hot", False)
    assert main.size == 2064 * 32 * 512 * 2
    assert indexer.size == plan.num_blocks * layout.small_page_size
    assert plan.num_blocks >= config.model_config.max_model_len // 512
    assert get_glm5_next_pool_bytes_per_block(groups) == layout.small_page_size


def test_host_mla_scheduler_admits_four_256k_requests(monkeypatch):
    monkeypatch.setenv("VLLM_ASCEND_310P_GLM_HOST_KV", "1")
    monkeypatch.setenv("VLLM_ASCEND_310P_ENABLE_MLA", "1")
    config = make_config()
    config.model_config.max_model_len = 262_144
    config.model_config.hf_text_config = SimpleNamespace(index_topk=2048, index_kpool=4)
    config.max_in_flight_tokens = config.model_config.max_model_len
    groups = get_glm5_next_kv_cache_groups(config, make_specs())
    layout = _get_glm5_next_cache_layout(groups)
    assert layout is not None
    fixed = sum(
        len(group.layer_names) * group.kv_cache_spec.page_size_bytes * config.scheduler_config.max_num_seqs
        for group in layout.mamba_groups
    )
    hot = 2064 * 32 * 512 * 2
    block_bytes = get_glm5_next_pool_bytes_per_block(groups)
    one_request_blocks = (get_glm5_next_max_memory_usage(config, groups) - fixed - hot) // block_bytes
    budget = fixed + hot + (1 + 4 * one_request_blocks) * block_bytes
    plan = get_glm5_next_kv_cache_config(config, groups, budget)
    generous_plan = get_glm5_next_kv_cache_config(config, groups, budget * 2)
    assert generous_plan.num_blocks == plan.num_blocks
    manager = KVCacheManager(
        generate_scheduler_kv_cache_config([plan]),
        max_model_len=config.model_config.max_model_len,
        scheduler_block_size=512,
        hash_block_size=512,
        max_in_flight_tokens=config.max_in_flight_tokens,
        enable_caching=False,
    )
    for index in range(4):
        request = Request(str(index), [1] * config.model_config.max_model_len, SamplingParams(), None)
        assert manager.allocate_slots(request, num_new_tokens=request.num_tokens) is not None


def test_scheduler_admits_four_contexts_with_live_kda_state():
    config = make_config()
    config.model_config.max_model_len = 8_192
    config.max_in_flight_tokens = config.model_config.max_model_len
    groups = get_glm5_next_kv_cache_groups(config, make_specs())
    bytes_per_block = get_glm5_next_pool_bytes_per_block(groups)
    one_request_bytes = get_glm5_next_max_memory_usage(config, groups)
    layout = _get_glm5_next_cache_layout(groups)
    assert layout is not None
    fixed_bytes = sum(
        len(group.layer_names) * group.kv_cache_spec.page_size_bytes * config.scheduler_config.max_num_seqs
        for group in layout.mamba_groups
    )
    request_blocks = (one_request_bytes - fixed_bytes) // bytes_per_block
    plan = get_glm5_next_kv_cache_config(config, groups, (1 + 4 * request_blocks) * bytes_per_block + fixed_bytes)
    manager = KVCacheManager(
        generate_scheduler_kv_cache_config([plan]),
        max_model_len=config.model_config.max_model_len,
        scheduler_block_size=512,
        hash_block_size=512,
        max_in_flight_tokens=config.max_in_flight_tokens,
        enable_caching=False,
    )
    requests = [
        Request(str(index), [1] * config.model_config.max_model_len, SamplingParams(), None) for index in range(4)
    ]
    for request in requests:
        needed = [
            group_manager.get_num_blocks_to_allocate(
                request.request_id, request.num_tokens, [], 0, 0, request.num_tokens
            )
            for group_manager in manager.coordinator.single_type_managers
        ]
        free = manager.block_pool.get_num_free_blocks()
        assert manager.allocate_slots(request, num_new_tokens=request.num_tokens) is not None, (
            free,
            needed,
            request_blocks,
            plan.num_blocks,
        )


def test_fixed_kda_state_must_fit_before_planning_history():
    config = make_config()
    groups = get_glm5_next_kv_cache_groups(config, make_specs())
    layout = _get_glm5_next_cache_layout(groups)
    assert layout is not None
    fixed_bytes = sum(
        len(group.layer_names) * group.kv_cache_spec.page_size_bytes * config.scheduler_config.max_num_seqs
        for group in layout.mamba_groups
    )
    with pytest.raises(ValueError, match="live KDA state exceeds"):
        get_glm5_next_kv_cache_config(config, groups, fixed_bytes - 1)


def test_live_only_kda_rejects_prefix_caching():
    config = make_config()
    config.cache_config.enable_prefix_caching = True
    with pytest.raises(ValueError, match="does not support prefix caching"):
        get_glm5_next_kv_cache_groups(config, make_specs())


def test_aligned_kda_prefix_cache_uses_page_backed_state():
    config = make_config()
    config.cache_config.enable_prefix_caching = True
    config.cache_config.mamba_cache_mode = "align"
    specs = make_specs()
    for spec in specs.values():
        if isinstance(spec, MambaSpec):
            object.__setattr__(spec, "mamba_cache_mode", "align")

    groups = get_glm5_next_kv_cache_groups(config, specs)
    layout = _get_glm5_next_cache_layout(groups)
    assert layout is not None and not layout.compact_mamba
    assert get_glm5_next_fixed_pool_bytes(config, groups) == 0

    bytes_per_block = get_glm5_next_pool_bytes_per_block(groups)
    plan = get_glm5_next_kv_cache_config(config, groups, 20 * bytes_per_block)
    assert plan.num_blocks == 20
    placements = {name: tensor for tensor in plan.kv_cache_tensors for name in get_kv_cache_tensor_layers(tensor)}
    main = placements[layout.mla_names[0]]
    assert all(placements[name] is main for group in layout.mamba_groups for name in group.layer_names)

    manager = KVCacheManager(
        generate_scheduler_kv_cache_config([plan]),
        max_model_len=config.model_config.max_model_len,
        scheduler_block_size=512,
        hash_block_size=512,
        max_in_flight_tokens=config.max_in_flight_tokens,
        enable_caching=True,
    )
    assert len(manager.coordinator.single_type_managers) == len(groups)
    assert isinstance(manager.coordinator, patch_kv_cache_coordinator.AscendHybridKVCacheCoordinator)
    cacheable = [is_prefix_cacheable(group.kv_cache_spec) for group in groups]
    assert cacheable == [True, False, True, True, True]
    assert [m.enable_caching for m in manager.coordinator.single_type_managers] == cacheable


@pytest.mark.parametrize("block_size,pool", [(512, 16), (640, 4)])
def test_aligned_kda_prefix_cache_can_reuse_a_full_prompt_block(block_size: int, pool: int):
    init_none_hash(sha256)
    config = make_config()
    config.max_in_flight_tokens = block_size
    config.cache_config.enable_prefix_caching = True
    config.cache_config.mamba_cache_mode = "align"
    specs = make_specs(pool=pool, block_size=block_size)
    for spec in specs.values():
        if isinstance(spec, MambaSpec):
            object.__setattr__(spec, "mamba_cache_mode", "align")
    groups = get_glm5_next_kv_cache_groups(config, specs)
    bytes_per_block = get_glm5_next_pool_bytes_per_block(groups)
    plan = get_glm5_next_kv_cache_config(config, groups, 256 * bytes_per_block)
    manager = KVCacheManager(
        generate_scheduler_kv_cache_config([plan]),
        max_model_len=config.model_config.max_model_len,
        scheduler_block_size=block_size,
        hash_block_size=block_size,
        max_in_flight_tokens=config.max_in_flight_tokens,
        enable_caching=True,
    )

    def request(request_id: str) -> Request:
        return Request(
            request_id,
            [1] * (3 * block_size),
            SamplingParams(max_tokens=1),
            None,
            block_hasher=get_request_block_hasher(block_size, sha256),
        )

    first = request("prefix-a")
    assert manager.allocate_slots(first, num_new_tokens=block_size) is not None
    first.num_computed_tokens = block_size
    manager.new_step_starts()
    assert manager.allocate_slots(first, num_new_tokens=block_size) is not None
    manager.free(first)
    manager.new_step_starts()
    second = request("prefix-b")
    cached_blocks, cached_tokens, _ = manager.get_computed_blocks(second)
    assert cached_tokens >= block_size
    assert not cached_blocks.blocks[1]
    assert (
        manager.allocate_slots(
            second,
            num_new_tokens=block_size,
            num_new_computed_tokens=cached_tokens,
            new_computed_blocks=cached_blocks,
        )
        is not None
    )
    manager.free(second)


def test_live_only_kda_rejects_speculative_decoding():
    config = make_config()
    config.speculative_config = SimpleNamespace(num_speculative_tokens=1)
    with pytest.raises(ValueError, match="does not support speculative decoding"):
        get_glm5_next_kv_cache_groups(config, make_specs())


def test_missing_paired_cache_is_rejected():
    specs = make_specs()
    del specs["model.layers.3.indexer.tail_cache"]
    with pytest.raises(ValueError, match="requires"):
        get_glm5_next_kv_cache_groups(make_config(), specs)


def test_misaligned_logical_block_is_rejected():
    specs = make_specs(pool=16)
    object.__setattr__(
        specs["model.layers.3.indexer.k_cache"],
        "tokens_per_state",
        15,
    )
    with pytest.raises(ValueError, match="divisible"):
        get_glm5_next_kv_cache_groups(make_config(), specs)
