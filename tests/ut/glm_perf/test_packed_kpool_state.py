# SPDX-License-Identifier: Apache-2.0
"""Real ordinary/resident writers against logical history and poisoned pages."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tests.ut.glm_perf import test_kpool_completed_prefill as shared_reference
from tools.glm_perf.integer_divide import DivisionTorch, rewrite_pool_remainders
from tools.glm_perf.prefill_memory_budget import PrefillConfig, coarser_state_page_design
from tools.glm_perf.resident_candidates import kpool_completed_prefill as compact

ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture(name="reference")
def writer_reference():
    return shared_reference.reference.__wrapped__()


@pytest.fixture
def cache_module():
    path = ROOT / "vllm_ascend/models/glm5next/kv_cache.py"
    tree = ast.parse(path.read_text())
    nodes = [
        n
        for n in tree.body
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id.startswith("PACKED_KPOOL_") for t in n.targets)
        or isinstance(n, ast.FunctionDef)
        and n.name == "get_kpool_state_block_size"
        or isinstance(n, ast.ClassDef)
        and n.name == "Glm5NextStateCache"
    ]
    state = {"config": None, "is_310p": True}
    namespace = {
        "torch": torch,
        "nn": torch.nn,
        "AttentionLayerBase": object,
        "get_current_vllm_config": lambda: state["config"],
        "is_310p": lambda: state["is_310p"],
        "AscendIndexerKPoolStateSpec": lambda **kwargs: SimpleNamespace(**kwargs),
    }
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    exec(
        compile(ast.fix_missing_locations(ast.Module(body=[future, *nodes], type_ignores=[])), str(path), "exec"),
        namespace,
    )
    return SimpleNamespace(**namespace), state


@pytest.mark.parametrize("block", [None, 4, 32])
@pytest.mark.parametrize("speculative", [0, 1])
def test_real_state_layer_target_and_draft_specs(cache_module, block, speculative):
    module, state = cache_module
    text = SimpleNamespace() if block is None else SimpleNamespace(ascend_glm_kpool_state_block_size=block)
    state["config"] = SimpleNamespace(
        model_config=SimpleNamespace(hf_text_config=text),
        speculative_config=SimpleNamespace(num_speculative_tokens=speculative),
        parallel_config=SimpleNamespace(pipeline_parallel_size=1),
        compilation_config=SimpleNamespace(static_forward_context={}),
    )
    for name in ("target.indexer.state", "draft.indexer.state"):
        layer = module.Glm5NextStateCache(
            state_dim=256, dtype=torch.float32, compress_ratio=4, cache_config=object(), prefix=name
        )
        spec = layer.get_kv_cache_spec(state["config"])
        assert spec.block_size == (4 if block is None else block)
        assert spec.sliding_window == spec.block_size + speculative
        assert spec.dtype == torch.float32 and spec.head_size == 256
        assert layer.compress_ratio == 4 and spec.indexes_kv_by_block_stride
    assert len(state["config"].compilation_config.static_forward_context) == 2


@pytest.mark.parametrize("block", [0, -1, True, 8, 16, 64, 32.0, "32"])
def test_unqualified_state_config_rejected_before_allocation(cache_module, block):
    module, _ = cache_module
    with pytest.raises(ValueError):
        module.get_kpool_state_block_size(SimpleNamespace(ascend_glm_kpool_state_block_size=block), 4, 256)


def test_flat_wrapper_override_reaches_real_state_constructor(cache_module):
    module, state = cache_module
    state["config"] = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(ascend_glm_kpool_state_block_size=32),
            hf_text_config=SimpleNamespace(ascend_glm_kpool_state_block_size=4),
        ),
        speculative_config=SimpleNamespace(num_speculative_tokens=1),
        parallel_config=SimpleNamespace(pipeline_parallel_size=1),
        compilation_config=SimpleNamespace(static_forward_context={}),
    )
    layer = module.Glm5NextStateCache(
        state_dim=256, dtype=torch.float32, compress_ratio=4, cache_config=object(), prefix="target.state"
    )
    assert layer.get_kv_cache_spec(state["config"]).block_size == 32


@pytest.mark.parametrize("ratio,width,hardware", [(8, 256, True), (4, 128, True), (4, 256, False)])
def test_packed_opt_in_rejects_wrong_geometry_or_hardware(cache_module, ratio, width, hardware):
    module, state = cache_module
    state["is_310p"] = hardware
    with pytest.raises(ValueError, match="packed GLM"):
        module.get_kpool_state_block_size(SimpleNamespace(ascend_glm_kpool_state_block_size=32), ratio, width)


class CpuIntegerNative:
    device = torch.device("cpu")

    def __init__(self):
        self.divisors = []

    def __call__(self, values, divisor):
        self.divisors.append(divisor)
        return torch.div(values, divisor, rounding_mode="floor")


@pytest.mark.parametrize("writer", ["ordinary", "rewritten", "compact", "compact_integer", "bound"])
@pytest.mark.parametrize("key_dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("speculative", [0, 1])
def test_real_writers_preserve_logical_history_prefixes_rejection_and_poison(reference, writer, key_dtype, speculative):
    generator = torch.Generator().manual_seed(97)
    requests, page_stride, storage_offset = 2, 10240, 13
    backing = torch.randn((requests * 3 + 1) * page_stride + storage_offset, generator=generator)
    expected_backing = backing.clone()
    state = backing.as_strided((requests * 3 + 1, 32, 256), (page_stride, 256, 1), storage_offset=storage_offset)
    expected_state = expected_backing.as_strided(state.shape, state.stride(), storage_offset=storage_offset)
    cache = torch.randn(requests, 1024, 1, 128, generator=generator).to(key_dtype)
    expected_cache = cache.clone()
    keys = torch.randn(requests, 4096, 128, generator=generator).bfloat16()
    gates = torch.randn(requests, 4096, 128, generator=generator)
    history_k, history_g = keys.clone(), gates.clone()
    ape = torch.randn(4, 128, generator=generator)
    indexer = SimpleNamespace(
        head_dim=128,
        tail_cache=SimpleNamespace(kv_cache=state, num_speculative_tokens=speculative),
        k_cache=SimpleNamespace(kv_cache=cache),
    )
    native = CpuIntegerNative()
    proxy = DivisionTorch(native)
    source = (ROOT / "vllm_ascend/models/glm5next/sparse_attn_indexer_kpool.py").read_text()
    rewritten = rewrite_pool_remainders(reference._write_pools, source, proxy)
    bound = compact.wrap_writer(
        reference._write_pools,
        reference._cache_tensor,
        reference._masked_storage_write,
        reference.compress_kpool,
        integer_ops=proxy,
    )

    def physical(request, position):
        return request * 3 + 1 + (position // 32) % 2

    # Rejection can overwrite an already-completed pool at position 31. Prefix
    # reuse resumes at an aligned 640 boundary without restoring state pages.
    steps = [
        ([0, 0], [1280, 0], False),
        ([0, 0], [32, 29], True),
        ([31, 28], [10, 9], False),
        ([41, 37], [1239, 1243], False),
        ([1280, 1280], [17, 8], False),
        ([1297, 1288], [1, 0], False),
        ([640, 640], [640, 640], False),
        ([0, 0], [13, 11], False),
    ]
    for starts, lengths, draft in steps:
        boundaries, positions, slots, key_slots, rows_k, rows_g = [0], [], [], [], [], []
        for request, (start, length) in enumerate(zip(starts, lengths)):
            end = start + length
            batch_k, batch_g = keys[request, start:end].clone(), gates[request, start:end].clone()
            if draft:
                batch_k[-1] += 1
                batch_g[-1] += 2
            history_k[request, start:end] = batch_k
            history_g[request, start:end] = batch_g
            rows_k.append(batch_k)
            rows_g.append(batch_g)
            for position in range(start, end):
                positions.append(position)
                # Sentinel rows must never write a clamped page-zero address.
                masked = request == 1 and start == 0 and length == 11 and position == end - 1
                slots.append(-1 if masked else physical(request, position) * 32 + position % 32)
                key_slots.append(request * 1024 + position // 4 if (position + 1) % 4 == 0 else -1)
                if (position + 1) % 4 == 0:
                    pool_start = position // 4 * 4
                    expected_cache[request, position // 4, 0] = reference.compress_kpool(
                        history_k[request, pool_start : pool_start + 4][None],
                        history_g[request, pool_start : pool_start + 4][None],
                        ape,
                    )[0].to(key_dtype)
            first_retained = max(0, end - 1 - speculative) // 4 * 4
            for position in range(max(start, first_retained), end):
                if request == 1 and start == 0 and length == 11 and position == end - 1:
                    continue
                expected_state[physical(request, position), position % 32] = torch.cat(
                    (history_k[request, position].float(), history_g[request, position].float())
                )
            boundaries.append(boundaries[-1] + length)
        actual_k, actual_g = torch.cat(rows_k), torch.cat(rows_g)
        positions = torch.tensor(positions, dtype=torch.int64)
        meta = SimpleNamespace(
            slot_mapping=torch.tensor(key_slots),
            raw_seq_lens=torch.tensor([a + b for a, b in zip(starts, lengths)]),
            cum_query_lens=torch.tensor(boundaries[1:], dtype=torch.int32),
        )
        state_meta = SimpleNamespace(slot_mapping=torch.tensor(slots), block_size=32)
        plan = compact.make_plan(torch.tensor(boundaries, dtype=torch.int32), len(slots), torch.device("cpu"))
        meta._glm_completed_pool_plan = plan
        if writer in ("ordinary", "rewritten", "bound"):
            fn = {"ordinary": reference._write_pools, "rewritten": rewritten, "bound": bound}[writer]
            fn(indexer, actual_k, actual_g, ape, positions, 4, meta, state_meta)
        else:
            compact.write_compact_pools(
                actual_k,
                actual_g,
                ape,
                positions,
                meta,
                state_meta,
                state,
                cache,
                plan,
                speculative,
                compress=reference.compress_kpool,
                storage_write=reference._masked_storage_write,
                integer_ops=proxy if writer == "compact_integer" else None,
            )
        assert torch.equal(backing, expected_backing), "state values, padded page gaps and allocation guards"
        assert torch.equal(cache, expected_cache), "compressed keys including overwritten speculative pools"
    if writer in ("rewritten", "compact_integer", "bound"):
        assert 32 in native.divisors


def test_admission_uses_actual_opt_in_spec_geometry(cache_module):
    module, _ = cache_module
    actual = module.get_kpool_state_block_size(SimpleNamespace(ascend_glm_kpool_state_block_size=32), 4, 256)
    assert actual == 32
    result = coarser_state_page_design(PrefillConfig(1280, 131072))
    assert result["compressor_tail_ids"] == 42 and result["minimum_cache_bytes"] == 2139095040


@pytest.mark.parametrize("writer", ["ordinary", "compact"])
def test_state_and_indexer_alias_one_physical_small_pool(reference, writer):
    backing = torch.full((4 * 40960 + 64,), 0xA5, dtype=torch.uint8)
    raw = backing[32:-32]
    state = raw.view(torch.float32).as_strided((4, 32, 256), (10240, 256, 1))
    cache = raw.view(torch.float16).as_strided((4, 160, 1, 128), (20480, 128, 128, 1))
    generator = torch.Generator().manual_seed(77)
    keys = torch.randn(37, 128, generator=generator).bfloat16()
    gates = torch.randn(37, 128, generator=generator)
    ape = torch.randn(4, 128, generator=generator)
    for pos in (24, 25, 26):
        state[2, pos] = torch.cat((keys[pos].float(), gates[pos].float()))
    expected_bytes = backing.clone()
    expected_raw = expected_bytes[32:-32]
    expected_state = expected_raw.view(torch.float32).as_strided(state.shape, state.stride())
    expected_cache = expected_raw.view(torch.float16).as_strided(cache.shape, cache.stride())
    for pos in (27, 31, 35):
        expected_cache[1, pos // 4, 0] = reference.compress_kpool(
            keys[pos - 3 : pos + 1][None], gates[pos - 3 : pos + 1][None], ape
        )[0].half()
    for pos in range(32, 37):
        expected_state[3, pos % 32] = torch.cat((keys[pos].float(), gates[pos].float()))
    positions = torch.arange(27, 37)
    state_meta = SimpleNamespace(block_size=32, slot_mapping=torch.where(positions < 32, 2, 3) * 32 + positions % 32)
    meta = SimpleNamespace(
        slot_mapping=torch.where((positions + 1) % 4 == 0, 160 + positions // 4, -1),
        raw_seq_lens=torch.tensor([37]),
        cum_query_lens=torch.tensor([10]),
    )
    if writer == "ordinary":
        indexer = SimpleNamespace(
            head_dim=128,
            tail_cache=SimpleNamespace(kv_cache=state, num_speculative_tokens=1),
            k_cache=SimpleNamespace(kv_cache=cache),
        )
        reference._write_pools(indexer, keys[27:], gates[27:], ape, positions, 4, meta, state_meta)
    else:
        plan = compact.make_plan(torch.tensor([0, 10], dtype=torch.int32), 10, torch.device("cpu"))
        compact.write_compact_pools(
            keys[27:],
            gates[27:],
            ape,
            positions,
            meta,
            state_meta,
            state,
            cache,
            plan,
            1,
            compress=reference.compress_kpool,
            storage_write=reference._masked_storage_write,
        )
    assert torch.equal(backing, expected_bytes)


@pytest.mark.parametrize("shape,metadata_block", [((1, 32, 256), 4), ((1, 31, 256), 31), ((1, 32, 255), 32)])
def test_writer_geometry_mismatch_rejected_before_writes(reference, shape, metadata_block):
    state = torch.full(shape, 19.0)
    cache = torch.full((1, 8, 1, 128), 29.0)
    indexer = SimpleNamespace(
        head_dim=128, tail_cache=SimpleNamespace(kv_cache=state), k_cache=SimpleNamespace(kv_cache=cache)
    )
    meta = SimpleNamespace(
        slot_mapping=torch.tensor([-1]), raw_seq_lens=torch.tensor([1]), cum_query_lens=torch.tensor([1])
    )
    state_meta = SimpleNamespace(block_size=metadata_block, slot_mapping=torch.tensor([0]))
    plan = compact.make_plan(torch.tensor([0, 1], dtype=torch.int32), 1, torch.device("cpu"))
    args = (torch.zeros(1, 128), torch.zeros(1, 128), torch.zeros(4, 128), torch.tensor([0]))
    with pytest.raises(RuntimeError, match="geometry"):
        reference._write_pools(indexer, *args, 4, meta, state_meta)
    with pytest.raises(ValueError, match="geometry"):
        compact.write_compact_pools(
            *args,
            meta,
            state_meta,
            state,
            cache,
            plan,
            1,
            compress=reference.compress_kpool,
            storage_write=reference._masked_storage_write,
        )
    assert (state == 19).all() and (cache == 29).all()


@pytest.mark.parametrize(
    "block,ratio,width,window,accepted",
    [
        (4, 4, 256, 5, True),
        (32, 4, 256, 33, True),
        (16, 16, 256, 16, True),
        (32, 4, 256, 5, False),
        (32, 4, 128, 33, False),
        (32, 8, 256, 33, False),
        (16, 4, 256, 17, False),
    ],
)
def test_real_cache_grouping_admits_only_supported_state_geometry(cache_module, block, ratio, width, window, accepted):
    module, _ = cache_module
    path = ROOT / "vllm_ascend/models/glm5next/cache_config.py"
    tree = ast.parse(path.read_text())
    function = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_create_glm5_next_attention_groups"
    )

    class Spec(SimpleNamespace):
        pass

    class Uniform:
        @staticmethod
        def from_specs(specs):
            return SimpleNamespace(kv_cache_specs=specs)

    specs = {
        "main": Spec(block_size=640),
        "indexer": Spec(block_size=640, ratio=ratio),
        "state": Spec(block_size=block, sliding_window=window, head_size=width, num_kv_heads=1),
    }
    namespace = dict(
        vars(module),
        MLAAttentionSpec=Spec,
        SlidingWindowMLASpec=Spec,
        _is_glm5_next_main_spec=lambda s: s is specs["main"],
        _is_glm5_next_indexer_spec=lambda s: s is specs["indexer"],
        _is_glm5_next_state_spec=lambda s: s is specs["state"],
        _sorted_layer_names=lambda names: tuple(names),
        _layer_indices=lambda names: None,
        get_kv_cache_compression_ratio=lambda spec: spec.ratio,
        UniformTypeKVCacheSpecs=Uniform,
        KVCacheGroupSpec=lambda names, spec: SimpleNamespace(layer_names=names, kv_cache_spec=spec),
    )
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    exec(
        compile(ast.fix_missing_locations(ast.Module(body=[future, function], type_ignores=[])), str(path), "exec"),
        namespace,
    )
    if accepted:
        groups = namespace[function.name](specs)
        assert len(groups) == 2 and groups[1].kv_cache_spec.kv_cache_specs["state"] is specs["state"]
    else:
        with pytest.raises(ValueError, match="qualified"):
            namespace[function.name](specs)
