# SPDX-License-Identifier: Apache-2.0
"""CPU parity against the actual pool writer, without importing the NPU runtime."""

import __future__

import ast
import importlib.util
from pathlib import Path
from types import FunctionType, ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from tools.glm_perf.integer_divide import DivisionTorch, rewrite_pool_remainders
from tools.glm_perf.resident_candidates import kpool_completed_prefill as candidate

ROOT = Path(__file__).resolve().parents[3]


def test_resident_source_exec_with_inherited_postponed_annotations():
    module = ModuleType("private_completed_pools_resident_candidate")
    source = Path(candidate.__file__).read_text()
    exec(compile(source, "<resident>", "exec", flags=__future__.annotations.compiler_flag), module.__dict__)
    plan = module.make_plan(torch.tensor([0, 640]), 640, torch.device("cpu"))
    assert plan.complete.shape == (160, 3) and plan.tail.shape == (5, 2)


@pytest.fixture(scope="module")
def reference():
    path = ROOT / "vllm_ascend/models/glm5next/kpool_ops.py"
    spec = importlib.util.spec_from_file_location("cpu_kpool_ops", path)
    ops = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ops)
    path = ROOT / "vllm_ascend/models/glm5next/sparse_attn_indexer_kpool.py"
    source = ast.parse(path.read_text())
    functions = [
        node
        for node in source.body
        if isinstance(node, ast.FunctionDef) and node.name in ("_masked_storage_write", "_cache_tensor")
    ]
    cls = next(node for node in source.body if isinstance(node, ast.ClassDef) and node.name == "SparseAttnIndexerKpool")
    functions += [node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "_write_pools"]
    namespace = {"torch": torch, "compress_kpool": ops.compress_kpool}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec"), namespace)
    return SimpleNamespace(**namespace)


def inputs(lengths, phases):
    boundaries = [0]
    positions, state_slots, key_slots, raw_lengths = [], [], [], []
    for request, (length, start) in enumerate(zip(lengths, phases)):
        current = torch.arange(start, start + length)
        positions.extend(current.tolist())
        # Two recycled state pages per request retain both sides of an MTP1 boundary.
        state_slots.extend((request * 8 + current.remainder(8)).tolist())
        key_slots.extend(torch.where((current + 1) % 4 == 0, request * 256 + current // 4, -1).tolist())
        raw_lengths.append(start + length)
        boundaries.append(boundaries[-1] + length)
    metadata = SimpleNamespace(
        slot_mapping=torch.tensor(key_slots, dtype=torch.int32),
        raw_seq_lens=torch.tensor(raw_lengths, dtype=torch.int32),
        cum_query_lens=torch.tensor(boundaries[1:], dtype=torch.int32),
    )
    return (
        torch.tensor(boundaries, dtype=torch.int32),
        torch.tensor(positions, dtype=torch.int32),
        metadata,
        SimpleNamespace(slot_mapping=torch.tensor(state_slots, dtype=torch.int32)),
    )


def caches(requests, dtype, seed):
    generator = torch.Generator().manual_seed(seed)
    # Include storage offsets, page gaps, row gaps and element strides. Compare
    # the entire allocation afterwards so writes into guards cannot hide.
    state_backing = torch.randn(requests * 2 + 2, 8, 512, generator=generator)
    key_backing = torch.randn(requests + 2, 512, 1, 256, generator=generator).to(dtype)
    return state_backing, key_backing, state_backing[1:-1, 1::2, ::2], key_backing[1:-1, ::2, :, ::2]


@pytest.mark.parametrize("phase", range(4))
@pytest.mark.parametrize("speculative", [0, 1])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("lengths", [[1, 2, 3], [0, 9, 0, 17], [640], [11, 13, 18]])
@pytest.mark.parametrize("use_integer_ops", [False, True])
def test_strided_caches_exact_for_all_pool_phases(reference, phase, speculative, dtype, lengths, use_integer_ops):
    phases = [phase + 4 * request for request in range(len(lengths))]
    boundaries, positions, metadata, state_metadata = inputs(lengths, phases)
    expected = caches(len(lengths), dtype, 73)
    actual = caches(len(lengths), dtype, 73)
    generator = torch.Generator().manual_seed(13)
    keys = torch.randn(sum(lengths), 256, generator=generator)[:, ::2].bfloat16()
    gates = torch.randn(sum(lengths), 256, generator=generator)[:, ::2]
    ape = torch.randn(4, 128, generator=generator)
    indexer = SimpleNamespace(
        head_dim=128,
        tail_cache=SimpleNamespace(kv_cache=expected[2], num_speculative_tokens=speculative),
        k_cache=SimpleNamespace(kv_cache=expected[3]),
    )
    reference._write_pools(indexer, keys, gates, ape, positions, 4, metadata, state_metadata)

    class CpuIntegerNative:
        device = torch.device("cpu")
        calls = 0

        def __call__(self, values, divisor):
            self.calls += 1
            return torch.div(values, divisor, rounding_mode="floor")

    native = CpuIntegerNative()
    candidate.write_compact_pools(
        keys,
        gates,
        ape,
        positions,
        metadata,
        state_metadata,
        actual[2],
        actual[3],
        candidate.make_plan(boundaries, sum(lengths), keys.device),
        speculative,
        compress=reference.compress_kpool,
        storage_write=reference._masked_storage_write,
        integer_ops=DivisionTorch(native) if use_integer_ops else None,
    )
    assert (native.calls > 0) == use_integer_ops
    assert torch.equal(actual[0], expected[0])
    assert torch.equal(actual[1], expected[1])


def test_continued_pool_and_reused_pages(reference):
    actual, expected = caches(1, torch.bfloat16, 12), caches(1, torch.bfloat16, 12)
    generator = torch.Generator().manual_seed(77)
    ape = torch.randn(4, 128, generator=generator)
    # Continuation at every possible phase, then reuse pages for a new request.
    for start, length in [(0, 13), (13, 10), (23, 18), (41, 11), (0, 19)]:
        boundaries, positions, metadata, state_meta = inputs([length], [start])
        keys = torch.randn(length, 128, generator=generator).bfloat16()
        gates = torch.randn(length, 128, generator=generator)
        indexer = SimpleNamespace(
            head_dim=128,
            tail_cache=SimpleNamespace(kv_cache=expected[2], num_speculative_tokens=1),
            k_cache=SimpleNamespace(kv_cache=expected[3]),
        )
        reference._write_pools(indexer, keys, gates, ape, positions, 4, metadata, state_meta)
        candidate.write_compact_pools(
            keys,
            gates,
            ape,
            positions,
            metadata,
            state_meta,
            actual[2],
            actual[3],
            candidate.make_plan(boundaries, length, keys.device),
            1,
            compress=reference.compress_kpool,
            storage_write=reference._masked_storage_write,
        )
        assert torch.equal(actual[0], expected[0])
        assert torch.equal(actual[1], expected[1])


def test_shared_physical_backing_and_wrapper_dispatch(reference):
    # Model cache groups have distinct live pages in a common allocation.
    # Poison/gaps in that allocation catch incorrect view-relative offsets.
    generator = torch.Generator().manual_seed(42)
    expected = torch.randn(32768, generator=generator)
    actual = expected.clone()

    def indexer(backing):
        state = backing.as_strided((2, 4, 256), (2048, 256, 1), storage_offset=128)
        keys = backing.view(torch.bfloat16).as_strided((2, 256, 1, 128), (32768, 128, 128, 1))
        return SimpleNamespace(
            head_dim=128,
            tail_cache=SimpleNamespace(kv_cache=state, num_speculative_tokens=1),
            k_cache=SimpleNamespace(kv_cache=keys),
        )

    boundaries, positions, metadata, state_meta = inputs([21], [3])
    # Key page 1; state pages 0/1 occupy a disjoint prefix in the allocation.
    metadata.slot_mapping[metadata.slot_mapping >= 0] += 256
    metadata._glm_completed_pool_plan = candidate.make_plan(boundaries, 21, positions.device)
    keys = torch.randn(21, 128, generator=generator).bfloat16()
    gates = torch.randn(21, 128, generator=generator)
    ape = torch.randn(4, 128, generator=generator)
    reference._write_pools(indexer(expected), keys, gates, ape, positions, 4, metadata, state_meta)
    write = candidate.wrap_writer(
        reference._write_pools, reference._cache_tensor, reference._masked_storage_write, reference.compress_kpool
    )
    write(indexer(actual), keys, gates, ape, positions, 4, metadata, state_meta)
    # Compare bytes, since BF16 pairs viewed as FP32 can encode NaN.
    assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_bound_compressor_retains_cache_output_dtype(reference, dtype):
    boundaries, positions, metadata, state_meta = inputs([21], [3])
    expected, actual = caches(1, dtype, 12), caches(1, dtype, 12)
    keys = torch.randn(21, 128)
    gates = torch.randn_like(keys)
    ape = torch.randn(4, 128)
    metadata._glm_completed_pool_plan = candidate.make_plan(boundaries, 21, keys.device)

    def indexer(cache):
        return SimpleNamespace(
            head_dim=128,
            tail_cache=SimpleNamespace(kv_cache=cache[2], num_speculative_tokens=1),
            k_cache=SimpleNamespace(kv_cache=cache[3]),
        )

    def compress(pool_keys, pool_gates, ape, *, out_dtype):
        assert out_dtype == dtype
        return reference.compress_kpool(pool_keys, pool_gates, ape).to(out_dtype)

    reference._write_pools(indexer(expected), keys, gates, ape, positions, 4, metadata, state_meta)
    writer = candidate.wrap_writer(
        reference._write_pools,
        reference._cache_tensor,
        reference._masked_storage_write,
        compress,
        preserve_cache_dtype=True,
    )
    writer(indexer(actual), keys, gates, ape, positions, 4, metadata, state_meta)
    assert all(torch.equal(a.view(torch.uint8), b.view(torch.uint8)) for a, b in zip(actual[:2], expected[:2]))


@pytest.mark.parametrize("lengths", [[0], [0, 0], [1, 1], [640]])
def test_invalid_slots_leave_backing_untouched(reference, lengths):
    boundaries, positions, metadata, state_meta = inputs(lengths, [0] * len(lengths))
    metadata.slot_mapping.fill_(-1)
    state_meta.slot_mapping.fill_(-1)
    actual = caches(len(lengths), torch.bfloat16, 73)
    before = [x.clone() for x in actual[:2]]
    keys = torch.zeros(sum(lengths), 128)
    candidate.write_compact_pools(
        keys,
        keys,
        torch.zeros(4, 128),
        positions,
        metadata,
        state_meta,
        actual[2],
        actual[3],
        candidate.make_plan(boundaries, sum(lengths), keys.device),
        1,
        compress=reference.compress_kpool,
        storage_write=reference._masked_storage_write,
    )
    assert all(torch.equal(a, b) for a, b in zip(actual, before))


def test_prefill_work_reduction_and_no_scalar_reads(reference, monkeypatch):
    boundaries, positions, metadata, state_meta = inputs([640], [0])
    actual = caches(1, torch.bfloat16, 12)
    keys = torch.zeros(640, 128, dtype=torch.bfloat16)
    plan = candidate.make_plan(boundaries, 640, keys.device)
    compress = Mock(wraps=reference.compress_kpool)
    # Spy on writes without CPU oracle's boolean filtering.
    write = Mock()
    monkeypatch.setattr(torch.Tensor, "item", Mock(side_effect=AssertionError("device scalar read")))
    monkeypatch.setattr(torch.Tensor, "tolist", Mock(side_effect=AssertionError("device list read")))
    index_select = torch.Tensor.index_select

    def supported_index_select(tensor, *args, **kwargs):
        if tensor.dtype == torch.bfloat16:
            raise AssertionError("310P aclnnIndexSelect does not support BF16")
        return index_select(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "index_select", supported_index_select)
    candidate.write_compact_pools(
        keys,
        keys,
        torch.zeros(4, 128),
        positions,
        metadata,
        state_meta,
        actual[2],
        actual[3],
        plan,
        1,
        compress=compress,
        storage_write=write,
    )
    assert compress.call_args.args[0].shape == (160, 4, 128)
    assert [call.args[2].shape for call in write.call_args_list] == [(5, 256), (160, 128)]


@pytest.mark.parametrize("boundaries,count", [([], 0), ([1, 9], 9), ([0, 8], 9), ([0, 8, 3, 9], 9)])
def test_invalid_partition_rejected(boundaries, count):
    with pytest.raises(ValueError, match="partition"):
        candidate.plan_rows(boundaries, count)


def test_builder_reuse_and_missing_cpu_metadata():
    metadata = SimpleNamespace(
        num_actual_tokens=12, compress_ratio=4, cum_query_lens_cpu=torch.tensor([0, 12]), positions=torch.arange(12)
    )
    original = Mock(return_value=metadata)
    first = candidate.wrap_builder(original)
    wrapped = candidate.wrap_builder(first)
    assert wrapped.__glm_resident_original__ is original
    assert wrapped(None, 0, None)._glm_completed_pool_plan.num_tokens == 12
    metadata.cum_query_lens_cpu = None
    assert wrapped(None, 0, None)._glm_completed_pool_plan is None
    metadata.cum_query_lens_cpu = torch.tensor([0, 8])
    metadata.num_actual_tokens = 8
    assert wrapped(None, 0, None)._glm_completed_pool_plan is None
    assert original.call_count == 3


@pytest.mark.parametrize(
    "tokens,pool,speculative,planned",
    [(2, 4, 1, True), (8, 4, 1, True), (12, 4, 2, True), (12, 2, 1, True), (12, 4, 1, False)],
)
def test_decode_and_unsupported_cases_use_original(tokens, pool, speculative, planned):
    original = Mock(return_value="fallback")
    cache = Mock(side_effect=AssertionError("must not inspect caches"))
    first = candidate.wrap_writer(original, cache, Mock(), Mock())
    wrapped = candidate.wrap_writer(first, cache, Mock(), Mock(), preserve_cache_dtype=True)
    metadata = SimpleNamespace(_glm_completed_pool_plan=object() if planned else None)
    indexer = SimpleNamespace(tail_cache=SimpleNamespace(num_speculative_tokens=speculative))
    keys = torch.empty(tokens, 128)
    assert wrapped(indexer, keys, keys, None, None, pool, metadata, None) == "fallback"
    original.assert_called_once()
    cache.assert_not_called()


@pytest.mark.parametrize("length,start", [(2, 3), (8, 0), (8, 3), (640, 1)])
@pytest.mark.parametrize("speculative", [0, 1])
@pytest.mark.parametrize("permanent", [False, True])
def test_rewritten_pool_remainders_preserve_full_strided_backings(reference, length, start, speculative, permanent):
    class Native:
        device = torch.device("cpu")

        def __call__(self, value, divisor):
            return torch.div(value, divisor, rounding_mode="floor")

    original = reference._write_pools
    if permanent:
        # Native production writers carry these converter bindings privately.
        namespace = dict(original.__globals__)
        namespace["_aicore_convert"] = lambda value, dtype: value.to(dtype)
        namespace["compress_kpool"] = lambda keys, gates, ape, out_dtype: reference.compress_kpool(keys, gates, ape).to(
            out_dtype
        )
        original = FunctionType(original.__code__, namespace, original.__name__)
    source = (ROOT / "vllm_ascend/models/glm5next/sparse_attn_indexer_kpool.py").read_text()
    writer = rewrite_pool_remainders(original, source, DivisionTorch(Native()))
    expected, actual = caches(1, torch.float16, 12), caches(1, torch.float16, 12)
    _, positions, metadata, state_meta = inputs([length], [start])
    keys = torch.randn(length, 128).half()
    gates = torch.randn(length, 128)
    ape = torch.randn(4, 128)

    def indexer(cache):
        return SimpleNamespace(
            head_dim=128,
            tail_cache=SimpleNamespace(kv_cache=cache[2], num_speculative_tokens=speculative),
            k_cache=SimpleNamespace(kv_cache=cache[3]),
        )

    reference._write_pools(indexer(expected), keys, gates, ape, positions, 4, metadata, state_meta)
    writer(indexer(actual), keys, gates, ape, positions, 4, metadata, state_meta)
    assert all(torch.equal(a.view(torch.uint8), b.view(torch.uint8)) for a, b in zip(actual[:2], expected[:2]))
    assert writer.__glm_remainder_original__ is original


def test_remainder_rewrite_rejects_changed_authoritative_writer(reference):
    source = (ROOT / "vllm_ascend/models/glm5next/sparse_attn_indexer_kpool.py").read_text()
    source = source.replace(
        "state_offsets = safe_state_slots - state_blocks * state_block_size",
        "state_offsets = safe_state_slots.remainder(state_block_size)",
    )
    with pytest.raises(ValueError, match="expressions changed"):
        rewrite_pool_remainders(reference._write_pools, source, DivisionTorch(None))
