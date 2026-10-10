# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run production NZ attention orchestration against a CPU gather oracle."""

import ast
import math
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

OPS_PATH = Path(__file__).resolve().parents[3] / "vllm_ascend/models/qwen4_exp/ops"


class CPUStorageNPUTensor(torch.Tensor):
    """CPU storage with only the device contract replaced for this harness."""

    @property
    def device(self):
        return SimpleNamespace(type="npu")

    def record_stream(self, stream):
        stream.recorded.append(self)


class CastCounter(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.int32_copies = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if func == torch.ops.aten._to_copy.default and kwargs.get("dtype") == torch.int32:
            self.int32_copies += 1
        return func(*args, **kwargs)


def _load_functions(path, names, namespace):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    definitions = [node for node in tree.body if getattr(node, "name", None) in names]
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(path), "exec"), namespace)


def _production_namespace(monkeypatch):
    gather_calls = []

    def gather_oracle(cache, groups, counts, tail_starts, tail_counts, table, output, heads, dim, transpose):
        # Independently implement page translation and causal selection. NZ
        # storage is represented as its logical shape by this CPU-only oracle.
        assert all(
            t.dtype == torch.int32 and t.is_contiguous() for t in (groups, counts, tail_starts, tail_counts, table)
        )
        gather_calls.append((groups, counts, tail_starts, tail_counts, table, transpose))
        raw = cache.as_subclass(torch.Tensor).permute(0, 2, 1, 3).reshape(-1, heads, dim)
        output.zero_()
        block_size = cache.shape[2]
        for row in range(groups.shape[0]):
            tokens = [
                (group_rank * 4 + offset, int(group) * 4 + offset)
                for group_rank, group in enumerate(groups[row, : counts[row]])
                for offset in range(4)
            ]
            tokens += [
                (groups.shape[1] * 4 + offset, int(tail_starts[row]) + offset)
                for offset in range(int(tail_counts[row]))
            ]
            for lane, token in tokens:
                slot = int(table[0, token // block_size]) * block_size + token % block_size
                if transpose:
                    output[row, :, :, lane] = raw[slot]
                else:
                    output[row, :, lane, :] = raw[slot]

    class CPUFactories:
        Tensor = torch.Tensor
        ops = SimpleNamespace(_C_ascend=SimpleNamespace(qsa_gather_value_nz_310=gather_oracle))

        def __getattr__(self, name):
            return getattr(torch, name)

        def arange(self, *args, **kwargs):
            kwargs["device"] = "cpu"
            return torch.arange(*args, **kwargs).as_subclass(CPUStorageNPUTensor)

    def empty_with_format(*, size, dtype, device, acl_format):
        assert acl_format == 29
        return torch.empty(size, dtype=dtype).as_subclass(CPUStorageNPUTensor)

    monkeypatch.setitem(sys.modules, "torch_npu", SimpleNamespace(empty_with_format=empty_with_format))
    namespace = {
        "__name__": __name__,
        "torch": CPUFactories(),
        "dataclass": dataclass,
        "Sequence": Sequence,
        "math": math,
        "_NZ_INNER": 16,
        "_COMPRESS_RATIO": 4,
        "_PREFILL_QUERY_TILE": 64,
        "_NZ_VALUE_GATHER_MIN_GROUPS": 256,
        "_GROUPED_MATMUL_SINGLE_OUTPUT": 3,
        "_validate_contract": lambda *args: None,
    }
    _load_functions(OPS_PATH / "qsa_indexer.py", {"QSAGroupSelection"}, namespace)
    _load_functions(
        OPS_PATH / "qsa_gather_nz_310.py",
        {
            "prepare_qsa_gather_metadata",
            "_qsa_gather_nz_310",
            "qsa_gather_key_transposed_nz_310",
            "qsa_gather_value_nz_310",
        },
        namespace,
    )
    _load_functions(
        OPS_PATH / "qsa_batched_attention_310.py", {"_request_slices", "qsa_batched_prefill_310"}, namespace
    )
    return namespace, gather_calls


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("strided", [False, True])
def test_prepared_metadata_preserves_values_and_reuses_storage(monkeypatch, dtype, strided):
    namespace, _ = _production_namespace(monkeypatch)
    groups = torch.tensor([[2, -1], [7, 3]], dtype=dtype)
    vector = torch.tensor([1, 2], dtype=dtype)
    table = torch.tensor([[8, 2, 1]], dtype=dtype)
    if strided:
        groups = groups.repeat_interleave(2, dim=1)[:, ::2]
        vector = vector.repeat_interleave(2)[::2]
        table = table.repeat_interleave(2, dim=1)[:, ::2]
    selection = namespace["QSAGroupSelection"](groups, vector, vector + 4, vector)
    prepared, prepared_table = namespace["prepare_qsa_gather_metadata"](selection, table)
    second, second_table = namespace["prepare_qsa_gather_metadata"](prepared, prepared_table)
    for field in ("group_indices", "group_counts", "tail_starts", "tail_counts"):
        actual = getattr(prepared, field)
        assert actual.dtype == torch.int32 and actual.is_contiguous()
        assert torch.equal(actual, getattr(selection, field))
        assert actual.data_ptr() == getattr(second, field).data_ptr()
    assert torch.equal(prepared_table, table)
    assert prepared_table.data_ptr() == second_table.data_ptr()


@pytest.mark.parametrize("query_tile", [1, 2, 4])
@pytest.mark.parametrize("parallel", [False, True])
@pytest.mark.parametrize(
    "metadata_dtypes",
    [
        (torch.int64, torch.int64, torch.int64),
        (torch.int32, torch.int64, torch.int32),
        (torch.int32, torch.int32, torch.int32),
    ],
)
def test_multi_request_attention_shares_casts_between_k_v_and_tiles(monkeypatch, query_tile, parallel, metadata_dtypes):
    namespace, calls = _production_namespace(monkeypatch)
    group_dtype, geometry_dtype, table_dtype = metadata_dtypes
    events = []

    class Stream:
        def __init__(self, name):
            self.name = name
            self.recorded = []

        def record_event(self):
            event = (self.name, len(events))
            events.append(("record", event))
            return event

        def wait_event(self, event):
            events.append(("wait", self.name, event))

    class StreamSwitch:
        def __init__(self, stream):
            self.stream = stream

        def __enter__(self):
            events.append(("enter", self.stream.name))

        def __exit__(self, *args):
            events.append(("exit", self.stream.name))

    main, key_stream, value_stream = Stream("main"), Stream("key"), Stream("value")
    monkeypatch.setitem(
        sys.modules,
        "vllm_ascend.utils",
        SimpleNamespace(current_stream=lambda: main, npu_stream_switch=StreamSwitch),
    )
    generator = torch.Generator().manual_seed(29)
    query = torch.randn(4, 2, 16, generator=generator).half().as_subclass(CPUStorageNPUTensor)
    key_cache = torch.randn(4, 1, 16, 16, generator=generator).half().as_subclass(CPUStorageNPUTensor)
    value_cache = torch.randn(4, 1, 16, 16, generator=generator).half().as_subclass(CPUStorageNPUTensor)
    # Different page order, selected groups and causal tails for each request.
    table = torch.tensor([[2, 0], [3, 1]], dtype=table_dtype).as_subclass(CPUStorageNPUTensor)
    groups = torch.full((4, 256), -1, dtype=group_dtype)
    groups[:, :2] = torch.tensor([[0, 2], [1, 3], [2, 0], [3, 1]])
    groups = groups.as_subclass(CPUStorageNPUTensor)
    counts = torch.tensor([2, 1, 2, 2], dtype=geometry_dtype).as_subclass(CPUStorageNPUTensor)
    tails = torch.tensor([12, 16, 20, 24], dtype=geometry_dtype).as_subclass(CPUStorageNPUTensor)
    tail_counts = torch.tensor([1, 2, 3, 4], dtype=geometry_dtype).as_subclass(CPUStorageNPUTensor)
    selection = namespace["QSAGroupSelection"](groups, counts, tails, tail_counts)
    with CastCounter() as counter:
        actual = namespace["qsa_batched_prefill_310"](
            query,
            key_cache,
            value_cache,
            selection,
            table,
            torch.tensor([0, 2, 4]),
            scale=0.25,
            query_tile=query_tile,
            query_lens=[2, 2],
            gather_streams=(key_stream, value_stream) if parallel else None,
        )
    # Each non-int32 operand converts once, independent of tile count,
    # request splits and K/V streams. Already prepared metadata adds no copies.
    expected_copies = (group_dtype != torch.int32) + 3 * (geometry_dtype != torch.int32) + (table_dtype != torch.int32)
    assert counter.int32_copies == expected_copies
    key_operands = [tuple(t.data_ptr() for t in call[:5]) for call in calls if call[-1]]
    value_operands = [tuple(t.data_ptr() for t in call[:5]) for call in calls if not call[-1]]
    assert key_operands == value_operands

    # Independent selected-token attention reference, with FP16 score/PV
    # rounding matching the retained attention path.
    expected = []
    raw_keys = key_cache.as_subclass(torch.Tensor).permute(0, 2, 1, 3).reshape(-1, 16)
    raw_values = value_cache.as_subclass(torch.Tensor).permute(0, 2, 1, 3).reshape(-1, 16)
    for row in range(4):
        request = row // 2
        tokens = [int(group) * 4 + offset for group in groups[row, : counts[row]] for offset in range(4)]
        tokens += list(range(int(tails[row]), int(tails[row] + tail_counts[row])))
        slots = [int(table[request, token // 16]) * 16 + token % 16 for token in tokens]
        logits = (query[row].as_subclass(torch.Tensor) * 0.25) @ raw_keys[slots].T
        probabilities = torch.softmax(logits.float(), dim=-1).half()
        expected.append(probabilities @ raw_values[slots])
    torch.testing.assert_close(actual.as_subclass(torch.Tensor), torch.stack(expected), rtol=0, atol=0)
    if parallel:
        # Each producer waits for metadata and previous buffer consumers, and
        # main waits for both gathers. Converted inputs retain stream ownership.
        tiles = math.ceil(4 / query_tile)
        assert sum(event[:2] == ("wait", "main") for event in events) == tiles * 2
        assert sum(event[:2] == ("wait", "key") for event in events) == tiles
        assert sum(event[:2] == ("wait", "value") for event in events) == tiles
        for stream in (key_stream, value_stream):
            metadata = [tensor for tensor in stream.recorded if tensor.dtype == torch.int32]
            assert len(metadata) >= tiles * 5
