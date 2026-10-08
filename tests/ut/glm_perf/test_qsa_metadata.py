# SPDX-License-Identifier: Apache-2.0
"""Metadata semantics and reversible attention integration without NPU imports."""

from types import SimpleNamespace

import pytest
import torch

from tools.glm_perf.instance_bindings import InstanceBindings
from tools.glm_perf.qsa_metadata import Geometry, NativeQsaMetadata, backing_span, geometry
from tools.glm_perf.qsa_metadata_binding import bind_attention
from tools.glm_perf.qsa_metadata_probe import reference


def test_signed_groups_dense_boundary_and_partial_tail():
    ids = torch.tensor([[-1, 0, 0, 0, -5, 0, 0, 0]] * 6, dtype=torch.int32)
    positions = torch.tensor([-1, 0, 7, 8, 9, 11], dtype=torch.int64)
    table = torch.tensor([[-1, 4, -21, 8, 40]], dtype=torch.int32)
    plan, logical = reference(ids, positions, table, 6, 0, 2, 20)
    assert plan[0].tolist() == [[-1, -2]] * 6
    assert plan[1].tolist() == [0, 1, 8, 2, 2, 2]
    assert plan[2].tolist() == [0, 0, 8, 8, 8, 12]
    assert plan[3].tolist() == [-1, -1, -1, 1, 2, 0]
    assert logical.tolist() == [[-1]]


def test_span_preserves_offset_and_gapped_rows_without_copy():
    backing = torch.arange(200, dtype=torch.int32)
    view = backing.as_strided((3, 7), (20, 2), 8)
    flat = backing_span(view)
    assert flat.numel() == 53 and flat.storage_offset() == 8
    assert flat.data_ptr() == view.data_ptr()
    flat[20] = -1
    assert view[1, 0] == -1 and backing[28] == -1


def test_unprepared_mixed_plan_is_unavailable_without_device_work(monkeypatch):
    native = NativeQsaMetadata.__new__(NativeQsaMetadata)
    native.device, native.configs = torch.device("cpu"), {}
    ids = torch.empty((8, 16), dtype=torch.int32)
    positions = torch.empty(2, dtype=torch.int64)
    table = torch.empty((1, 41), dtype=torch.int32)
    options = dict(rows=2, token_start=0, budget=4, block_size=640)
    case = geometry(ids, positions, table, **options)
    monkeypatch.setattr(torch, "tensor", lambda *args, **kwargs: pytest.fail("availability allocated a descriptor"))
    assert not native.available(ids, positions, table, **options)
    native.configs[case] = object()
    assert native.available(ids, positions, table, **options)
    assert not native.available(ids, positions, table, **dict(options, token_start=2))
    assert not native.available(ids.float(), positions, table, **options)


@pytest.mark.parametrize(
    "rows,start,budget,block_size", [(0, 0, 4, 640), (2, 2, 4, 640), (2, 0, 8, 640), (2, 0, 4, 160)]
)
def test_invalid_layout_is_rejected_before_launch(rows, start, budget, block_size):
    with pytest.raises(ValueError):
        geometry(
            torch.empty((3, 16), dtype=torch.int32),
            torch.empty(2, dtype=torch.int64),
            torch.empty((1, 41), dtype=torch.int32),
            rows,
            start,
            budget,
            block_size,
        )


@pytest.mark.parametrize(
    "field,value",
    [("split", 4), ("position_bytes", 2), ("ids_row_stride", 2), ("table_row_stride", 2), ("token_start", -1)],
)
def test_descriptor_rejects_unqualified_geometry(field, value):
    values = dict(
        rows=2,
        requests=4,
        budget=4,
        token_start=0,
        ids_row_stride=16,
        ids_column_stride=1,
        position_stride=1,
        position_bytes=8,
        table_width=41,
        table_row_stride=41,
        table_column_stride=1,
        split=20,
    )
    values[field] = value
    with pytest.raises(ValueError):
        Geometry(**values)


def _qsa_cache_block_table(table, block_size):
    return ("original table", table)


class Attention:
    host_kv_layer = None

    def _get_kpool_qsa_plan(self, positions, start, rows):
        return ("original plan", start, rows)

    def _forward_decode_fused(self, query, key, value, metadata):
        plan = self._get_kpool_qsa_plan(metadata.decode.input_positions, 0, query.shape[0])
        table = _qsa_cache_block_table(metadata.decode.block_table, key.shape[2])
        if getattr(self, "fail", False):
            raise RuntimeError("attention failed")
        return plan, table

    def _forward_prefill_paged_latent(self, query, cache, metadata):
        plan = self._get_kpool_qsa_plan(
            metadata.prefill.input_positions, metadata.num_decode_tokens, metadata.prefill.actual_seq_lengths_q[-1]
        )
        return plan, _qsa_cache_block_table(metadata.prefill.block_table, cache[0].shape[2])


@pytest.mark.parametrize("failure", [False, True])
def test_attention_binding_is_scoped_to_call_and_restores_target_and_draft(failure):
    owners = [Attention(), Attention()]
    for owner in owners:
        owner.glm_indexer = SimpleNamespace(topk_indices_buffer=object(), topk_tokens=16, index_kpool=4)
    roots = [SimpleNamespace(modules=lambda owner=owner: [SimpleNamespace(impl=owner)]) for owner in owners]
    runner = SimpleNamespace(model=roots[0], drafter=SimpleNamespace(model=roots[1]))
    calls = []

    def plan(ids, positions, table, **kwargs):
        calls.append(kwargs)
        return ("native plan", kwargs["token_start"], kwargs["rows"]), ("native table", table)

    bindings = InstanceBindings()
    native = SimpleNamespace(plan=plan, available=lambda *args, **kwargs: True, fallbacks=0)
    assert bind_attention(bindings, runner, native) == 2
    query = torch.empty((2, 1))
    key = torch.empty((1, 1, 640))
    meta = SimpleNamespace(input_positions=object(), block_table=torch.empty((1, 41)), actual_seq_lengths_q=[2])
    metadata = SimpleNamespace(decode=meta, prefill=meta, num_decodes=1, num_decode_tokens=3)
    owner = owners[0]
    owner.fail = failure
    if failure:
        with pytest.raises(RuntimeError, match="attention failed"):
            owner._forward_decode_fused(query, key, key, metadata)
    else:
        assert owner._forward_decode_fused(query, key, key, metadata)[0] == ("native plan", 0, 2)
    assert owner._get_kpool_qsa_plan(None, 7, 8) == ("original plan", 7, 8)
    assert owners[1]._forward_prefill_paged_latent(query, (key, key), metadata)[0] == ("native plan", 3, 2)
    assert calls[-1]["token_start"] == 3 and calls[-1]["budget"] == 4
    native.available = lambda *args, **kwargs: False
    assert owners[1]._forward_prefill_paged_latent(query, (key, key), metadata)[0] == ("original plan", 3, 2)
    assert native.fallbacks == 1
    owners[1].host_kv_layer = object()
    assert owners[1]._forward_decode_fused(query, key, key, metadata)[0] == ("original plan", 0, 2)
    bindings.restore()
    assert all("_forward_decode_fused" not in owner.__dict__ for owner in owners)


def test_attention_binding_rejects_missing_glm_instance():
    runner = SimpleNamespace(model=SimpleNamespace(modules=lambda: []))
    with pytest.raises(ValueError, match="no loaded GLM"):
        bind_attention(InstanceBindings(), runner, None)
