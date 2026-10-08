# SPDX-License-Identifier: Apache-2.0
"""Combined target/draft preparation and reversible attention capture hooks."""

from types import SimpleNamespace

import pytest
import torch

from tests.ut.glm_perf.test_qsa_metadata import Attention
from tools.glm_perf.qsa_metadata import Geometry, geometry
from tools.glm_perf.resident_candidates.qsa_metadata import PreparedMetadata, extend_replacements, prepare_runner
from tools.glm_perf.resident_rpc_guard import WORKER_PREFIX


def fixture_runner():
    owners = [Attention(), Attention()]
    for owner in owners:
        owner.glm_indexer = SimpleNamespace(
            index_kpool=4, topk_tokens=16, topk_indices_buffer=torch.empty((2, 16), dtype=torch.int32)
        )
    modules = [SimpleNamespace(impl=owner) for owner in owners]
    roots = [SimpleNamespace(modules=lambda module=module: [module]) for module in modules]
    table = SimpleNamespace(
        is_mamba_group=False, block_size=32, block_table=SimpleNamespace(gpu=torch.empty((2, 41), dtype=torch.int32))
    )
    return SimpleNamespace(
        model=roots[0],
        drafter=SimpleNamespace(model=roots[1]),
        input_batch=SimpleNamespace(block_table=SimpleNamespace(block_tables=[table])),
    ), owners


def native_fixture(events):
    native = SimpleNamespace(configs={}, calls=0, fallbacks=0, geometries={})

    def prepare(cases):
        events.append("prepare")
        native.configs.update((case, object()) for case in cases)

    native.prepare = prepare
    return native


def test_target_and_draft_shapes_are_prepared_with_the_frozen_geometry_type():
    runner, _ = fixture_runner()
    native = native_fixture([])
    created = []

    def frozen_geometry(*args):
        created.append(args)
        return Geometry(*args)

    assert prepare_runner(native, runner, frozen_geometry) == [dict(shape=[2, 41], stride=[41, 1])]
    assert len(created) == 16 and len(native.configs) == 8
    assert {g.rows for g in native.configs} == {1, 2}
    assert {g.requests for g in native.configs} == {1, 2}
    assert {g.position_bytes for g in native.configs} == {4, 8}
    assert {g.token_start for g in native.configs} == {0}
    assert {g.split for g in native.configs} == {20}


def test_earlier_frozen_helper_is_adapted_without_mutating_its_api(monkeypatch):
    ids = torch.empty((8, 16), dtype=torch.int32)
    positions = torch.empty(2, dtype=torch.int64)
    table = torch.empty((1, 41), dtype=torch.int32)
    options = dict(rows=2, token_start=0, budget=4, block_size=640)
    case = geometry(ids, positions, table, **options)
    old = SimpleNamespace(device=torch.device("cpu"), configs={case: object()}, calls=0, geometries={})
    prepared = PreparedMetadata(old, geometry)
    monkeypatch.setattr(torch, "tensor", lambda *args, **kwargs: pytest.fail("availability allocated metadata"))
    assert prepared.available(ids, positions, table, **options)
    assert not prepared.available(ids, positions, table, **dict(options, token_start=2))
    assert not prepared.available(ids.float(), positions, table, **options)
    assert prepared.fallbacks == 0 and not hasattr(old, "fallbacks")


@pytest.mark.parametrize("cause", ["table", "attention", "pool"])
def test_unqualified_runner_is_rejected_before_preparation(cause):
    runner, owners = fixture_runner()
    if cause == "table":
        runner.input_batch.block_table.block_tables = []
    elif cause == "attention":
        owners[0].glm_indexer = owners[1].glm_indexer = None
    else:
        owners[0].glm_indexer.index_kpool = 8
    native = native_fixture([])
    with pytest.raises(ValueError):
        prepare_runner(native, runner, Geometry)
    assert not native.configs


@pytest.mark.parametrize("failure", [None, "receipt", "exception"])
def test_capture_preserves_parent_policies_and_restores_every_binding(failure):
    runner, owners = fixture_runner()
    events = []
    native = native_fixture(events)
    session = SimpleNamespace(graphs_dirty=False)
    worker = SimpleNamespace(
        model_runner=runner, _resident_session=lambda: session, _resident_error=lambda error: {"error": str(error)}
    )

    def capture(self):
        events.append("parent_capture")
        assert all("_get_kpool_qsa_plan" in owner.__dict__ for owner in owners)
        if failure == "exception":
            raise RuntimeError("parent failed")
        return {"error": "parent failed"} if failure else {"captured": True}

    def apply(self, generation):
        events.append("parent_apply")
        assert all("_get_kpool_qsa_plan" not in owner.__dict__ for owner in owners)
        return {"generation": generation}

    marker = object()
    changes = {
        WORKER_PREFIX + "resident_capture": capture,
        WORKER_PREFIX + "resident_apply": apply,
        WORKER_PREFIX + "resident_status": lambda self: {"parent": True},
        "other_policy": marker,
    }
    result = extend_replacements(changes, native, Geometry)
    assert result["other_policy"] is marker
    receipt = result[WORKER_PREFIX + "resident_capture"](worker)
    assert events == ["prepare", "parent_capture"]
    assert ("error" in receipt) == bool(failure)
    status = result[WORKER_PREFIX + "resident_status"](worker)
    assert status["parent"] and status["fused_qsa_metadata"]["attention_instances"] == 2
    if failure:
        assert status["fused_qsa_metadata"]["instance_methods"] == 0
    if failure == "exception":
        assert session.graphs_dirty
    assert result[WORKER_PREFIX + "resident_apply"](worker, "next") == {"generation": "next"}
    assert all(
        not owner.__dict__.keys() & {"_get_kpool_qsa_plan", "_forward_decode_fused", "_forward_prefill_paged_latent"}
        for owner in owners
    )
