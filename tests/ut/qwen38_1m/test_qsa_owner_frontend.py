# SPDX-License-Identifier: Apache-2.0
"""Check that the experimental schedule preserves current QSA formulas."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.qwen4exp.resident_candidates import qsa_owner_prefill
from tools.qwen4exp.resident_candidates.qsa_owner_prefill import assigned_names, is_cache_write, transform_frontend


def frontend_source():
    root = ast.parse(Path("vllm_ascend/models/qwen4_exp/model.py").read_text())
    layer = next(node for node in root.body if isinstance(node, ast.ClassDef) and node.name == "_QSAAttention")
    return ast.unparse(
        next(node for node in layer.body if isinstance(node, ast.FunctionDef) and node.name == "forward")
    )


def test_moves_only_local_projections_normalization_and_kv_write():
    source = frontend_source()
    before = ast.parse(source).body[0]
    after = transform_frontend(source).body[0]
    compile(ast.Module(body=[after], type_ignores=[]), "<owner-test>", "exec")
    local = next(
        node for node in after.body if isinstance(node, ast.FunctionDef) and node.name == "prepare_owner_local"
    )
    old_statements = [
        node
        for node in before.body
        if (assigned_names(node) and assigned_names(node) <= {"q", "k", "v", "gate"}) or is_cache_write(node)
    ]
    assert len(old_statements) == 10
    assert [ast.dump(node) for node in local.body[:-1]] == [ast.dump(node) for node in old_statements]
    update = next(
        i
        for i, node in enumerate(after.body)
        if isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name)
        and node.value.func.id == "qsa_index_cache_update_310"
    )
    assert update < after.body.index(local)
    select = next(
        node for node in ast.walk(after) if isinstance(node, ast.FunctionDef) and node.name == "select_owner_groups"
    )
    assert assigned_names(select.body[-2]) == {"selection"}
    assert isinstance(select.body[-1], ast.Return)
    # One eager selector remains; the graph selector stays in the original
    # function called by the wrapper, never in this pure-prefill schedule.
    calls = [node for node in ast.walk(after) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)]
    assert sum(node.func.id == "qsa_indexer_select_groups_310" for node in calls) == 1


def test_refuses_changed_projection_schedule():
    tree = ast.parse(frontend_source())
    tree.body[0].body = [node for node in tree.body[0].body if assigned_names(node) != {"gate"}]
    with pytest.raises(ValueError, match="local projection/cache schedule"):
        transform_frontend(ast.unparse(tree))


def test_refuses_changed_capture_selection_branch():
    source = frontend_source().replace("capture._capturing", "capture.new_capture_field")
    with pytest.raises(ValueError, match="eager selection schedule"):
        transform_frontend(source)


def test_refuses_non_function_source():
    with pytest.raises(ValueError, match="ordinary function"):
        transform_frontend("value = 3")


def test_owner_rotation_covers_six_ranks_with_every_fourth_layer(monkeypatch):
    class Schedule:
        def __init__(self, policy, *args, **kwargs):
            self.policy = policy

    monkeypatch.setattr(qsa_owner_prefill, "QSAOwnerOverlap", Schedule)
    monkeypatch.setattr(qsa_owner_prefill.torch, "npu", None, raising=False)
    communicator = SimpleNamespace(available=True, disabled=False, rank=0, world_size=6)
    state = qsa_owner_prefill.OwnerPrefillState(communicator)
    config = SimpleNamespace(full_attention_interval=4)
    layers = [SimpleNamespace(indexer=SimpleNamespace(config=config, layer_idx=i)) for i in range(0, 48, 4)]
    owners = [state.for_layer(layer).policy.owner for layer in layers]
    # layer_idx % 6 would omit every odd owner for this model.
    assert owners == list(range(6)) * 2
    assert state.for_layer(layers[0]) is state.for_layer(layers[6])
    config.full_attention_interval = 0
    with pytest.raises(ValueError, match="positive full-attention interval"):
        state.for_layer(layers[0])


def test_transport_uses_global_owner_and_waits_on_the_submitting_stream(monkeypatch):
    submitted = []
    current = SimpleNamespace(npu_stream=23)
    monkeypatch.setattr(qsa_owner_prefill.torch, "npu", SimpleNamespace(current_stream=lambda: current), raising=False)
    group = SimpleNamespace(rank_in_group=0, world_size=2, ranks=[2, 5], device_group=object())

    def broadcast(packet, **kwargs):
        submitted.append(kwargs)
        return SimpleNamespace(wait=lambda: submitted.append("wait"))

    monkeypatch.setattr(qsa_owner_prefill.torch.distributed, "broadcast", broadcast)
    transport = qsa_owner_prefill.ExistingHcclBroadcast(group)
    transport.broadcast(object(), 1, current)
    assert submitted == [{"src": 5, "group": group.device_group, "async_op": True}, "wait"]
    with pytest.raises(RuntimeError, match="selection stream"):
        transport.broadcast(object(), 0, SimpleNamespace(npu_stream=24))
    assert len(submitted) == 2
