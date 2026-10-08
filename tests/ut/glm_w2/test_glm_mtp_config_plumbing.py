# SPDX-License-Identifier: Apache-2.0
"""Execute the configuration patch helpers without importing platform patches."""

import ast
import copy
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace

import torch

from vllm_ascend.models.glm5next_w2.mtp_config import (
    PACKED_GLM_MTP_ARCHITECTURE,
    PACKED_GLM_TARGET_ARCHITECTURES,
    normalize_glm_mtp_hf_config,
)


def helpers():
    path = Path(__file__).resolve().parents[3] / "vllm_ascend/patch/platform/patch_speculative_config.py"
    nodes = [
        node
        for node in ast.parse(path.read_text()).body
        if isinstance(node, ast.FunctionDef)
        and node.name in ("_normalize_legacy_qwen3_dspark_config", "_normalize_packed_glm_draft")
    ]
    scope = dict(
        PACKED_GLM_MTP_ARCHITECTURE=PACKED_GLM_MTP_ARCHITECTURE,
        PACKED_GLM_TARGET_ARCHITECTURES=PACKED_GLM_TARGET_ARCHITECTURES,
        normalize_glm_mtp_hf_config=normalize_glm_mtp_hf_config,
        replace=replace,
        PretrainedConfig=object,
        _orig_hf_config_override=lambda config: config,
        _normalize_kimi_dflash_rope=lambda config: None,
    )
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), scope)
    return scope


@dataclass
class Arch:
    architectures: list
    model_type: str = "glm5_next_mtp"
    text_model_type: str = "glm5_next_mtp"
    is_mm_prefix_lm: bool = False


def test_nested_glm_draft_is_flattened_before_upstream_override():
    text = SimpleNamespace(num_nextn_predict_layers=1, model_type="glm5_next_text")
    config = SimpleNamespace(
        model_type="glm5_next", architectures=["Glm5NextForConditionalGeneration"], text_config=text
    )
    result = helpers()["_normalize_legacy_qwen3_dspark_config"](config)
    assert result.architectures == ["Glm5NextMTPModel"]
    assert result.n_predict == 1 and text.model_type == "glm5_next_text"


def test_target_dict_overrides_reach_draft_and_registry_is_refreshed():
    seen = []
    text = SimpleNamespace(model_type="glm5_next_text", num_nextn_predict_layers=1)
    target = SimpleNamespace(
        architectures=["Glm5NextW2ForCausalLM"],
        hf_text_config=text,
        hf_config=SimpleNamespace(ascend_glm_nz_packed_codes=True),
    )

    def inspect(architectures, model):
        seen.append(architectures)
        assert model.hf_config is model.hf_text_config
        return "packed-model-info", architectures[0]

    draft = SimpleNamespace(
        hf_config=SimpleNamespace(model_type="glm5_next_mtp", num_nextn_predict_layers=1),
        model_arch_config=Arch(["Glm5NextMTPModel"]),
        registry=SimpleNamespace(inspect_model_cls=inspect),
    )
    spec = SimpleNamespace(method="mtp", target_model_config=target, draft_model_config=draft)
    helpers()["_normalize_packed_glm_draft"](spec)
    assert draft.hf_config.ascend_glm_nz_packed_codes
    assert seen == [["Glm5NextW2MTPModel"]]
    assert draft._model_info == "packed-model-info" and draft._architecture == "Glm5NextW2MTPModel"
    assert target.architectures == ["Glm5NextW2ForCausalLM"]


def test_other_draft_families_are_untouched():
    spec = SimpleNamespace(method="mtp", target_model_config=SimpleNamespace(architectures=["Qwen4Exp"]))
    helpers()["_normalize_packed_glm_draft"](spec)
    assert not hasattr(spec, "draft_model_config")


def test_single_draft_does_not_enable_unused_indexer_compaction():
    path = Path(__file__).resolve().parents[3] / "vllm_ascend/spec_decode/llm_base_proposer.py"
    tree = ast.parse(path.read_text())
    assignments = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Attribute) and t.attr == "_share_mtp_indices" for t in node.targets)
    ]
    statement = assignments[-1]
    code = compile(ast.fix_missing_locations(ast.Module(body=[statement], type_ignores=[])), str(path), "exec")
    for count, configured, expected in [(1, True, False), (2, True, True), (2, False, False)]:
        proposer = SimpleNamespace(num_speculative_tokens=count)
        exec(code, {"self": proposer, "draft_hf_config": SimpleNamespace(index_share_for_mtp_iteration=configured)})
        assert proposer._share_mtp_indices is expected


def test_device_draft_slots_follow_request_moves_and_padding(monkeypatch):
    path = Path(__file__).resolve().parents[3] / "vllm_ascend/spec_decode/multi_kv_cache_group_proposer.py"
    tree = ast.parse(path.read_text())
    fn = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "compute_packed_draft_slots"
    )
    namespace = {"torch": torch, "PADDING_SLOT_ID": -1}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[])), str(path), "exec"), namespace)
    compute = namespace[fn.name]
    table = torch.tensor([[5, 8], [2, 3], [-1, -1]], dtype=torch.int32)
    rows = torch.arange(8, dtype=torch.int32)

    def no_range(*args, **kwargs):
        raise AssertionError("draft slot mapping must reuse row indices during capture")

    monkeypatch.setattr(torch, "arange", no_range)
    actual = compute(table, torch.tensor([0, 2, 3, 3]), torch.tensor([3, 4, 6, 0]), 4, rows)
    assert actual.tolist() == [23, 32, 14, -1]
    actual = compute(table.flip(0), torch.tensor([0, 0, 2, 3]), torch.tensor([1, 2, 7, -1]), 4, rows)
    assert actual.tolist() == [9, 10, 35, -1]
    actual = compute(table, torch.tensor([0, 4, 4, 4]), torch.tensor([-1, 8, 0, 7]), 4, rows)
    assert actual.tolist() == [-1, -1, 20, 35]
    actual = compute(table, torch.tensor([0, 0, 0, 0]), torch.tensor([0, 1]), 4, rows)
    assert actual.tolist() == [-1, -1]


def test_graph_secondary_metadata_reuses_buffer_without_cpu_mapper():
    path = Path(__file__).resolve().parents[3] / "vllm_ascend/spec_decode/multi_kv_cache_group_proposer.py"
    tree = ast.parse(path.read_text())
    helper = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "compute_packed_draft_slots"
    )
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef))
    method = next(
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name == "_common_attn_metadata_for_draft_group"
    )
    namespace = {"torch": torch, "copy": copy, "PADDING_SLOT_ID": -1}
    exec(compile(ast.Module(body=[helper, method], type_ignores=[]), str(path), "exec"), namespace)
    table = torch.tensor([[5, 8], [2, 3]], dtype=torch.int32)
    # Deliberately omit compute_slot_mapping: 310P's CPU mapper must not be called.
    block_table = SimpleNamespace(get_device_tensor=lambda: table, block_size=4)
    slots = torch.full((8,), 123, dtype=torch.int32)
    proposer = SimpleNamespace(
        _uses_multi_group_kv_cache=True,
        _glm_draft_graph_supported=True,
        kv_cache_gid=0,
        arange=torch.arange(8, dtype=torch.int32),
        runner=SimpleNamespace(input_batch=SimpleNamespace(block_table={1: block_table})),
        _multi_group_slot_mapping_buffers={(1, 0): slots},
        _draft_block_table_width=lambda group: 2,
    )
    common = SimpleNamespace(
        num_reqs=2,
        num_actual_tokens=3,
        query_start_loc=torch.tensor([0, 2, 3], dtype=torch.int32),
        positions=torch.tensor([3, 4, 6]),
        slot_mapping=torch.full((8,), 77),
    )
    invoke = namespace[method.name]
    result = invoke(proposer, common, SimpleNamespace(kv_cache_group_id=1), 8)
    assert result.slot_mapping.data_ptr() == slots.data_ptr()
    assert slots.tolist() == [23, 32, 14, -1, -1, -1, -1, -1]
    assert common.slot_mapping.tolist() == [77] * 8
    common.num_actual_tokens = 2
    common.query_start_loc.copy_(torch.tensor([0, 1, 2]))
    common.positions[:2].copy_(torch.tensor([2, 5]))
    table.copy_(torch.tensor([[2, 3], [5, 8]]))
    replay = invoke(proposer, common, SimpleNamespace(kv_cache_group_id=1), 8)
    assert replay.slot_mapping.data_ptr() == result.slot_mapping.data_ptr()
    assert slots.tolist() == [10, 33, -1, -1, -1, -1, -1, -1]
