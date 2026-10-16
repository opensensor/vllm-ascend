# SPDX-License-Identifier: Apache-2.0
"""Portable full-data gate/control tests; no accelerator qualification claims."""

import copy
import importlib
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tools.sidecard import execution_model_gate as gate
from tools.sidecard.stage_diagnostics import (
    DiagnosticWriter,
    capture_boundary,
    capture_chunk,
    capture_layers,
    capture_selection,
    capture_vocabulary,
    private_json,
    sha256_file,
)


@pytest.fixture
def precision():
    return {
        "fixed_before_native_device_tests": True,
        "model_bounds": {
            "same_math": {"all_tensors": "bitwise"},
            "changed_math": {
                "floating_state": {"atol": 0.001, "rtol": 0.001},
                "vocabulary": {"atol": 0.02, "rtol": 0.002},
                "selectors": "bitwise",
                "integer_boolean_state": "bitwise",
                "nonfinite": "reject",
                "unknown_tensor_kind": "reject",
            },
        },
    }


def small_plan(chunk=2, architecture=gate.BLOCK_ARCHITECTURE):
    return gate.make_plan(
        "case",
        [1, 2, 3, 4, 5],
        {"layer_types": ["full_attention"] * 19, "hidden_size": 2, "index_topk": 4, "kv_lora_rank": 2, "vocab_size": 4},
        {"architecture": architecture, "max_context": 16, "prefill_chunk_size": chunk},
        output_tokens=3,
    )


def synthesize(root, plan):
    captures = {}
    for owner in gate.OWNERS:
        writer = DiagnosticWriter(root / owner, owner, max_chunk_bytes=32)
        for key, declared in plan["expected"][owner].items():
            value = (
                np.asarray(declared["value"], dtype=declared["dtype"])
                if "value" in declared
                else np.zeros(declared["shape"], dtype=declared["dtype"])
            )
            writer.append(key, value, declared["kind"])
        captures[owner] = writer.close(allocation_geometry=plan["allocation_geometry"][owner])
    return captures


def mutate_manifest(path, transform):
    body = json.loads(path.read_bytes())
    transform(body)
    private_json(path, body)


def test_streaming_capture_is_private_bounded_and_complete(tmp_path):
    writer = DiagnosticWriter(tmp_path / "owner", "cuda", max_chunk_bytes=8)
    writer.append("state", np.array([[1, 2], [3, 4]], dtype="float32"), "floating_state")
    writer.append("state", np.array([[5, 6]], dtype="float32"), "floating_state", offset=2)
    path = writer.close()
    body, rows = gate.validate_capture(path, {"state": dict(shape=[3, 2], dtype="float32", kind="floating_state")}, {})
    assert body["max_chunk_bytes"] == 8
    assert rows["state"]["segments"] == [{"offset": 0, "rows": 2}, {"offset": 2, "rows": 1}]
    assert os.stat(path).st_mode & 0o777 == 0o600
    assert os.stat(path.parent).st_mode & 0o777 == 0o700
    assert np.fromfile(path.parent / rows["state"]["path"], dtype="float32").tolist() == [1, 2, 3, 4, 5, 6]


@pytest.mark.parametrize("offset", [1, -1, True])
def test_writer_rejects_missing_initial_chunk(tmp_path, offset):
    writer = DiagnosticWriter(tmp_path / "owner", "cuda")
    with pytest.raises(ValueError):
        writer.append("state", np.zeros((2, 2), dtype="float32"), "floating_state", offset=offset)


def test_writer_rejects_overwrite_and_dtype_change(tmp_path):
    writer = DiagnosticWriter(tmp_path / "owner", "cuda")
    writer.append("state", np.zeros((2, 2), dtype="float32"), "floating_state")
    with pytest.raises(ValueError, match="overlap"):
        writer.append("state", np.zeros((1, 2), dtype="float32"), "floating_state", offset=1)
    with pytest.raises(ValueError, match="geometry"):
        writer.append("state", np.zeros((1, 2), dtype="float16"), "floating_state", offset=2)
    with pytest.raises(ValueError, match="contiguous"):
        writer.append("strided", np.zeros((3, 4), dtype="float32")[:, ::2], "floating_state")


def test_capture_boundaries_requires_every_deferred_field(tmp_path):
    writer = DiagnosticWriter(tmp_path / "owner", "cuda")
    with pytest.raises(ValueError, match="complete deferred"):
        capture_boundary(
            writer, "prefill/boundary/prefix", [np.zeros((1, 2), dtype="float16"), None, None, None], offset=0
        )


def test_capture_actual_layers_preserves_storage_and_active_tail(tmp_path):
    indexer = SimpleNamespace(
        pool_count=1,
        tail_length=1,
        pools=np.arange(128 * 4, dtype="float16").reshape(4, 128),
        tail_keys=np.ones((3, 128), dtype="float32"),
        tail_gates=np.full((3, 128), 2, dtype="float32"),
    )
    layer = SimpleNamespace(
        index=18, length=5, linear=False, cache=np.arange(32, dtype="float16").reshape(16, 2), indexer=indexer
    )
    original = layer.cache
    writer = DiagnosticWriter(tmp_path / "owner", "npu-rank0", max_chunk_bytes=32)
    geometry = capture_layers(writer, "prefill/state", [layer])
    assert layer.cache is original
    assert geometry["18"]["kv"] == [16, 2]
    path = writer.close(allocation_geometry=geometry)
    body = json.loads(path.read_bytes())
    records = {row["key"]: row for row in body["tensors"]}
    assert records["prefill/state/layer18/tail_keys"]["shape"] == [1, 128]
    assert records["prefill/state/layer18/kv"]["shape"] == [5, 2]
    indexer.tail_length = 0
    with pytest.raises(ValueError, match="pool/tail"):
        capture_layers(DiagnosticWriter(tmp_path / "bad", "npu-rank0"), "prefill/state", [layer])


def test_linear_cache_uses_existing_zero_length_field(tmp_path):
    layer = SimpleNamespace(
        index=0,
        length=0,
        linear=True,
        state=np.ones((2, 3, 3), dtype="float32"),
        conv=np.ones((18, 3), dtype="float16"),
    )
    writer = DiagnosticWriter(tmp_path / "owner", "cuda")
    capture_layers(writer, "prefill/state", {0: layer})
    assert writer.records["prefill/state/layer0/recurrent"]["shape"] == [2, 3, 3]
    model = {"layer_types": ["linear_attention"], "hidden_size": 2, "index_topk": 4, "kv_lora_rank": 2, "vocab_size": 4}
    plan = gate.make_plan("case", [1], model, {"max_context": 16}, output_tokens=3)
    assert plan["expected"]["cuda"]["prefill/state/layer0/length"]["value"] == [0]


def test_selection_padding_preserves_actual_indices_and_dense_sentinel(tmp_path):
    writer = DiagnosticWriter(tmp_path / "owner", "cuda", max_chunk_bytes=64)
    capture_selection(writer, "select", np.array([[2, 0], [-1, 1]], dtype="int64"), offset=0, rows=2, width=4)
    capture_selection(writer, "select", None, offset=2, rows=1, width=4)
    path = writer.close()
    row = json.loads(path.read_bytes())["tensors"][0]
    assert np.fromfile(path.parent / row["path"], dtype="int64").reshape(3, 4).tolist() == [
        [2, 0, -1, -1],
        [-1, 1, -1, -1],
        [-1, -1, -1, -1],
    ]


def test_complete_gate_compares_all_data_across_changed_chunk_geometry(tmp_path, precision):
    left_plan, right_plan = small_plan(2), small_plan(3)
    left, right = synthesize(tmp_path / "left", left_plan), synthesize(tmp_path / "right", right_plan)
    result = gate.compare_captures(right, left, right_plan, left_plan, "same_math", precision)
    assert result["passed"]
    assert any(row["key"] == "decode/state/layer18/tail_keys" for row in result["tensors"])
    assert any(row["key"] == "chunk_vocabulary@4" for row in result["tensors"])


@pytest.mark.parametrize(
    "change",
    [
        "missing_tensor",
        "missing_tail",
        "bad_shape",
        "bad_dtype",
        "missing_segment",
        "bad_path",
        "unknown_kind",
        "owner",
    ],
)
def test_gate_rejects_incomplete_or_changed_capture(tmp_path, precision, change):
    plan = small_plan()
    left, right = synthesize(tmp_path / "left", plan), synthesize(tmp_path / "right", plan)

    def mutate(body):
        if change in ("missing_tensor", "missing_tail"):
            body["tensors"] = [row for row in body["tensors"] if row["key"] != "decode/state/layer18/tail_keys"]
        elif change == "bad_shape":
            body["tensors"][0]["shape"][0] += 1
        elif change == "bad_dtype":
            body["tensors"][0]["dtype"] = "float32"
        elif change == "missing_segment":
            body["tensors"][0]["segments"] = []
        elif change == "bad_path":
            body["tensors"][0]["path"] = "../escaped.bin"
        elif change == "unknown_kind":
            body["tensors"][0]["kind"] = "unobserved"
        else:
            body["owner"] = "npu-rank1"

    mutate_manifest(right["npu-rank0"], mutate)
    with pytest.raises(ValueError):
        gate.compare_captures(right, left, plan, plan, "same_math", precision)


def test_gate_rejects_missing_rank_and_changed_prompt(tmp_path, precision):
    plan = small_plan()
    left = synthesize(tmp_path / "left", plan)
    missing = {key: value for key, value in left.items() if key != "npu-rank1"}
    with pytest.raises(ValueError, match="rank owners"):
        gate.compare_captures(missing, left, plan, plan, "same_math", precision)
    different = copy.deepcopy(plan)
    different["prompt_ids_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="prompt"):
        gate.compare_captures(left, left, different, plan, "same_math", precision)


@pytest.mark.parametrize(
    "kind,dtype,left,right,accepted",
    [
        ("floating_state", "float32", [1], [1.0005], True),
        ("floating_state", "float32", [1], [1.01], False),
        ("vocabulary", "float32", [1], [1.01], True),
        ("selectors", "int64", [1], [2], False),
        ("integer_boolean_state", "int64", [1], [2], False),
    ],
)
def test_predeclared_bounds_apply_to_every_element(tmp_path, precision, kind, dtype, left, right, accepted):
    np.asarray(left, dtype=dtype).tofile(tmp_path / "a")
    np.asarray(right, dtype=dtype).tofile(tmp_path / "b")
    result = gate.compare_tensor(tmp_path / "a", tmp_path / "b", dtype, kind, "changed_math", precision)
    assert result["passed"] is accepted
    assert result["elements"] == len(left)


@pytest.mark.parametrize("mode", ["same_math", "changed_math"])
def test_nonfinite_rejected_even_when_bytes_match(tmp_path, precision, mode):
    np.array([np.nan], dtype="float32").tofile(tmp_path / "a")
    with pytest.raises(ValueError, match="nonfinite"):
        gate.compare_tensor(tmp_path / "a", tmp_path / "a", "float32", "floating_state", mode, precision)


def test_same_math_requires_signed_zero_and_preserves_bf16_range(tmp_path, precision):
    np.array([0.0, 131072.0], dtype="float32").tofile(tmp_path / "a")
    np.array([-0.0, 131072.0], dtype="float32").tofile(tmp_path / "b")
    result = gate.compare_tensor(tmp_path / "a", tmp_path / "b", "float32", "floating_state", "same_math", precision)
    assert not result["passed"]
    assert result["max_abs_error"] == 0


@pytest.mark.parametrize("mutation", ["unfixed", "missing_bound", "nan_bound", "relaxed_selectors", "unknown"])
def test_precision_contract_cannot_be_implicit(tmp_path, precision, mutation):
    if mutation == "unfixed":
        precision["fixed_before_native_device_tests"] = False
    elif mutation == "missing_bound":
        precision["model_bounds"]["changed_math"]["floating_state"].pop("rtol")
    elif mutation == "nan_bound":
        precision["model_bounds"]["changed_math"]["vocabulary"]["atol"] = float("nan")
    elif mutation == "relaxed_selectors":
        precision["model_bounds"]["changed_math"]["selectors"] = {"atol": 1}
    else:
        precision["model_bounds"]["changed_math"]["unknown_tensor_kind"] = "ignore"
    with pytest.raises((ValueError, KeyError)):
        gate.validate_precision(precision)


def test_graph_replay_and_execution_identity_are_required():
    plan = small_plan()
    good = dict(
        schema_version=1,
        owners=list(gate.OWNERS),
        complete_state=True,
        complete_selectors=True,
        graph_replay_observed=True,
        execution_path_sha256=plan["execution_path_sha256"],
    )
    gate.validate_capabilities(SimpleNamespace(capabilities=lambda: good), plan)
    for field, value in (
        ("complete_selectors", False),
        ("graph_replay_observed", False),
        ("execution_path_sha256", "0" * 64),
    ):
        bad = {**good, field: value}
        with pytest.raises(ValueError):
            gate.validate_capabilities(SimpleNamespace(capabilities=lambda bad=bad: bad), plan)


def fixture_manifest():
    rows = []
    for name, length in (("2K", 2048), ("8K", 8192), ("full-code", 23820)):
        request = (
            {"prompt": [1, 2]}
            if name != "full-code"
            else {
                "messages": [{"role": "user", "content": "private code"}],
                "tools": [{"type": "function", "function": {"name": "read_file"}}],
            }
        )
        rows.append(
            dict(
                name=name,
                request=request,
                request_sha256=gate.canonical_sha256(request),
                expected_prompt_tokens=length,
                expected_tool_count=len(request.get("tools", [])),
            )
        )
    return dict(schema_version=1, fixtures=rows)


def test_original_code_and_tool_identity_cannot_change():
    manifest = fixture_manifest()
    gate.validate_fixtures(manifest)
    manifest["fixtures"][2]["request"]["tools"].clear()
    with pytest.raises(ValueError, match="changed"):
        gate.validate_fixtures(manifest)


def test_config_authenticates_all_categories_and_runtime_options(tmp_path):
    spec = {
        "max_context": 131072,
        "architecture": gate.BLOCK_ARCHITECTURE,
        "execution_diagnostics": {"enabled": True, "authentication": {"path": "auth.json", "sha256": "pending"}},
    }
    files = []
    for category in ("source", "native", "checkpoint"):
        path = tmp_path / category
        path.write_text(category)
        files.append(dict(path=category, sha256=sha256_file(path), category=category))
    receipt = dict(
        identity={key: "a" * 64 for key in ("source_sha256", "checkpoint_sha256", "native_sha256")},
        architecture=gate.BLOCK_ARCHITECTURE,
        config_sha256=gate.config_identity(spec),
        files=files,
    )
    private_json(tmp_path / "auth.json", receipt)
    spec["execution_diagnostics"]["authentication"]["sha256"] = sha256_file(tmp_path / "auth.json")
    gate.authenticate_config(spec, tmp_path)
    spec["npu_decode_graphs"] = True
    with pytest.raises(ValueError, match="configuration"):
        gate.authenticate_config(spec, tmp_path)
    spec.pop("npu_decode_graphs")
    (tmp_path / "native").write_text("changed")
    with pytest.raises(ValueError, match="input changed"):
        gate.authenticate_config(spec, tmp_path)


def test_expert_architecture_has_distinct_state_and_rank_coverage():
    plan = small_plan(architecture=gate.EXPERT_ARCHITECTURE)
    assert "prefill/expert/layer18/input" in plan["expected"]["npu-rank0"]
    assert "prefill/state/layer18/kv" in plan["expected"]["cuda"]
    assert "prefill/state/layer18/kv" not in plan["expected"]["npu-rank0"]


@pytest.mark.parametrize("change", ["unqualified", "contract", "architecture", "checkpoint", "missing_code"])
def test_reference_admission_rejects_unqualified_or_mismatched_receipts(tmp_path, change):
    authentication = dict(architecture=gate.BLOCK_ARCHITECTURE, identity={"checkpoint_sha256": "a" * 64})
    receipt = dict(
        passed=True,
        reference_qualified=True,
        precision_contract_sha256="b" * 64,
        **copy.deepcopy(authentication),
        workloads=[{"name": name} for name in sorted(gate.REQUIRED_FIXTURES)],
    )
    if change == "unqualified":
        receipt["reference_qualified"] = False
    elif change == "contract":
        receipt["precision_contract_sha256"] = "c" * 64
    elif change == "architecture":
        receipt["architecture"] = gate.EXPERT_ARCHITECTURE
    elif change == "checkpoint":
        receipt["identity"]["checkpoint_sha256"] = "c" * 64
    else:
        receipt["workloads"].pop()
    path = tmp_path / "reference.json"
    private_json(path, receipt)
    with pytest.raises(ValueError):
        gate.load_reference(path, sha256_file(path), authentication, "b" * 64)


def test_reference_startup_qualification_binds_exact_default_off_config(tmp_path):
    spec = {"execution_diagnostics": {"authentication": {"path": "auth", "sha256": "pending"}}}
    receipt = dict(
        passed=True,
        default_off=True,
        identity={"source_sha256": "a" * 64},
        architecture=gate.BLOCK_ARCHITECTURE,
        config_sha256=gate.config_identity(spec),
    )
    path = tmp_path / "startup.json"
    private_json(path, receipt)
    authentication = dict(
        identity=receipt["identity"],
        architecture=receipt["architecture"],
        qualification=dict(passed=True, default_off=True, evidence=dict(path=path.name, sha256=sha256_file(path))),
    )
    gate.qualify_default_reference(authentication, spec, tmp_path)
    spec["execution_candidates"] = {"kda": True}
    with pytest.raises(ValueError, match="configuration"):
        gate.qualify_default_reference(authentication, spec, tmp_path)


@pytest.mark.parametrize("change", ["file", "prompt", "length", "id"])
def test_authenticated_teacher_ids_and_prompt_binding_cannot_change(tmp_path, change):
    plan = small_plan()
    path = tmp_path / "teacher.json"
    teacher = dict(ids=[1, 2, 3], prompt_ids_sha256=plan["prompt_ids_sha256"])
    private_json(path, teacher)
    reference = dict(path=path.name, sha256=sha256_file(path))
    assert gate.load_teacher({"teacher_ids": reference}, tmp_path, plan, 4) == [1, 2, 3]
    if change == "file":
        teacher["ids"][0] = 2
    elif change == "prompt":
        teacher["prompt_ids_sha256"] = "0" * 64
    elif change == "length":
        teacher["ids"].pop()
    else:
        teacher["ids"][0] = 4
    private_json(path, teacher)
    if change != "file":
        reference["sha256"] = sha256_file(path)
    with pytest.raises(ValueError):
        gate.load_teacher({"teacher_ids": reference}, tmp_path, plan, 4)


def test_full_tensor_comparison_catches_last_element_after_many_io_chunks(tmp_path, precision):
    rows = gate.DEFAULT_CHUNK_BYTES // 4 * 3 + 1
    values = np.zeros(rows, dtype="float32")
    values.tofile(tmp_path / "reference")
    values[-1] = 1
    values.tofile(tmp_path / "candidate")
    result = gate.compare_tensor(
        tmp_path / "reference", tmp_path / "candidate", "float32", "floating_state", "changed_math", precision
    )
    assert not result["passed"]
    assert result["elements"] == rows
    assert result["max_abs_error"] == 1


def test_iteration_and_vocabulary_helpers_retain_complete_vectors(tmp_path):
    writer = DiagnosticWriter(tmp_path / "cuda", "cuda", max_chunk_bytes=8)
    capture_chunk(writer, 0, 3, iteration=0)
    capture_chunk(writer, 3, 1, iteration=1)
    capture_vocabulary(writer, np.arange(7, dtype="float32"), step=0, prompt_position=2, prompt_chunk=0)
    capture_vocabulary(writer, np.arange(7, dtype="float32"), step=1)
    path = writer.close()
    records = {row["key"]: row for row in json.loads(path.read_bytes())["tensors"]}
    assert records["vocabulary"]["shape"] == [2, 7]
    assert records["chunk_vocabulary"]["shape"] == [1, 7]
    assert np.fromfile(path.parent / records["chunks"]["path"], dtype="int64").tolist() == [0, 3, 3, 1]


def test_import_and_help_do_not_load_accelerator_libraries():
    code = (
        "import sys; import tools.sidecard.execution_model_gate; "
        "assert 'torch' not in sys.modules; assert 'torch_npu' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True, cwd=Path(__file__).resolve().parents[3])
    module = importlib.import_module("tools.sidecard.execution_model_gate")
    assert module.OUTPUT_TOKENS == 256
