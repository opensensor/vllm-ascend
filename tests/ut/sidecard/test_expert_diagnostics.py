# SPDX-License-Identifier: Apache-2.0
"""Portable expert diagnostic protocol/data tests; no accelerator evidence."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tools.sidecard import expert_diagnostics as expert
from tools.sidecard.execution_model_gate import (
    EXPERT_ARCHITECTURE,
    OWNERS,
    compare_captures,
    config_identity,
    make_plan,
    validate_capture,
)
from tools.sidecard.stage_diagnostics import private_json, sha256_file


@pytest.fixture
def setup(tmp_path):
    root, checkpoint = tmp_path / "private", tmp_path / "checkpoint"
    root.mkdir(mode=0o700)
    checkpoint.mkdir()
    model = dict(layer_types=["full_attention"] * 19, hidden_size=2, index_topk=4, kv_lora_rank=2, vocab_size=4)
    private_json(checkpoint / "config.json", {"text_config": model})
    spec = dict(
        checkpoint=str(checkpoint),
        max_context=16,
        prefill_chunk_size=2,
        prefill_window_size=4,
        architecture=EXPERT_ARCHITECTURE,
        expert_transport="direct-dma",
        execution_diagnostics=dict(
            enabled=True, directory_root=str(root), authentication=dict(path="auth.json", sha256="pending")
        ),
    )

    def write():
        artifacts = []
        for category in ("source", "native", "checkpoint"):
            path = checkpoint / "config.json" if category == "checkpoint" else tmp_path / category
            if category != "checkpoint":
                path.write_text(category)
            artifacts.append(dict(path=str(path), sha256=sha256_file(path), category=category))
        private_json(
            tmp_path / "auth.json",
            dict(
                config_sha256=config_identity(spec),
                architecture=EXPERT_ARCHITECTURE,
                files=artifacts,
                identity={key: "a" * 64 for key in ("source_sha256", "native_sha256", "checkpoint_sha256")},
            ),
        )
        spec["execution_diagnostics"]["authentication"]["sha256"] = sha256_file(tmp_path / "auth.json")
        private_json(tmp_path / "runtime.json", spec)

    write()
    return SimpleNamespace(root=root, spec=spec, model=model, config=tmp_path / "runtime.json", write=write)


def layers():
    return {
        index: SimpleNamespace(
            index=index,
            linear=False,
            length=0,
            cache=np.zeros((16, 2), dtype="float16"),
            indexer=SimpleNamespace(
                pool_count=0,
                tail_length=0,
                pools=np.zeros((4, 128), dtype="float16"),
                tail_keys=np.zeros((3, 128), dtype="float32"),
                tail_gates=np.zeros((3, 128), dtype="float32"),
            ),
        )
        for index in range(19)
    }


def operands(position, rows):
    active = [position % 2 == 0, True]
    ids = np.zeros((rows, 8), dtype="int32")
    ids[:, 1] = 144
    weights = np.zeros((rows, 8), dtype="float32")
    weights[:, 0] = 0.25 if active[0] else 0
    weights[:, 1] = 0.25
    return np.ones((rows, 2), dtype="float16"), ids, weights, active


def fixture(setup):
    workers = {f"npu-rank{rank}": expert.ExpertOwner(setup.config, rank, {18: []}, lambda: None) for rank in range(2)}
    adapter = expert.ExpertCoordinatorDiagnostic(
        setup.config,
        layers(),
        lambda owner, command: workers[owner].handle_command(command),
        lambda: None,
    )
    plan = make_plan("case", [1, 2, 3, 4, 5], setup.model, setup.spec, output_tokens=3)
    directory = setup.root / "fixture"
    directory.mkdir(mode=0o700)
    adapter.begin_fixture(plan, directory)
    return adapter, workers, plan


def window(adapter, workers, sequences, position, rows):
    observer = adapter.observer
    observer.begin_window(position, rows)
    x, ids, weights, active = operands(position, rows)
    for layer in observer.layers.values():
        for offset in range(0, rows, 2):
            size = min(2, rows - offset)
            observer.selection(layer, None, rows=size)
            if layer.index >= 3:
                observer.route(layer, ids[offset : offset + size].astype("int64"), weights[offset : offset + size])
        if layer.index == 18:
            adapter.dispatch_call(18, x, ids, weights, active, sequences)
            for rank in range(2):
                if active[rank]:
                    execution = (
                        "decode-graph"
                        if rows == 1 and adapter.observer.spec.get("direct_dma", {}).get("npu_expert_graphs")
                        else "eager"
                    )
                    workers[f"npu-rank{rank}"].observe_compute(
                        sequences[rank],
                        18,
                        x,
                        ids,
                        weights,
                        np.full((rows, 2), rank + 1, dtype="float32"),
                        execution=execution,
                    )
                    sequences[rank] += 1
    for offset in range(0, rows, 2):
        observer.vocabulary(np.arange(4, dtype="float32"), position=position + offset, rows=min(2, rows - offset))
    for layer in observer.layers.values():
        layer.length = position + rows
        layer.indexer.pool_count, layer.indexer.tail_length = divmod(layer.length, 4)
    observer.end_window()


def completed(setup):
    adapter, workers, plan = fixture(setup)
    sequences = [7, 11]
    for position, rows in ((0, 4), (4, 1)):
        window(adapter, workers, sequences, position, rows)
    adapter.finish_phase("prefill")
    for position in (5, 6):
        window(adapter, workers, sequences, position, 1)
    adapter.finish_phase("decode")
    return adapter.end_fixture(), plan


def test_full_expert_gate_preserves_layer_major_batches_and_proves_native_skips(setup):
    captures, plan = completed(setup)
    for owner in OWNERS:
        body, _ = validate_capture(
            captures[owner],
            plan["expected"][owner],
            plan["allocation_geometry"][owner],
            expert_schedule=plan["expert_physical_schedule"],
            expert_execution=plan["expert_execution"],
        )
        if owner != "cuda":
            calls = body["expert_provenance"]["calls"]
            assert [row["call"]["rows"] for row in calls] == [4, 1, 1, 1]
            assert len(calls) == 4
    body = json.loads(captures["npu-rank0"].read_bytes())
    skip = body["expert_provenance"]["calls"][2]
    assert skip["status"] == "verified_zero_skip"
    assert skip["execution"] == "skipped"
    expert.validate_expert_handoffs(captures, plan)
    precision = dict(
        fixed_before_native_device_tests=True,
        model_bounds=dict(
            same_math={"all_tensors": "bitwise"},
            changed_math=dict(
                floating_state=dict(atol=0, rtol=0),
                vocabulary=dict(atol=0, rtol=0),
                selectors="bitwise",
                integer_boolean_state="bitwise",
                nonfinite="reject",
                unknown_tensor_kind="reject",
            ),
        ),
    )
    assert compare_captures(captures, captures, plan, plan, "same_math", precision)["passed"]


def test_expert_capture_without_physical_provenance_cannot_qualify(setup):
    captures, plan = completed(setup)
    body = json.loads(captures["npu-rank0"].read_bytes())
    body.pop("expert_provenance")
    private_json(captures["npu-rank0"], body)
    with pytest.raises(ValueError, match="provenance"):
        validate_capture(
            captures["npu-rank0"],
            plan["expected"]["npu-rank0"],
            {},
            expert_schedule=plan["expert_physical_schedule"],
            expert_execution=plan["expert_execution"],
        )


@pytest.mark.parametrize("change", ["status", "position", "rank", "sequence", "span", "mode"])
def test_physical_call_witness_cannot_change(setup, change):
    captures, plan = completed(setup)
    path = captures["npu-rank0"]
    body = json.loads(path.read_bytes())
    receipt = body["expert_provenance"]["calls"][0]
    if change == "status":
        receipt["status"] = "verified_zero_skip"
    elif change in ("position", "rank", "sequence"):
        receipt["call"][change] += 1
    elif change == "span":
        receipt["tensor_spans"]["input"]["sha256"] = "0" * 64
    else:
        receipt["execution"] = "decode-graph"
    records = {row["key"]: row for row in body["tensors"]}
    with pytest.raises(ValueError):
        expert.validate_expert_provenance(
            body, path.parent, records, plan["expert_physical_schedule"], plan["expert_execution"]
        )


def test_zero_skip_requires_authentic_inputs_and_cannot_hide_owned_routes(setup):
    adapter, workers, _ = fixture(setup)
    adapter.observer.begin_window(0, 4)
    x, ids, weights, _ = operands(0, 4)
    with pytest.raises(ValueError, match="positive native-owned"):
        adapter.dispatch_call(18, x, ids, weights, [False, True], [0, 0])
    assert not workers["npu-rank0"].receipts


def test_actual_compute_needs_metadata_and_real_tag_geometry(setup):
    adapter, workers, _ = fixture(setup)
    worker = workers["npu-rank0"]
    x, ids, weights, active = operands(0, 4)
    output = np.zeros((4, 2), dtype="float32")
    with pytest.raises(ValueError, match="lacks authenticated"):
        worker.observe_compute(0, 18, x, ids, weights, output, execution="eager")
    adapter.observer.begin_window(0, 4)
    adapter.dispatch_call(18, x, ids, weights, active, [0, 0])
    with pytest.raises(ValueError, match="COMMAND/upload"):
        worker.observe_compute(1, 18, x, ids, weights, output, execution="eager")


def test_expert_partial_dtype_and_mode_must_match_actual_configuration(setup):
    setup.spec["direct_dma"] = {"npu_expert_graphs": True}
    setup.spec["prefill_chunk_size"] = setup.spec["prefill_window_size"] = 1
    setup.write()
    adapter, workers, _ = fixture(setup)
    adapter.observer.begin_window(0, 1)
    x, ids, weights, active = operands(0, 1)
    adapter.dispatch_call(18, x, ids, weights, active, [0, 0])
    with pytest.raises(ValueError, match="replay path"):
        workers["npu-rank0"].observe_compute(
            0, 18, x, ids, weights, np.zeros((1, 2), dtype="float32"), execution="eager"
        )
    with pytest.raises(ValueError, match="dtype"):
        workers["npu-rank0"].observe_compute(
            0, 18, x, ids, weights, np.zeros((1, 2), dtype="float16"), execution="decode-graph"
        )


def test_layer_major_window_requires_actual_dense_rows_and_all_vectors(setup):
    adapter, _, _ = fixture(setup)
    observer = adapter.observer
    observer.begin_window(0, 4)
    with pytest.raises(ValueError, match="actual row count"):
        observer.selection(observer.layers[0], None)
    with pytest.raises(ValueError, match="concurrent windows"):
        observer.begin_window(0, 4)
    with pytest.raises(ValueError, match="EVERY|every"):
        observer.vocabulary(np.zeros(4, dtype="float32"))
    with pytest.raises(ValueError, match="omitted actual"):
        observer.end_window()


def test_native_phase_cannot_complete_missing_physical_calls(setup):
    _, workers, _ = fixture(setup)
    with pytest.raises(ValueError, match="missing/unfinished"):
        workers["npu-rank0"].finish_phase("prefill")


def test_cuda_graph_registration_and_replay_observe_current_selector_refs(setup):
    observer = expert.ExpertCudaObserver(setup.config, layers(), lambda: None)
    observer.capture_begin("route")
    ids = np.zeros((1, 8), dtype="int64")
    observer.route(observer.layers[3], ids, np.ones((1, 8), dtype="float32"))
    observer.selection(observer.layers[3], None, rows=1)
    observer.capture_end("route")
    assert observer.graphs["route"][("indexer", 3)] == (None, 1)
    assert observer.writer is None


def test_expert_graph_replay_captures_current_refs_without_eager_substitution(setup):
    setup.spec["cuda_decode_graphs"] = True
    setup.write()
    workers = {f"npu-rank{rank}": expert.ExpertOwner(setup.config, rank, {18: []}, lambda: None) for rank in range(2)}
    adapter = expert.ExpertCoordinatorDiagnostic(
        setup.config, layers(), lambda owner, command: workers[owner].handle_command(command), lambda: None
    )
    observer = adapter.observer
    observer.capture_begin("all")
    refs = {}
    for layer in observer.layers.values():
        observer.selection(layer, None, rows=1)
        if layer.index >= 3:
            ids, weights = np.zeros((1, 8), dtype="int64"), np.zeros((1, 8), dtype="float32")
            observer.route(layer, ids, weights)
            refs[layer.index] = ids, weights
    observer.capture_end("all")
    plan = make_plan("case", [1, 2, 3, 4, 5], setup.model, setup.spec, output_tokens=3)
    directory = setup.root / "fixture"
    directory.mkdir(mode=0o700)
    adapter.begin_fixture(plan, directory)
    sequences = [7, 11]
    window(adapter, workers, sequences, 0, 4)
    window(adapter, workers, sequences, 4, 1)
    adapter.finish_phase("prefill")
    for position in (5, 6):
        observer.begin_window(position, 1)
        x, ids, weights, active = operands(position, 1)
        for captured_ids, captured_weights in refs.values():
            captured_ids[:] = ids
            captured_weights[:] = weights
        observer.replayed("all")
        adapter.dispatch_call(18, x, ids, weights, active, sequences)
        for rank in range(2):
            if active[rank]:
                workers[f"npu-rank{rank}"].observe_compute(
                    sequences[rank], 18, x, ids, weights, np.ones((1, 2), dtype="float32"), execution="eager"
                )
                sequences[rank] += 1
        observer.vocabulary(np.arange(4, dtype="float32"))
        for layer in observer.layers.values():
            layer.length = position + 1
            layer.indexer.pool_count, layer.indexer.tail_length = divmod(layer.length, 4)
        observer.end_window()
    adapter.finish_phase("decode")
    captures = adapter.end_fixture()
    assert json.loads(captures["cuda"].read_bytes())["runtime_receipt"]["replay_counts"] == {"all": 2}
    expert.validate_expert_handoffs(captures, plan)


def test_handoff_checks_real_cuda_operands_not_just_native_self_consistency(setup):
    captures, plan = completed(setup)
    body = json.loads(captures["cuda"].read_bytes())
    record = next(row for row in body["tensors"] if row["key"] == "prefill/expert/layer18/input")
    path = captures["cuda"].parent / record["path"]
    data = np.fromfile(path, dtype="float16")
    data[-1] = 2
    data.tofile(path)
    with pytest.raises(ValueError, match="operand bytes"):
        expert.validate_expert_handoffs(captures, plan)


def test_changed_skip_artifact_and_native_hot_weights_fail(setup):
    captures, plan = completed(setup)
    path = captures["npu-rank0"]
    body = json.loads(path.read_bytes())
    receipt = body["expert_provenance"]["calls"][2]
    skip_path = Path(receipt["skip_inputs"]["path"])
    skip_path.write_bytes(skip_path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="skip.*changed"):
        expert.validate_expert_provenance(
            body,
            path.parent,
            {row["key"]: row for row in body["tensors"]},
            plan["expert_physical_schedule"],
            plan["expert_execution"],
        )
    _, ids, weights, _ = operands(0, 1)
    with pytest.raises(ValueError, match="hot CUDA"):
        expert._route_count(ids, weights, 0, [0])


def test_native_command_nonce_cannot_be_reused_from_other_fixture(setup):
    adapter, workers, plan = fixture(setup)
    call = dict(**plan["expert_physical_schedule"][0], rank=0, sequence=0, active=True, index=0)
    command = dict(action="call", nonce="f" * 32, plan_sha256=adapter.command["plan_sha256"], call=call)
    with pytest.raises(ValueError, match="stale native"):
        workers["npu-rank0"].handle_command(command)
