# SPDX-License-Identifier: Apache-2.0
"""Portable gate runtime lifecycle/ownership tests, without device allocation."""

import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tools.sidecard import execution_diagnostic_runtime as runtime
from tools.sidecard.execution_model_gate import BLOCK_ARCHITECTURE, OWNERS, config_identity, make_plan, validate_capture
from tools.sidecard.stage_diagnostics import private_json, sha256_file


@pytest.fixture
def setup(tmp_path):
    root = tmp_path / "private"
    root.mkdir(mode=0o700)
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    model = dict(layer_types=["full_attention"] * 45, hidden_size=2, index_topk=4, kv_lora_rank=2, vocab_size=4)
    private_json(checkpoint / "config.json", {"text_config": model})
    spec = dict(
        checkpoint=str(checkpoint),
        max_context=16,
        prefill_chunk_size=2,
        architecture=BLOCK_ARCHITECTURE,
        stage_transport="direct-dma",
        execution_diagnostics=dict(
            enabled=True, directory_root=str(root), authentication=dict(path="auth.json", sha256="pending")
        ),
    )

    def write_config():
        artifacts = []
        for category in ("source", "native", "checkpoint"):
            path = tmp_path / f"input-{category}"
            path.write_text(category)
            artifacts.append(dict(path=str(path), sha256=sha256_file(path), category=category))
        receipt = dict(
            config_sha256=config_identity(spec),
            architecture=spec["architecture"],
            files=artifacts,
            identity={key: "a" * 64 for key in ("source_sha256", "native_sha256", "checkpoint_sha256")},
        )
        private_json(tmp_path / "auth.json", receipt)
        spec["execution_diagnostics"]["authentication"]["sha256"] = sha256_file(tmp_path / "auth.json")
        private_json(tmp_path / "config.json", spec)

    write_config()
    return SimpleNamespace(config=tmp_path / "config.json", root=root, spec=spec, model=model, write=write_config)


def layers(owner):
    indices = [index for index in range(45) if (index < 18 or index >= 33) == (owner == "cuda")]
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
        for index in indices
    }


def deferred(rows):
    return (
        np.zeros((rows, 2), dtype="float16"),
        np.zeros((rows, 4, 2), dtype="float32"),
        np.zeros((rows, 4, 1), dtype="float32"),
        np.zeros((rows, 4, 4), dtype="float32"),
    )


def make_observer(setup, owner="cuda"):
    return runtime.OwnerDiagnostic(setup.config, owner, layers(owner), lambda: None)


def begin_owner(setup, observer, *, directory=None):
    plan = make_plan("case", [1, 2, 3], setup.model, setup.spec, output_tokens=2)
    directory = directory or setup.root / "fixture"
    directory.mkdir(mode=0o700)
    path = directory / "plan.json"
    private_json(path, dict(plan=plan, config_sha256=config_identity(setup.spec), directory=str(directory)))
    command = dict(
        action="begin",
        nonce="a" * 32,
        plan_sha256=runtime.canonical_sha256(plan),
        plan=dict(path=str(path), sha256=sha256_file(path)),
        directory=str(directory),
    )
    observer.handle_command(command)
    return plan, command


def complete_chunk(observer, position, rows, *, skip_route=None, graph=None):
    observer.begin_chunk(position, rows)
    if observer.owner == "cuda":
        observer.boundary("prefix", deferred(rows))
    else:
        observer.boundary("input", deferred(rows))
    if graph is not None:
        observer.replayed(graph)
    else:
        for layer in observer.layers.values():
            if layer.index >= 3 and layer.index != skip_route:
                observer.route(layer, np.zeros((rows, 8), dtype="int64"), np.zeros((rows, 8), dtype="float32"))
            observer.selection(layer, None)
        if observer.owner == "cuda":
            observer.suffix_state(deferred(rows))
    if observer.owner == "cuda":
        observer.vocabulary(np.arange(4, dtype="float32"))
    else:
        observer.boundary("output", deferred(rows))
    for layer in observer.layers.values():
        layer.length = position + rows
        layer.indexer.pool_count, layer.indexer.tail_length = divmod(layer.length, 4)
    observer.end_chunk()


def test_complete_controller_collects_three_authentic_owner_manifests(setup):
    workers = {owner: make_observer(setup, owner) for owner in OWNERS[1:]}
    adapter = runtime.CoordinatorDiagnostic(
        setup.config, layers("cuda"), lambda owner, command: workers[owner].handle_command(command), lambda: None
    )
    plan = make_plan("case", [1, 2, 3], setup.model, setup.spec, output_tokens=2)
    directory = setup.root / "fixture"
    directory.mkdir(mode=0o700)
    adapter.begin_fixture(plan, directory)
    observers = {"cuda": adapter.observer, **workers}
    for position, rows in plan["schedule"]:
        for observer in observers.values():
            complete_chunk(observer, position, rows)
        if position + rows == plan["prompt_tokens"]:
            adapter.finish_phase("prefill")
    adapter.finish_phase("decode")
    captures = adapter.end_fixture()
    assert set(captures) == set(OWNERS)
    for owner in OWNERS:
        body, _ = validate_capture(captures[owner], plan["expected"][owner], plan["allocation_geometry"][owner])
        assert body["runtime_receipt"]["chunks"] == 3
        assert body["owner"] == owner


def test_disabled_diagnostics_skip_authentication_and_storage(tmp_path):
    path = tmp_path / "config.json"
    private_json(path, {})
    assert runtime.validate_runtime(path) is None
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize("change", ["socket", "expert", "public_root", "relative_root"])
def test_unsupported_or_unprivate_runtime_rejected_before_allocation(setup, change):
    if change == "socket":
        setup.spec["stage_transport"] = "socket"
    elif change == "expert":
        setup.spec["architecture"] = "cuda-backbone-ascend-experts"
    elif change == "public_root":
        os.chmod(setup.root, 0o755)
    else:
        setup.spec["execution_diagnostics"]["directory_root"] = "relative"
    setup.write()
    with pytest.raises(ValueError):
        runtime.validate_runtime(setup.config)


def test_owner_callbacks_reject_foreign_layers_and_incomplete_suffix(setup):
    observer = make_observer(setup)
    foreign = SimpleNamespace(index=18, linear=False)
    with pytest.raises(ValueError, match="owner"):
        observer.route(foreign, None, None)
    with pytest.raises(ValueError, match="pre-contraction"):
        observer.suffix_state((None, None, None, None))


def test_graph_capture_never_observes_or_transfers_tensors(setup):
    observer = make_observer(setup)
    observer.synchronize = lambda: pytest.fail("fence during graph capture")
    observer.capture_begin("span3")
    observer.route(observer.layers[3], np.zeros((1, 8), dtype="int64"), np.zeros((1, 8), dtype="float32"))
    observer.capture_end("span3")
    assert observer.writer is None
    assert len(observer.graphs["span3"]) == 1


def test_discard_capture_is_startup_only_and_preserves_other_graphs(setup):
    observer = make_observer(setup)
    observer.capture_begin("valid")
    with pytest.raises(ValueError, match="active capture"):
        observer.discard_capture("valid")
    observer.capture_end("valid")
    observer.capture_begin("rejected")
    observer.capture_end("rejected")
    observer.discard_capture("rejected")
    observer.discard_capture("unknown")
    assert set(observer.graphs) == {"valid"}
    begin_owner(setup, observer)
    with pytest.raises(ValueError, match="fixture"):
        observer.discard_capture("valid")


def test_captured_aliases_and_late_registration_are_rejected(setup):
    observer = make_observer(setup)
    observer.capture_begin("span3")
    ids = np.zeros((1, 8), dtype="int64")
    observer.route(observer.layers[3], ids, np.zeros((1, 8), dtype="float32"))
    with pytest.raises(ValueError, match="alias"):
        observer.route(observer.layers[4], ids, np.zeros((1, 8), dtype="float32"))
    observer.capture_end("span3")
    begin_owner(setup, observer)
    with pytest.raises(ValueError, match="before any fixture"):
        observer.capture_begin("late")


def test_missing_selector_fails_before_chunk_receipt(setup):
    observer = make_observer(setup)
    begin_owner(setup, observer)
    with pytest.raises(ValueError, match="missing.*selector"):
        complete_chunk(observer, 0, 2, skip_route=3)
    assert "chunks" not in observer.writer.records


def test_stale_chunks_phases_and_delayed_graph_observations_fail(setup):
    observer = make_observer(setup)
    begin_owner(setup, observer)
    with pytest.raises(ValueError, match="stale/reordered"):
        observer.begin_chunk(2, 1)
    with pytest.raises(ValueError, match="phase"):
        observer.finish_phase("decode")
    with pytest.raises(ValueError, match="delayed"):
        observer.replayed("unregistered")
    with pytest.raises(ValueError, match="incomplete"):
        observer.close()


def test_boundary_bytes_are_observed_before_later_alias_overwrite(setup):
    observer = make_observer(setup)
    begin_owner(setup, observer)
    observer.begin_chunk(0, 2)
    state = deferred(2)
    state[0].fill(7)
    observer.boundary("prefix", state)
    state[0].fill(9)
    record = observer.writer.records["prefill/boundary/prefix/hidden"]
    assert np.fromfile(observer.writer.directory / record["path"], dtype="float16").tolist() == [7] * 4
    with pytest.raises(ValueError, match="suffix must use"):
        observer.boundary("suffix", state)


def capture_all_graph_refs(observer):
    observer.capture_begin("all")
    refs = {}
    for layer in observer.layers.values():
        if layer.index >= 3:
            ids, weights = np.zeros((1, 8), dtype="int64"), np.zeros((1, 8), dtype="float32")
            observer.route(layer, ids, weights)
            refs[layer.index] = ids
        observer.selection(layer, None)
    observer.suffix_state(deferred(1))
    observer.capture_end("all")
    return refs


def test_graph_replay_observes_persistent_current_refs_and_full_cache(setup):
    setup.spec["cuda_decode_graphs"] = True
    setup.write()
    observer = make_observer(setup)
    refs = capture_all_graph_refs(observer)
    plan, _ = begin_owner(setup, observer)
    complete_chunk(observer, 0, 2)
    complete_chunk(observer, 2, 1)
    observer.finish_phase("prefill")
    refs[3].fill(2)
    complete_chunk(observer, 3, 1, graph="all")
    observer.finish_phase("decode")
    path = observer.close()
    body, records = validate_capture(path, plan["expected"]["cuda"], plan["allocation_geometry"]["cuda"])
    assert body["runtime_receipt"]["replay_counts"] == {"all": 1}
    assert (
        np.fromfile(path.parent / records["decode/selection/layer3/route"]["path"], dtype="int64").tolist() == [2] * 8
    )


def test_configured_graph_cannot_silently_run_eager_decode(setup):
    setup.spec["cuda_decode_graphs"] = True
    setup.write()
    observer = make_observer(setup)
    capture_all_graph_refs(observer)
    begin_owner(setup, observer)
    complete_chunk(observer, 0, 2)
    complete_chunk(observer, 2, 1)
    observer.finish_phase("prefill")
    with pytest.raises(ValueError, match="replaced by eager"):
        complete_chunk(observer, 3, 1)


@pytest.mark.parametrize("change", ["nonce", "plan", "phase"])
def test_owner_control_rejects_stale_nonce_hash_and_phase(setup, change):
    observer = make_observer(setup)
    _, command = begin_owner(setup, observer)
    control = dict(action="phase", phase="prefill", nonce=command["nonce"], plan_sha256=command["plan_sha256"])
    if change == "nonce":
        control["nonce"] = "b" * 32
    elif change == "plan":
        control["plan_sha256"] = "b" * 64
    else:
        control["phase"] = "decode"
    with pytest.raises(ValueError):
        observer.handle_command(control)


def test_private_plan_escape_drift_and_changed_coverage_are_rejected(setup, tmp_path):
    observer = make_observer(setup)
    plan, command = begin_owner(setup, observer)
    other = make_observer(setup, "npu-rank0")
    escaped = {**command, "plan": {"path": str(tmp_path / "outside"), "sha256": "a" * 64}}
    with pytest.raises((ValueError, FileNotFoundError)):
        other.handle_command(escaped)
    path = Path(command["plan"]["path"])
    wrapper = json.loads(path.read_bytes())
    wrapper["plan"]["expected"]["npu-rank0"].pop("prefill/state/layer18/kv")
    private_json(path, wrapper)
    with pytest.raises(ValueError, match="drifted"):
        other.handle_command(command)
    command["plan"]["sha256"] = sha256_file(path)
    command["plan_sha256"] = runtime.canonical_sha256(wrapper["plan"])
    with pytest.raises(ValueError, match="complete-state"):
        other.handle_command(command)
    assert plan["architecture"] == BLOCK_ARCHITECTURE


def test_reply_nonce_and_unbounded_inline_plan_are_rejected(setup):
    adapter = runtime.CoordinatorDiagnostic(setup.config, layers("cuda"), lambda owner, command: {}, lambda: None)
    with pytest.raises(ValueError, match="stale command"):
        adapter._roundtrip("npu-rank0", dict(action="phase", phase="prefill", nonce="a" * 32, plan_sha256="b" * 64))
    with pytest.raises(ValueError, match="16KiB"):
        runtime.encode_command(dict(plan="x" * runtime.MAX_CONTROL_BYTES))


def test_changed_configuration_and_mid_fixture_plan_are_rejected(setup):
    observer = make_observer(setup)
    _, command = begin_owner(setup, observer)
    complete_chunk(observer, 0, 2)
    complete_chunk(observer, 2, 1)
    path = Path(command["plan"]["path"])
    original = path.read_bytes()
    path.write_bytes(original + b" ")
    with pytest.raises(ValueError, match="drifted during"):
        observer.finish_phase("prefill")
    path.write_bytes(original)
    setup.spec["npu_decode_graphs"] = True
    setup.write()
    with pytest.raises(ValueError, match="configuration drifted"):
        observer.finish_phase("prefill")


def test_completed_fixture_nonce_cannot_be_replayed(setup):
    observer = make_observer(setup)
    _, command = begin_owner(setup, observer)
    complete_chunk(observer, 0, 2)
    complete_chunk(observer, 2, 1)
    observer.finish_phase("prefill")
    complete_chunk(observer, 3, 1)
    observer.finish_phase("decode")
    observer.close()
    with pytest.raises(ValueError, match="fresh 128-bit"):
        observer.handle_command(command)


def test_symlink_to_outside_private_root_is_rejected(setup, tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    link = setup.root / "linked"
    link.symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="escapes"):
        runtime._inside(setup.root, link)


def test_privileged_file_handoff_rejects_links_and_unexpected_files(setup, tmp_path):
    directory = setup.root / "worker"
    directory.mkdir(mode=0o700)
    (directory / "manifest.json").write_text("{}")
    external = tmp_path / "external"
    external.write_text("private bytes")
    os.link(external, directory / "data.bin")
    body = {"tensors": [{"path": "data.bin"}]}
    with pytest.raises(ValueError, match="hardlinks"):
        runtime._handoff_private_files(directory, body)
    (directory / "data.bin").unlink()
    (directory / "data.bin").write_text("private bytes")
    runtime._handoff_private_files(directory, body)
    (directory / "unknown").write_text("unrelated")
    with pytest.raises(ValueError, match="unexpected files"):
        runtime._handoff_private_files(directory, body)
