# SPDX-License-Identifier: Apache-2.0
"""Resident composition and restoration, including failed capture/probes."""

import hashlib
import json

import pytest

from tools.glm_perf import reconstruction_control as control

BASE = "# retain this comment\ndef replacements(native_resources):\n    return {'existing': native_resources}\n"


def test_composition_preserves_base_text_and_existing_replacements(monkeypatch):
    from tools.glm_perf.resident_candidates import expert_reconstruction

    calls = []
    monkeypatch.setattr(
        expert_reconstruction,
        "extend_replacements",
        lambda changes, resources, profile: calls.append((changes, resources, profile)) or changes,
    )
    source = control.compose_source(BASE, "w4a8")
    assert source.startswith(BASE.replace("def replacements(", "def _reconstruction_base_replacements("))
    scope = {}
    exec(source, scope)
    resource = object()
    assert scope["replacements"](resource) == {"existing": resource}
    assert calls == [({"existing": resource}, resource, "w4a8")]
    with pytest.raises(ValueError):
        control.compose_source(source, "gm")


@pytest.mark.parametrize(
    "source, profile", [("x = 1", "gm"), (BASE, "unknown"), (BASE + "\ndef replacements(): pass\n", "gm")]
)
def test_invalid_factory_is_rejected_without_worker_mutations(source, profile):
    with pytest.raises(ValueError):
        control.compose_source(source, profile)


class Client:
    base_url = "http://localhost"
    timeout = 1

    def __init__(self, fail_switch=False, fail_restore=False, paused=False):
        self.paused, self.fail_switch, self.fail_restore = paused, fail_switch, fail_restore
        self.switches = []
        self.resumed = 0
        self.current = "decode_both"

    def _status_matches(self, rows):
        return True

    def _acknowledged_rpc(self, *args, **kwargs):
        return [
            {
                "rank": 0,
                "pid": 71,
                "weight_storage_digest": "weights",
                "graphs_dirty": False,
                "candidate": self.current,
                "mode": "graph",
                "digest": hashlib.sha256(BASE.encode()).hexdigest(),
                "native_loaded": {"reconstruction_v1": {}},
                "native_failed": False,
                **({"reconstruction": {"native_dispatches": 1}} if self.current.startswith("reconstruction_") else {}),
            }
        ]

    def request(self, *args, **kwargs):
        if args[0].startswith("/pause?"):
            self.paused = True
            return {"status": "paused"}
        return {"is_paused": self.paused}

    def switch(self, selection):
        self.switches.append(selection)
        if len(self.switches) == 1 and self.fail_switch:
            self.paused = True
            raise RuntimeError("capture failed")
        if selection.candidate == "decode_both" and self.fail_restore:
            raise RuntimeError("restoration failed")
        self.current = selection.candidate
        return self._acknowledged_rpc()

    def resume(self):
        self.resumed += 1
        self.paused = False


def mock_suite(monkeypatch, passed=True):
    monkeypatch.setattr(control, "make_groups", lambda *args: [])
    monkeypatch.setattr(control, "run_groups", lambda *args, **kwargs: [{"valid": True, "passed": passed}])
    monkeypatch.setattr(control, "summarize", lambda rows: {"count": len(rows)})


def test_successful_trial_restores_exact_current_fusions(monkeypatch, tmp_path):
    mock_suite(monkeypatch)
    client = Client()
    result = control.run_trial(client, BASE, "w4a8", tmp_path / "trial.json", "glm")
    assert result["restored"] is True
    assert client.switches[-1].source == BASE
    assert client.switches[-1].candidate == "decode_both"
    assert client.switches[-2].candidate == "baseline" and client.switches[-2].source == ""
    assert client.resumed == 1


def test_failed_capture_restores_and_resumes_previously_active_server(tmp_path):
    client = Client(fail_switch=True)
    output = tmp_path / "trial.json"
    with pytest.raises(RuntimeError, match="capture failed"):
        control.run_trial(client, BASE, "w4a8", output, "glm")
    assert client.switches[-1].source == BASE
    assert client.resumed == 1 and client.paused is False
    assert json.loads(output.read_text())["restored"] is True


def test_failed_probe_restores_before_returning_failure(monkeypatch, tmp_path):
    mock_suite(monkeypatch, passed=False)
    client = Client()
    with pytest.raises(RuntimeError, match="probe failed"):
        control.run_trial(client, BASE, "w4a8", tmp_path / "trial.json", "glm")
    assert client.switches[-1].source == BASE


def test_failed_restoration_retains_error_receipt(monkeypatch, tmp_path):
    mock_suite(monkeypatch)
    client = Client(fail_restore=True)
    output = tmp_path / "trial.json"
    with pytest.raises(RuntimeError, match="restoration failed"):
        control.run_trial(client, BASE, "w4a8", output, "glm")
    assert json.loads(output.read_text())["restoration_error"] == "restoration failed"


def test_existing_pause_and_wrong_base_are_rejected_before_switch(tmp_path):
    client = Client(paused=True)
    with pytest.raises(ValueError, match="already paused"):
        control.run_trial(client, BASE, "w4a8", tmp_path / "trial.json", "glm")
    assert client.switches == []
    client.paused = False
    with pytest.raises(ValueError, match="not the current"):
        control.run_trial(client, BASE + "\n", "w4a8", tmp_path / "trial.json", "glm")
    assert client.switches == []


def make_gate(tmp_path):
    names = ("glm_w4a8_pack.bin", "glm_w4a8_matmul.bin", "glm_reconstruction_bridge_v1.so")
    for name in names:
        (tmp_path / name).write_bytes(name.encode())
    gate = {
        "schema_version": 1,
        "complete": True,
        "binaries": {name: hashlib.sha256((tmp_path / name).read_bytes()).hexdigest() for name in names},
        "records": [{"activation_pack_exact": True, "native_reference_passed": True}],
        "graph_records": [{"passed": True}],
    }
    return gate


def test_manifest_requires_gates_for_the_exact_binary_and_graph(tmp_path):
    gate = make_gate(tmp_path)
    report = tmp_path / "gate.json"
    report.write_text(json.dumps(gate))
    value = control.manifest(tmp_path, report).value
    compile(value["validation_source"], "manifest", "exec")
    assert value["operators"] == ["glm_reconstruction_v1::launch"]
    assert len(value["assets"]) == 6
    (tmp_path / "glm_w4a8_pack.bin").write_bytes(b"changed")
    with pytest.raises(ValueError, match="exact native binaries"):
        control.manifest(tmp_path, report)
    gate = make_gate(tmp_path)
    gate["graph_records"] = []
    report.write_text(json.dumps(gate))
    with pytest.raises(ValueError, match="graph replay"):
        control.manifest(tmp_path, report)


def test_manifest_rejects_ungated_w3_and_incomplete_component_runs(tmp_path):
    gate = make_gate(tmp_path)
    report = tmp_path / "gate.json"
    report.write_text(json.dumps(gate))
    with pytest.raises(ValueError, match="W3 resources"):
        control.manifest(tmp_path, report, include_w3=True)
    gate["complete"] = False
    report.write_text(json.dumps(gate))
    with pytest.raises(ValueError, match="completed independent"):
        control.manifest(tmp_path, report)


def test_manifest_requires_exact_pipeline_support_library(tmp_path):
    gate = make_gate(tmp_path)
    support = tmp_path / "support.so"
    support.write_bytes(b"existing swiglu and combine operators")
    gate["support_libraries"] = [{"path": str(support), "sha256": hashlib.sha256(support.read_bytes()).hexdigest()}]
    report = tmp_path / "gate.json"
    report.write_text(json.dumps(gate))
    value = control.manifest(tmp_path, report).value
    assert gate["support_libraries"][0] in value["assets"]
    support.write_bytes(b"another build")
    with pytest.raises(ValueError, match="support library differs"):
        control.manifest(tmp_path, report)


def test_complete_pipeline_manifest_does_not_register_existing_support_operators(tmp_path):
    gate = make_gate(tmp_path)
    options = {"version": 1, "all_bits": True, "prefill_native": True, "tile_pipeline": True, "output_columns": 128}
    (tmp_path / "provenance.json").write_text(json.dumps({"_build": options}))
    gate["build_options"] = options
    gate["records"] = [
        {"weight_bits": bits, "geometry": {"rows": 128}, "activation_pack_exact": True, "native_reference_passed": True}
        for bits in (2, 3, 4)
    ]
    for key in ("graph_records", "prefill_graph_records", "moe_pipeline_records"):
        gate[key] = [{"weight_bits": bits, "passed": True} for bits in (2, 3, 4)]
    gate["lazy_metadata_records"] = [{"weight_bits": bits, "native_reference_passed": True} for bits in (2, 3, 4)]
    report = tmp_path / "gate.json"
    report.write_text(json.dumps(gate))
    value = control.manifest(tmp_path, report).value
    assert value["operators"] == ["glm_reconstruction_v1::launch"]
    assert "_C_ascend::npu_w2_swiglu_310" in value["validation_source"]
    gate["prefill_graph_records"] = []
    report.write_text(json.dumps(gate))
    with pytest.raises(ValueError, match="prefill native resource"):
        control.manifest(tmp_path, report)


def test_trial_rejects_capture_that_only_used_fallback_and_restores(tmp_path):
    class FallbackClient(Client):
        def _acknowledged_rpc(self, *args, **kwargs):
            receipts = super()._acknowledged_rpc(*args, **kwargs)
            if "reconstruction" in receipts[0]:
                receipts[0]["reconstruction"]["native_dispatches"] = 0
            return receipts

    client = FallbackClient()
    with pytest.raises(RuntimeError, match="did not exercise native"):
        control.run_trial(client, BASE, "w4a8", tmp_path / "trial.json", "glm")
    assert client.switches[-1].source == BASE


def test_full_coverage_trial_rejects_partial_native_capture_before_requests(tmp_path):
    class PartialClient(Client):
        def _acknowledged_rpc(self, *args, **kwargs):
            rows = super()._acknowledged_rpc(*args, **kwargs)
            if "reconstruction" in rows[0]:
                rows[0]["reconstruction"]["fallback_dispatches"] = 1
                rows[0]["reconstruction"]["bank_coverage"] = {"W3": {"native": 0, "fallback": 1}}
            return rows

    client = PartialClient()
    with pytest.raises(RuntimeError, match="still contains fallback"):
        control.run_trial(client, BASE, "w4a8", tmp_path / "trial.json", "glm", require_full_coverage=True)
    assert client.switches[-1].source == BASE
    assert client.resumed == 1


def test_full_coverage_trial_retains_requests_when_prefill_falls_back(monkeypatch, tmp_path):
    mock_suite(monkeypatch)

    class PrefillFallbackClient(Client):
        requests_finished = False

        def _acknowledged_rpc(self, *args, **kwargs):
            rows = super()._acknowledged_rpc(*args, **kwargs)
            if "reconstruction" in rows[0]:
                rows[0]["reconstruction"].update(
                    fallback_dispatches=int(self.requests_finished), bank_coverage={"W3": {"native": 1}}
                )
            return rows

    client = PrefillFallbackClient()

    def requests(*args, **kwargs):
        client.requests_finished = True
        return [{"valid": True, "passed": True}]

    monkeypatch.setattr(control, "run_groups", requests)
    output = tmp_path / "trial.json"
    with pytest.raises(RuntimeError, match="requests still contain fallback"):
        control.run_trial(client, BASE, "int4a8", output, "glm", require_full_coverage=True)
    report = json.loads(output.read_text())
    assert report["restored"] is True
    assert report["records"] == [{"valid": True, "passed": True}]
    assert report["trial_error"]["message"] == "requests still contain fallback expert projections"
    assert client.switches[-1].source == BASE
    assert client.resumed == 1


def test_versioned_candidate_imports_the_frozen_adapter():
    source = control.compose_source(BASE, "w4a8", "reconstruction_v2")
    assert "from glm_reconstruction_v2_helpers.expert_reconstruction import" in source
    assert "resource_name='reconstruction_v2'" in source
    compile(source, "candidate", "exec")
    with pytest.raises(ValueError):
        control.compose_source(BASE, "w4a8", "arbitrary_package")


@pytest.mark.parametrize("tile_pipeline", (False, True))
def test_private_helper_package_keeps_geometry_types_and_namespace_consistent(tmp_path, tile_pipeline):
    import shutil
    import sys
    from pathlib import Path

    gate = make_gate(tmp_path)
    old_bridge = tmp_path / "glm_reconstruction_bridge_v1.so"
    bridge = tmp_path / "glm_reconstruction_bridge_v913.so"
    old_bridge.rename(bridge)
    gate["binaries"][bridge.name] = gate["binaries"].pop(old_bridge.name)
    namespace = "glm_reconstruction_v913"
    package = namespace + "_helpers"
    root = tmp_path / package
    root.mkdir()
    (root / "__init__.py").write_text("")
    shutil.copy2(Path(control.__file__).with_name("reconstruction_native.py"), root / "reconstruction_native.py")
    (root / "glm_int4.py").write_text(
        "from .reconstruction_native import ProjectionGeometry\n"
        "class NativeW4Projection:\n"
        "    def __init__(self, pack, matrix, geometries, *, namespace, tile_pipeline=False, output_columns=16):\n"
        "        assert all(isinstance(g, ProjectionGeometry) for g in geometries)\n"
        "        self.namespace = namespace\n"
        "        self.tile_pipeline, self.output_columns = tile_pipeline, output_columns\n"
    )
    (root / "reconstruction_probe.py").write_text("# isolated probe fixture\n")
    (root / "expert_reconstruction.py").write_text("# isolated adapter fixture\n")
    options = {
        "version": 913,
        "helper_package": package,
        "tile_pipeline": tile_pipeline,
        "output_columns": 128 if tile_pipeline else 16,
    }
    (tmp_path / "provenance.json").write_text(json.dumps({"_build": options}))
    gate["build_options"] = options
    report = tmp_path / "gate.json"
    report.write_text(json.dumps(gate))
    value = control.manifest(tmp_path, report).value
    assert value["name"] == "reconstruction_v913"
    assert value["operators"] == [namespace + "::launch"]
    gate["build_options"] = {**options, "tile_pipeline": not tile_pipeline}
    report.write_text(json.dumps(gate))
    with pytest.raises(ValueError, match="metadata options"):
        control.manifest(tmp_path, report)
    scope = {}
    exec(value["validation_source"], scope)
    try:
        projection = scope["prepare"]()["w4a8"]
        assert projection.namespace == namespace
        assert projection.tile_pipeline is tile_pipeline
        assert projection.output_columns == (128 if tile_pipeline else 16)
    finally:
        for name in list(sys.modules):
            if name == package or name.startswith(package + "."):
                sys.modules.pop(name)
