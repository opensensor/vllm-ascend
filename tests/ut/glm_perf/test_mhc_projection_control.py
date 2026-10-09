# SPDX-License-Identifier: Apache-2.0
"""Projection admission binds exact binaries, helpers and changed-input gates."""

import json

import pytest

from tools.glm_perf.mhc_projection_control import digest, manifest


def fixtures(tmp_path):
    package = tmp_path / "glm_reconstruction_v1011_helpers"
    package.mkdir()
    helper = package / "mhc_projection.py"
    helper.write_text("# fixture\n")
    init = package / "__init__.py"
    init.write_text("")
    kernel = tmp_path / "glm_mhc_projection.bin"
    kernel.write_bytes(b"kernel")
    bridge = tmp_path / "glm_reconstruction_bridge_v1011.so"
    bridge.write_bytes(b"bridge")
    provenance = {
        "_build": {"version": 1011, "namespace": "glm_reconstruction_v1011", "helper_package": package.name},
        "_helpers": {helper.name: digest(helper), init.name: digest(init)},
        kernel.name: {"binary_sha256": digest(kernel)},
        "reconstruction_bridge.cpp": {"binary_sha256": digest(bridge)},
    }
    (tmp_path / "provenance.json").write_text(json.dumps(provenance))
    report = {
        "complete": True,
        "binary_sha256": digest(kernel),
        "helper_sha256": digest(helper),
        "cases": [dict(rows=n, finite=True, graph_changed_inputs=True) for n in (1, 16, 33, 640, 1280)],
    }
    gates = tmp_path / "gates.json"
    gates.write_text(json.dumps(report))
    return gates, report, helper


def test_complete_manifest_compiles_and_freezes_all_imported_code(tmp_path):
    gates, _, _ = fixtures(tmp_path)
    result = manifest(tmp_path, gates)
    result.verify_files()
    compile(result.value["validation_source"], "projection admission", "exec")
    assert {p["path"].split("/")[-1] for p in result.value["assets"]} == {
        "glm_mhc_projection.bin",
        "mhc_projection.py",
        "__init__.py",
        "provenance.json",
    }


@pytest.mark.parametrize("missing", ["complete", "binary_sha256", "helper_sha256"])
def test_unqualified_report_rejected(tmp_path, missing):
    gates, report, _ = fixtures(tmp_path)
    report.pop(missing)
    gates.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="matching arithmetic and replay"):
        manifest(tmp_path, gates)


def test_changed_input_replay_is_mandatory(tmp_path):
    gates, report, _ = fixtures(tmp_path)
    report["cases"][-1]["graph_changed_inputs"] = False
    gates.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="matching arithmetic and replay"):
        manifest(tmp_path, gates)


@pytest.mark.parametrize("tamper", ["helper", "initializer", "bridge"])
def test_changed_import_or_bridge_rejected_before_device_access(tmp_path, tamper):
    gates, _, helper = fixtures(tmp_path)
    path = (
        helper
        if tamper == "helper"
        else helper.with_name("__init__.py")
        if tamper == "initializer"
        else tmp_path / "glm_reconstruction_bridge_v1011.so"
    )
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="differs from provenance"):
        manifest(tmp_path, gates)
