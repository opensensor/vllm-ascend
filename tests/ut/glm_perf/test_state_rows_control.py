# SPDX-License-Identifier: Apache-2.0
"""State-copy manifests require complete gates and immutable helper/binary hashes."""

import hashlib
import json

import pytest

from tools.glm_perf.state_rows_control import manifest


@pytest.fixture
def evidence(tmp_path):
    package = tmp_path / "glm_reconstruction_v9011_helpers"
    package.mkdir()
    helper = package / "state_rows.py"
    helper.write_text("# frozen helper\n")
    options = {
        "namespace": "glm_reconstruction_v9011",
        "version": 9011,
        "helper_package": package.name,
        "selected_state_rows": True,
    }
    paths = [
        tmp_path / "glm_reconstruction_bridge_v9011.so",
        tmp_path / "glm_state_rows_gather.bin",
        tmp_path / "glm_state_rows_scatter.bin",
    ]
    for path in paths:
        path.write_bytes(b"isolated manifest fixture")
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    provenance = {"_build": options, "_helpers": {helper.name: hashlib.sha256(helper.read_bytes()).hexdigest()}}
    (tmp_path / "provenance.json").write_text(json.dumps(provenance))
    records = [
        {
            "payload_shape": shape,
            "gap": gap,
            "selected": selected,
            "index_bytes": width,
            "passed": True,
            "changed_replay": True,
            "all_backing_bytes_checked": True,
            "negative_slots": True,
            "fresh_flags": True,
        }
        for shape in ((2, 16, 16), (16, 128, 128))
        for gap in (0, 192)
        for selected in (1, 4)
        for width in (4, 8)
    ]
    gate = tmp_path / "gates.json"
    gate.write_text(json.dumps({"complete": True, "build_options": options, "records": records, "binaries": hashes}))
    return tmp_path, gate, helper, paths


def test_complete_manifest_verifies_every_asset(evidence):
    root, gate, _, _ = evidence
    value = manifest(root, gate)
    value.verify_files()
    assert value.name == "state_rows_v9011"
    assert value.value["operators"] == ["glm_reconstruction_v9011::launch"]


@pytest.mark.parametrize("field", ["changed_replay", "all_backing_bytes_checked", "negative_slots", "fresh_flags"])
def test_incomplete_graph_or_backing_contract_rejected(evidence, field):
    root, gate, _, _ = evidence
    report = json.loads(gate.read_text())
    report["records"][0][field] = False
    gate.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="changed graph replay"):
        manifest(root, gate)


def test_changed_helper_is_rejected(evidence):
    root, gate, helper, _ = evidence
    helper.write_text("# changed helper\n")
    with pytest.raises(ValueError, match="helper changed"):
        manifest(root, gate)


def test_changed_binary_is_rejected(evidence):
    root, gate, _, paths = evidence
    paths[1].write_bytes(b"different kernel")
    with pytest.raises(ValueError, match="exact binaries"):
        manifest(root, gate)
