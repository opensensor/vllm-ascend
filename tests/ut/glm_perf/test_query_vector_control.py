# SPDX-License-Identifier: Apache-2.0
"""Compilation alone never admits the vector converter to resident workers."""

import hashlib
import json

import pytest

from tools.glm_perf.query_bf16_vector_probe import COUNTS
from tools.glm_perf.query_vector_control import manifest


@pytest.fixture
def evidence(tmp_path):
    asset, bridge = tmp_path / "query.bin", tmp_path / "bridge.so"
    asset.write_bytes(b"kernel fixture")
    bridge.write_bytes(b"unique namespace fixture")
    provenance = dict(
        version=954,
        namespace="glm_query_vector_v954",
        reused_bridge=False,
        assets={asset.name: hashlib.sha256(asset.read_bytes()).hexdigest()},
        bridge={"path": str(bridge), "sha256": hashlib.sha256(bridge.read_bytes()).hexdigest()},
    )
    (tmp_path / "provenance.json").write_text(json.dumps(provenance))
    gate = tmp_path / "gates.json"
    gate.write_text(
        json.dumps(
            dict(
                complete=True,
                provenance=provenance,
                records=[
                    dict(count=count, passed=True, changed_replay=True, input_guards=True, output_padding=True)
                    for count in COUNTS
                ],
            )
        )
    )
    return tmp_path, gate


def test_exact_complete_gates_pin_assets_and_bridge(evidence):
    root, gate = evidence
    result = manifest(root, gate)
    result.verify_files()
    assert result.value["operators"] == ["glm_query_vector_v954::launch"]
    assert "device=native.device.index" in result.value["validation_source"]


@pytest.mark.parametrize("field", ["passed", "changed_replay", "input_guards", "output_padding"])
def test_failed_or_incomplete_hardware_contract_is_rejected(evidence, field):
    root, gate = evidence
    report = json.loads(gate.read_text())
    report["records"][0][field] = False
    gate.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="exhaustive"):
        manifest(root, gate)


def test_compile_only_report_is_rejected(evidence):
    root, gate = evidence
    gate.write_text(json.dumps(dict(complete=False, records=[])))
    with pytest.raises(ValueError, match="matching hardware"):
        manifest(root, gate)


def test_existing_operator_namespace_cannot_be_loaded_again(evidence):
    root, gate = evidence
    path = root / "provenance.json"
    provenance = json.loads(path.read_text())
    provenance["reused_bridge"] = True
    path.write_text(json.dumps(provenance))
    with pytest.raises(ValueError, match="fresh versioned"):
        manifest(root, gate)
