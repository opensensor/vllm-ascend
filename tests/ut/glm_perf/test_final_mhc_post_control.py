# SPDX-License-Identifier: Apache-2.0
"""Final mixer admission pins the actual unrounded binaries and replay gates."""

import json

import pytest

from tools.glm_perf.mhc_final_post_control import manifest
from tools.glm_perf.resident_native import file_digest


def fixture(tmp_path):
    for name in ("mhc_post_native.py", "mhc_post_fp16.bin", "mhc_post_fp32.bin", "glm_reconstruction_bridge_v1012.so"):
        (tmp_path / name).write_bytes(name.encode())
    provenance = {
        "version": 1012,
        "namespace": "glm_reconstruction_v1012",
        "state_rounding": "none_fp32",
        "finish_only": False,
        "helper_sha256": file_digest(tmp_path / "mhc_post_native.py"),
        "binaries": {
            f"mhc_post_fp{bits}.bin": {"sha256": file_digest(tmp_path / f"mhc_post_fp{bits}.bin")} for bits in (16, 32)
        },
    }
    bridge_provenance = {
        "_build": {"namespace": "glm_reconstruction_v1012"},
        "reconstruction_bridge.cpp": {"binary_sha256": file_digest(tmp_path / "glm_reconstruction_bridge_v1012.so")},
    }
    (tmp_path / "mhc-provenance.json").write_text(json.dumps(provenance))
    (tmp_path / "provenance.json").write_text(json.dumps(bridge_provenance))
    report = {
        "complete": True,
        "provenance": provenance,
        "helper_sha256": provenance["helper_sha256"],
        "cases": [
            dict(rows=rows, input_dtype=dtype, finite=True, graph_changed_inputs=True, fp32_output_unrounded=True)
            for rows in (640, 1280)
            for dtype in ("torch.float16", "torch.float32")
        ],
    }
    path = tmp_path / "gates.json"
    path.write_text(json.dumps(report))
    return path, report


def test_manifest_freezes_all_loaded_code_and_compiles_validation(tmp_path):
    gates, _ = fixture(tmp_path)
    result = manifest(tmp_path, gates)
    result.verify_files()
    compile(result.value["validation_source"], "final mixer validation", "exec")
    assert len(result.value["assets"]) == 5


@pytest.mark.parametrize("field", ["finite", "graph_changed_inputs", "fp32_output_unrounded"])
def test_gate_failure_blocks_admission(tmp_path, field):
    gates, report = fixture(tmp_path)
    report["cases"][0][field] = False
    gates.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="complete FP32 arithmetic"):
        manifest(tmp_path, gates)


@pytest.mark.parametrize("name", ["mhc_post_native.py", "mhc_post_fp32.bin", "glm_reconstruction_bridge_v1012.so"])
def test_tampered_assets_rejected_before_device_mutation(tmp_path, name):
    gates, _ = fixture(tmp_path)
    (tmp_path / name).write_bytes(b"changed")
    with pytest.raises(ValueError, match="differs from qualified provenance"):
        manifest(tmp_path, gates)
