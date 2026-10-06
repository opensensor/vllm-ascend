# SPDX-License-Identifier: Apache-2.0
"""A kernel namespace must agree across its manifest and constructor."""

import hashlib
import json

import pytest

from tools.glm_perf.bf16_control import manifest


def bundle(tmp_path):
    options = {"namespace": "glm_bf16_v4", "name": "bf16_cast_v5", "bridge": "glm_bf16_bridge_v4.so"}
    (tmp_path / "options.json").write_text(json.dumps(options))
    for filename in ("gates.json", "compression-gates.json"):
        (tmp_path / filename).write_text(
            json.dumps({"complete": True, "records": [{"bitwise_equal": True, "changed_input_replay": True}]})
        )
    for filename in ("glm_bf16_bridge_v4.so", "glm_bf16_cast.bin", "bf16_cast.py", "candidate.py"):
        (tmp_path / filename).write_text(filename)
    digests = {
        name: hashlib.sha256((tmp_path / name).read_bytes()).hexdigest()
        for name in ("glm_bf16_cast.bin", "glm_bf16_bridge_v4.so", "bf16_cast.py", "candidate.py")
    }
    for name in ("gates.json", "compression-gates.json"):
        data = json.loads((tmp_path / name).read_text())
        data["binaries"] = digests
        (tmp_path / name).write_text(json.dumps(data))
    return tmp_path


def test_namespace_is_shared_by_manifest_and_kernel_constructor(tmp_path):
    result = manifest(bundle(tmp_path))
    assert result.value["operators"] == ["glm_bf16_v4::launch"]
    assert "NativeBF16Cast(root,'glm_bf16_v4')" in result.value["validation_source"]
    result.verify_files()
    (tmp_path / "candidate.py").write_text("changed")
    with pytest.raises(ValueError, match="digest mismatch"):
        result.verify_files()


def test_bundle_rejects_missing_replay_evidence(tmp_path):
    bundle(tmp_path)
    (tmp_path / "compression-gates.json").write_text(
        json.dumps({"complete": True, "records": [{"bitwise_equal": True}]})
    )
    with pytest.raises(ValueError, match="replay gates"):
        manifest(tmp_path)


def test_bundle_rejects_gate_for_different_binary(tmp_path):
    bundle(tmp_path)
    (tmp_path / "glm_bf16_cast.bin").write_bytes(b"different kernel")
    with pytest.raises(ValueError, match="exact binaries"):
        manifest(tmp_path)
