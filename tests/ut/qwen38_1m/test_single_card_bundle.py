# SPDX-License-Identifier: Apache-2.0
"""Admission fails on missing, modified or escaped snapshot assets."""

import hashlib
import json

import pytest

from tools.qwen4exp.verify_single_card_bundle import verify_bundle


def snapshot(root):
    content = b"qualified source"
    (root / "model.py").write_bytes(content)
    manifest = {"model.py": hashlib.sha256(content).hexdigest()}
    (root / "files.sha256.json").write_text(json.dumps(manifest))
    return manifest


def test_exact_snapshot(tmp_path):
    snapshot(tmp_path)
    assert verify_bundle(tmp_path) == 1


@pytest.mark.parametrize("missing", [False, True])
def test_bad_asset(tmp_path, missing):
    snapshot(tmp_path)
    if missing:
        (tmp_path / "model.py").unlink()
    else:
        (tmp_path / "model.py").write_text("different runtime")
    with pytest.raises(ValueError, match="missing|mismatch"):
        verify_bundle(tmp_path)


@pytest.mark.parametrize("name", ["../outside.py", "/tmp/outside.py"])
def test_path_escape(tmp_path, name):
    (tmp_path / "files.sha256.json").write_text(json.dumps({name: "a" * 64}))
    with pytest.raises(ValueError, match="escapes"):
        verify_bundle(tmp_path)


def test_linked_asset(tmp_path):
    manifest = snapshot(tmp_path)
    (tmp_path / "copy.py").write_bytes((tmp_path / "model.py").read_bytes())
    (tmp_path / "model.py").unlink()
    (tmp_path / "model.py").symlink_to("copy.py")
    (tmp_path / "files.sha256.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="linked"):
        verify_bundle(tmp_path)
