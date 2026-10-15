# SPDX-License-Identifier: Apache-2.0

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


def load_module():
    path = Path(__file__).resolve().parents[3] / "tools/qwen4exp/ple_arm_udf/prepare_bootstrap.py"
    spec = importlib.util.spec_from_file_location("ple_udf_bootstrap", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_bootstrap_preserves_runtime_and_vendor_bytes(tmp_path):
    module = load_module()
    runtime = tmp_path / "runtime"
    (runtime / "conf").mkdir(parents=True)
    (runtime / "compat").mkdir()
    # A container fixture tests byte preservation, not CMS signature validity.
    vendor_bytes = bytes.fromhex("aa55aa55") + b"unit-test-container"
    (runtime / module.VENDOR_ARCHIVE).write_bytes(vendor_bytes)
    original = "name:vendor.tar.gz\ninstall_path:0\n"
    (runtime / "conf/ascend_package_load.ini").write_text(original)
    (runtime / "lib64").mkdir()
    (runtime / "conf/unrelated.ini").write_text("unchanged")
    overlay = tmp_path / "overlay"
    package = module.prepare(runtime, overlay)
    assert (runtime / "conf/ascend_package_load.ini").read_text() == original
    assert (overlay / "lib64").resolve() == runtime / "lib64"
    assert (overlay / "conf/unrelated.ini").resolve() == runtime / "conf/unrelated.ini"
    assert "install_path:1" in (overlay / "conf/ascend_package_load.ini").read_text()
    assert package.read_bytes() == vendor_bytes
    manifest = json.loads((overlay / "ple-bootstrap-manifest.json").read_text())
    assert manifest["sha256"] == hashlib.sha256(vendor_bytes).hexdigest()
    assert manifest["device_signature_verified"] is False
    assert manifest["device_deployment_verified"] is False
    assert manifest["device_submit"] is False
    with pytest.raises(FileExistsError):
        module.prepare(runtime, overlay)


def test_bootstrap_rejects_already_modified_config(tmp_path):
    module = load_module()
    (tmp_path / "conf").mkdir()
    (tmp_path / "conf/ascend_package_load.ini").write_text(f"name:{module.PACKAGE_NAME}\n")
    with pytest.raises(ValueError, match="already"):
        module.prepare(tmp_path, tmp_path / "overlay")
    assert not (tmp_path / "overlay").exists()


def test_bootstrap_rejects_unsigned_archive(tmp_path):
    module = load_module()
    (tmp_path / "conf").mkdir()
    (tmp_path / "compat").mkdir()
    (tmp_path / "conf/ascend_package_load.ini").write_text("name:vendor.tar.gz\n")
    (tmp_path / module.VENDOR_ARCHIVE).write_bytes(b"unsigned marker")
    with pytest.raises(ValueError, match="CMS archive"):
        module.prepare(tmp_path, tmp_path / "overlay")
    assert not (tmp_path / "overlay").exists()
