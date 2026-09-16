# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the 310P environment freeze / revision recorder (plan T0.2).

The recorder is exercised entirely with injected fake collectors so no NPU,
torch-npu, CANN, or real package metadata is required.
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).parents[3]
_MODULE_PATH = _REPO_ROOT / "tools" / "qwen38_1m" / "env_freeze.py"


def _load_env_freeze():
    spec = importlib.util.spec_from_file_location("qwen38_env_freeze", _MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Register before exec so dataclass annotation resolution can find the module.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


env_freeze = _load_env_freeze()


def _fake_revisions():
    """A full set of injected component revisions (all required keys present)."""
    R = env_freeze.ComponentRevision
    return {
        "vllm": R(name="vllm", version="0.11.0", path="/vllm-workspace/vllm", source="fake"),
        "vllm_ascend": R(
            name="vllm_ascend",
            version="0.11.0.dev",
            path="/vllm-workspace/vllm-ascend",
            source="fake",
        ),
        "torch_npu": R(name="torch_npu", version="2.5.1", path=None, source="fake"),
        "cann": R(name="cann", version="8.0.RC3", path="/usr/local/Ascend", source="fake"),
        "transformers": R(name="transformers", version="4.51.0", path=None, source="fake"),
        "tokenizers": R(name="tokenizers", version="0.21.0", path=None, source="fake"),
        "modelslim": R(name="modelslim", version="master-abc123", path=None, source="fake"),
    }


def test_freeze_emits_all_required_keys():
    report = env_freeze.freeze(
        revision_collector=_fake_revisions,
        checkpoint_hash_collector=lambda: "sha256:deadbeef",
        now=lambda: 100.0,
    )
    assert report.schema_version == env_freeze.SCHEMA_VERSION
    assert report.timestamp == 100.0
    # Every required component key is present.
    for key in env_freeze.REQUIRED_COMPONENTS:
        assert key in report.components, key
        assert report.components[key].version is not None
    # checkpoint_hash is a required top-level key.
    assert report.checkpoint_hash == "sha256:deadbeef"

    payload = report.to_dict()
    for key in env_freeze.REQUIRED_COMPONENTS:
        assert key in payload["components"]
    assert "checkpoint_hash" in payload
    # Explicit spec-mandated keys.
    for key in (
        "vllm",
        "vllm_ascend",
        "torch_npu",
        "cann",
        "transformers",
        "tokenizers",
        "modelslim",
    ):
        assert key in payload["components"]


def test_freeze_raises_when_required_component_missing():
    def missing_modelslim():
        revs = _fake_revisions()
        del revs["modelslim"]
        return revs

    with pytest.raises(ValueError, match="modelslim"):
        env_freeze.freeze(revision_collector=missing_modelslim, now=lambda: 1.0)


def test_freeze_allows_partial_when_require_all_false():
    def only_vllm():
        return {"vllm": env_freeze.ComponentRevision(name="vllm", version="0.11.0")}

    report = env_freeze.freeze(revision_collector=only_vllm, require_all=False, now=lambda: 1.0)
    assert "vllm" in report.components
    assert report.checkpoint_hash is None


def test_import_check_passes_on_expected_paths():
    module_paths = {
        "vllm": "/vllm-workspace/vllm/__init__.py",
        "vllm_ascend": "/vllm-workspace/vllm-ascend/vllm_ascend/__init__.py",
    }
    # Should not raise.
    env_freeze.import_check(module_paths)


def test_import_check_raises_on_wrong_vllm_path():
    module_paths = {
        "vllm": "/usr/lib/python3.14/site-packages/vllm/__init__.py",
        "vllm_ascend": "/vllm-workspace/vllm-ascend/vllm_ascend/__init__.py",
    }
    with pytest.raises(env_freeze.ImportCheckError, match="vllm"):
        env_freeze.import_check(module_paths)


def test_import_check_raises_on_wrong_vllm_ascend_path():
    module_paths = {
        "vllm": "/vllm-workspace/vllm/__init__.py",
        "vllm_ascend": "/opt/vllm-ascend/__init__.py",
    }
    with pytest.raises(env_freeze.ImportCheckError, match="vllm_ascend"):
        env_freeze.import_check(module_paths)


def test_import_check_raises_on_missing_module():
    with pytest.raises(env_freeze.ImportCheckError, match="vllm"):
        env_freeze.import_check({"vllm_ascend": "/vllm-workspace/vllm-ascend/__init__.py"})


def test_import_check_prefixes_are_parameterizable():
    module_paths = {
        "vllm": "/custom/root/vllm/__init__.py",
        "vllm_ascend": "/custom/root/vllm-ascend/__init__.py",
    }
    # Wrong under the default, but accepted with overridden prefixes.
    with pytest.raises(env_freeze.ImportCheckError):
        env_freeze.import_check(module_paths)
    env_freeze.import_check(
        module_paths,
        expected_vllm_prefix="/custom/root/vllm",
        expected_vllm_ascend_prefix="/custom/root/vllm-ascend",
    )


def test_json_round_trip():
    original = env_freeze.freeze(
        revision_collector=_fake_revisions,
        checkpoint_hash_collector=lambda: "sha256:cafef00d",
        now=lambda: 42.0,
    )
    restored = env_freeze.EnvFreezeReport.from_dict(json.loads(original.to_json()))
    assert restored.to_dict() == original.to_dict()
    assert restored.timestamp == 42.0
    assert restored.checkpoint_hash == "sha256:cafef00d"
    assert restored.components["vllm"].path == "/vllm-workspace/vllm"


def test_human_summary_reports_components_and_hash():
    report = env_freeze.freeze(
        revision_collector=_fake_revisions,
        checkpoint_hash_collector=lambda: "sha256:deadbeef",
        now=lambda: 1.0,
    )
    summary = report.human_summary()
    assert "vllm" in summary
    assert "0.11.0" in summary
    assert "sha256:deadbeef" in summary
    assert "torch_npu" in summary


def test_module_import_does_not_import_torch_npu():
    # Loading the module must not have imported torch_npu (host has no NPU).
    assert "torch_npu" not in sys.modules
