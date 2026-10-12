# SPDX-License-Identifier: Apache-2.0
"""Reproduction snapshots preserve isolation and do not overwrite evidence."""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tools.sidecard.layer_runtime import prepare, stop


def test_snapshot_excludes_environment_and_dirty_glm_experiments(tmp_path):
    source = Path(__file__).resolve().parents[3]
    output = tmp_path / "run"
    specification = {"npu_environment": "/private/runtime-environment.json"}
    configuration = prepare(output, specification, source)
    assert json.loads(configuration.read_text())["npu_environment"] == specification["npu_environment"]
    copied = list((output / "source" / "tools" / "glm_perf").glob("*"))
    assert sorted(path.name for path in copied) == [
        "__init__.py",
        "deployment_profiles.py",
        "fused_moe_profile.py",
        "glm_int4.py",
    ]
    assert not (output / "runtime-environment.json").exists()
    assert "tools/sidecard/layer_server.py" in json.loads((output / "source-sha256.json").read_text())
    assert "tools/sidecard/pcie/peer.py" in json.loads((output / "source-sha256.json").read_text())
    assert (output / "source/tools/sidecard/pcie/__init__.py").is_file()
    with pytest.raises(FileExistsError):
        prepare(output, specification, source)


def test_snapshot_rejects_output_inside_source(tmp_path):
    with pytest.raises(ValueError, match="outside"):
        prepare(tmp_path / "artifacts", {}, tmp_path)


def test_stop_rejects_a_pid_that_has_been_reused(tmp_path):
    (tmp_path / "process.json").write_text(json.dumps({"pid": os.getpid()}))
    with pytest.raises(ValueError, match="no longer belongs"):
        stop(tmp_path)


@pytest.mark.parametrize("parent_offset,return_code", [(0, 0), (1, 1)])
def test_actor_parent_guard_fails_closed_after_parent_exit(parent_offset, return_code):
    code = (
        f"import os; from tools.sidecard.layer_actor import guard_parent; guard_parent(os.getppid() + {parent_offset})"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
    assert result.returncode == return_code
    if return_code:
        assert "coordinator exited" in result.stderr


def test_snapshot_freezes_authenticated_expert_ownership(tmp_path):
    source = Path(__file__).resolve().parents[3]
    ownership = tmp_path / "ownership.json"
    ownership.write_text(
        json.dumps(
            {
                "checkpoint_sha256": "checkpoint",
                "gpu_hot_experts": {str(i): [1, 71, 146, 280] for i in range(18, 33)},
            }
        )
    )
    original = ownership.read_bytes()
    specification = {
        "architecture": "cuda-backbone-ascend-experts",
        "checkpoint_sha256": "checkpoint",
        "expert_placement": str(ownership),
        "expert_placement_sha256": hashlib.sha256(original).hexdigest(),
    }
    config = prepare(tmp_path / "frozen", specification, source)
    frozen = Path(json.loads(config.read_bytes())["expert_placement"])
    ownership.write_text("changed")
    assert frozen.read_bytes() == original
    assert frozen.parent == config.parent
    with pytest.raises(ValueError, match="authentication"):
        prepare(tmp_path / "rejected", specification, source)
    assert not (tmp_path / "rejected").exists()
