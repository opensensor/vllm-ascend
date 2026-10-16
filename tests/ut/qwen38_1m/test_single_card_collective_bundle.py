# SPDX-License-Identifier: Apache-2.0
import ast
import hashlib
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.qwen4exp import single_card_collective_bundle as module
from tools.qwen4exp.single_card_collective_bundle import communicator_source

CONFIG = """def init_ascend_config(vllm_config):
    additional_config = vllm_config.additional_config if vllm_config.additional_config is not None else {}
    excluded = {
        "enable_flashcomm1",
    }
    return {k: v for k, v in additional_config.items() if k not in excluded}
"""


ROOT = Path(__file__).resolve().parents[3]
CURRENT = (ROOT / "vllm_ascend/distributed/device_communicators/npu_communicator.py").read_text()
FROZEN = """from vllm.distributed.device_communicators.base_device_communicator import DeviceCommunicatorBase
class NPUCommunicator(DeviceCommunicatorBase):
    def __init__(self):
        self.all2all_manager = _NpuAll2AllManager()
"""


def test_collective_backport_excludes_unrelated_pipeline_transport():
    source = communicator_source(FROZEN, CURRENT)
    compile(source, "<test>", "exec")
    cls = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef))
    assert {n.name for n in cls.body if isinstance(n, ast.FunctionDef)} == {
        "__init__",
        "all_reduce",
        "close_custom_collective",
        "destroy",
    }
    assert "self.custom_sum = create(" in source
    assert "pipeline_transport" not in source
    assert "PipelineAllGatherTransport" not in source
    assert "close_custom_collective()" in source
    assert 'custom.get("execution_mode", "eager") != "graph"' in source


@pytest.mark.parametrize(
    "frozen,current",
    [(FROZEN + "# drift", CURRENT), (FROZEN, CURRENT.replace("self.custom_sum = None", "self.custom_sum = False"))],
)
def test_changed_seam_rejected(frozen, current):
    with pytest.raises(ValueError, match="seam"):
        communicator_source(frozen, current)


def test_prepared_tools_take_precedence_over_inherited_regular_package(tmp_path, monkeypatch):
    bundle, repo, output = (tmp_path / name for name in ("bundle", "repo", "experiment"))
    frozen_hashes, current_hashes = {}, {}
    for relative in module.FROZEN:
        old = (
            CONFIG
            if relative == "ascend_config.py"
            else FROZEN
            if relative.endswith("npu_communicator.py")
            else "BASE = 0\n"
        )
        new = CURRENT if relative.endswith("npu_communicator.py") else "BASE = 1\n"
        for path, source in ((bundle / "runtime/vllm_ascend" / relative, old), (repo / "vllm_ascend" / relative, new)):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(source)
        frozen_hashes[relative] = hashlib.sha256(old.encode()).hexdigest()
        if not relative.endswith("npu_communicator.py"):
            current_hashes[relative] = hashlib.sha256(new.encode()).hexdigest()
    (repo / "tools").mkdir()
    (repo / "tools/sidecard.py").write_text("VALUE = 'qualified'\n")
    foreign = tmp_path / "inherited"
    (foreign / "tools").mkdir(parents=True)
    (foreign / "tools/__init__.py").write_text("# Another controller's regular package.\n")
    monkeypatch.setattr(module, "prepare", lambda root, check: {"verified_files": 3})
    monkeypatch.setattr(module, "FROZEN", frozen_hashes)
    monkeypatch.setattr(module, "CURRENT_GRAPHS", current_hashes)
    receipt = module.prepare_collective(bundle, output, repo)
    actual = subprocess.check_output(
        [sys.executable, "-c", "from tools.sidecard import VALUE; print(VALUE)"],
        cwd=tmp_path,
        env=dict(os.environ, PYTHONPATH=os.pathsep.join((receipt["tooling"], str(foreign)))),
        text=True,
    )
    assert actual.strip() == "qualified"
    assert (bundle / "runtime/vllm_ascend/compilation/acl_graph.py").read_text() == "BASE = 0\n"
    with pytest.raises(ValueError, match="fresh destination"):
        module.prepare_collective(bundle, output, repo)


def test_collective_config_preserves_mapping_and_rejects_wrong_types():
    namespace = {}
    exec(module.config_source(CONFIG), namespace)
    validate = namespace["init_ascend_config"]
    mapping = {"npu_collective": {"execution_mode": "graph"}, "qwen_prefix_mamba_min_slots": 7}
    assert validate(SimpleNamespace(additional_config=mapping)) == {"qwen_prefix_mamba_min_slots": 7}
    assert mapping["npu_collective"] == {"execution_mode": "graph"}
    assert validate(SimpleNamespace(additional_config=None)) == {}
    for invalid in (False, [], "graph"):
        with pytest.raises(ValueError, match="dict or None"):
            validate(SimpleNamespace(additional_config={"npu_collective": invalid}))
    with pytest.raises(ValueError, match="seams"):
        module.config_source("# drift")
