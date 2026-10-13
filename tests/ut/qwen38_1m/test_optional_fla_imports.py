# SPDX-License-Identifier: Apache-2.0

import ast
import builtins
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

# Load this standard-library helper alone. Importing the ops package registers
# unrelated hardware kernels, which requires a built plugin even for CPU UTs.
HELPER_PATH = Path(__file__).resolve().parents[3] / "vllm_ascend/ops/fla_backend.py"
HELPER_SPEC = importlib.util.spec_from_file_location("optional_fla_backend_test", HELPER_PATH)
fla_backend = importlib.util.module_from_spec(HELPER_SPEC)
HELPER_SPEC.loader.exec_module(fla_backend)


def test_available_backend_retains_the_real_operator(monkeypatch):
    operator = object()
    monkeypatch.setattr(fla_backend, "import_module", lambda name: SimpleNamespace(real_op=operator))
    assert fla_backend.load_fla_op("real_op") is operator


def test_absent_optional_backend_imports_but_execution_fails_clearly(monkeypatch):
    def unavailable(name):
        raise ModuleNotFoundError(name="fla_npu")

    monkeypatch.setattr(fla_backend, "import_module", unavailable)
    operator = fla_backend.load_fla_op("recurrent_gated_delta_rule")
    with pytest.raises(RuntimeError, match="optional fla_npu backend"):
        operator()


def test_broken_transitive_backend_dependency_is_not_suppressed(monkeypatch):
    def broken(name):
        raise ModuleNotFoundError(name="a_required_dependency")

    monkeypatch.setattr(fla_backend, "import_module", broken)
    with pytest.raises(ModuleNotFoundError):
        fla_backend.load_fla_op("real_op")


@pytest.mark.parametrize(
    "triton, device310p, loaded", [(True, True, False), (False, False, False), (True, False, True)]
)
def test_drafter_guard_checks_hardware_even_when_triton_is_installed(monkeypatch, triton, device310p, loaded):
    path = Path(__file__).resolve().parents[3] / "vllm_ascend/spec_decode/llm_base_proposer.py"
    tree = ast.parse(path.read_text())
    guard = next(
        node
        for node in tree.body
        if isinstance(node, ast.If)
        and any(
            isinstance(child, ast.ImportFrom) and child.module == "vllm_ascend.models.deepseek_v41.dspark"
            for child in node.body
        )
    )
    imported = []
    operator = object()

    def import_drafter(name, *args, **kwargs):
        imported.append(name)
        return SimpleNamespace(DSparkDeepseekV41ForCausalLM=operator)

    namespace = {
        "HAS_TRITON": triton,
        "ascend_utils": SimpleNamespace(is_310p=lambda: device310p),
        "_HIDDEN_STATE_DRAFTER_TYPES": (),
        "__builtins__": dict(vars(builtins), __import__=import_drafter),
    }
    exec(compile(ast.Module(body=[guard], type_ignores=[]), str(path), "exec"), namespace)
    assert bool(imported) is loaded
    assert namespace["_HIDDEN_STATE_DRAFTER_TYPES"] == ((operator,) if loaded else ())
