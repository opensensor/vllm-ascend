# SPDX-License-Identifier: Apache-2.0
"""Regression coverage for helper packages imported by runtime modules."""

import ast
import subprocess
import sys
from pathlib import Path

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def test_runtime_tool_packages_include_deepseek_w2():
    syntax_tree = ast.parse((_REPOSITORY_ROOT / "setup.py").read_text(encoding="utf-8"))
    assignment = next(
        node
        for node in syntax_tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "RUNTIME_TOOL_PACKAGES" for target in node.targets)
    )

    assert ast.literal_eval(assignment.value) == ["tools.deepseek_w2", "tools.glm_perf", "tools.glm_w2"]

    setup_call = next(
        node
        for node in ast.walk(syntax_tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "setup"
    )
    packages_value = next(keyword.value for keyword in setup_call.keywords if keyword.arg == "packages")
    assert any(isinstance(node, ast.Name) and node.id == "RUNTIME_TOOL_PACKAGES" for node in ast.walk(packages_value))


def test_glm_runtime_helpers_import_from_built_package_outside_checkout(tmp_path):
    from setuptools import Distribution
    from setuptools.command.build_py import build_py

    output = tmp_path / "wheel-content"
    distribution = Distribution(
        {
            "packages": ["tools.deepseek_w2", "tools.glm_perf", "tools.glm_w2"],
            "package_dir": {"": str(_REPOSITORY_ROOT)},
        }
    )
    distribution.script_name = str(_REPOSITORY_ROOT / "setup.py")
    command = build_py(distribution)
    command.build_lib = str(output)
    command.ensure_finalized()
    command.run()
    code = (
        "import sys; from pathlib import Path; sys.path.insert(0, sys.argv[1]); "
        "import tools.glm_perf.native_checkpoint as native; "
        "import tools.glm_perf.deployment_profiles as profiles; "
        "import tools.glm_w2.convert_full as converter; "
        "assert profiles.profile(3).chips == 6; "
        "assert all(Path(m.__file__).is_relative_to(sys.argv[1]) for m in (native, profiles, converter))"
    )
    result = subprocess.run([sys.executable, "-c", code, str(output)], cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
