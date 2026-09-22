# SPDX-License-Identifier: Apache-2.0
"""Regression coverage for helper packages imported by runtime modules."""

import ast
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

    assert ast.literal_eval(assignment.value) == ["tools.deepseek_w2"]

    setup_call = next(
        node
        for node in ast.walk(syntax_tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "setup"
    )
    packages_value = next(keyword.value for keyword in setup_call.keywords if keyword.arg == "packages")
    assert any(isinstance(node, ast.Name) and node.id == "RUNTIME_TOOL_PACKAGES" for node in ast.walk(packages_value))
