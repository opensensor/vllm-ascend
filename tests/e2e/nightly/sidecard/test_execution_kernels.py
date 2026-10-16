# SPDX-License-Identifier: Apache-2.0
"""Explicit exclusive-lane component and teacher-forced model qualification."""

from pathlib import Path

import pytest


def test_execution_kernels_and_model(request):
    manifest = request.config.getoption("--sidecard-execution-manifest")
    if manifest is None:
        pytest.skip("requires an explicit authenticated execution manifest and exclusive accelerator ownership")
    from tools.sidecard.execution_nightly import run

    assert run(Path(manifest))["passed"]
