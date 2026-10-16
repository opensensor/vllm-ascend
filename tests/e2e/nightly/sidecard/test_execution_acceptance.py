# SPDX-License-Identifier: Apache-2.0
"""Explicit opt-in completed generation tests; defaults never allocate devices."""

import pytest


@pytest.mark.parametrize("tier", ["quality", "context", "stability"])
def test_execution_acceptance(request, tier):
    base_url = request.config.getoption("--sidecard-acceptance-url", default=None)
    fixtures = request.config.getoption("--sidecard-acceptance-fixtures", default=None)
    output = request.config.getoption("--sidecard-acceptance-output", default=None)
    if not all((base_url, fixtures, output)):
        pytest.skip("requires explicit URL/private fixture/output inputs and exclusive hardware owner")
    from pathlib import Path

    from tools.sidecard.execution_acceptance import run

    report = run(base_url, fixtures, Path(output) / f"acceptance-{tier}.json", tier)
    assert report["passed"], report.get("pending")
    assert report["hardware_validated"]
