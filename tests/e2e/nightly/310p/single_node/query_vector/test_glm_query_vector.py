# SPDX-License-Identifier: Apache-2.0
"""Exhaustive bit-pattern, tail ownership and changed ACL graph replay gate."""

from pathlib import Path

import pytest

from tools.glm_perf.query_bf16_vector_probe import run


def test_vector_query_replay(request, tmp_path):
    build = request.config.getoption("--glm-query-vector-build")
    if build is None:
        pytest.skip("pass --glm-query-vector-build to explicitly authorize this device gate")
    report = run(Path(build), tmp_path / "query-vector-gates.json", allow_device_gate=True)
    assert report["complete"] and len(report["records"]) == 13
