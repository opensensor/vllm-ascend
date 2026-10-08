# SPDX-License-Identifier: Apache-2.0
"""Hardware replay/parity gates for an explicitly supplied append-only build."""

from pathlib import Path

import pytest
import torch

from tools.glm_perf.qsa_metadata_probe import case, load


@pytest.fixture(scope="module")
def native(request):
    build = request.config.getoption("--glm-qsa-metadata-build")
    if build is None:
        pytest.skip("pass --glm-qsa-metadata-build identifying a qualified 310P build")
    return load(Path(build))[0]


@pytest.mark.parametrize("rows", [2, 8, 640])
@pytest.mark.parametrize("requests", [1, 4])
@pytest.mark.parametrize("budget", [4, 512])
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("split", [1, 20])
def test_signed_strided_metadata_replay(native, rows, requests, budget, dtype, split):
    assert case(native, rows, requests, budget, dtype, split)["passed"]
