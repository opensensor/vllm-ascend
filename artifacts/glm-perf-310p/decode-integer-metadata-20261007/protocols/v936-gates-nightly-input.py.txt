# SPDX-License-Identifier: Apache-2.0
"""Exact metadata division across output DMA/core boundaries and graph replay."""

import argparse
from pathlib import Path

import pytest
import torch

from tools.glm_perf.integer_divide_probe import case, load


class BuildOptions:
    def pytest_addoption(self, parser):
        parser.addoption("--integer-build-dir")


@pytest.fixture(scope="module")
def native(pytestconfig):
    path = pytestconfig.getoption("--integer-build-dir", default=None)
    if path is None:
        pytest.skip("requires an independently compiled exact metadata bundle")
    return load(Path(path).resolve(strict=True))[0]


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("divisor", [4, 160, 640])
@pytest.mark.parametrize("count", [1, 2, 8, 15, 16, 17, 63, 64, 65, 640])
def test_signed_division_replays_and_owned_dma_tail(native, dtype, divisor, count):
    result = case(native, dtype, divisor, count)
    assert result["passed"] and result["changed_input_replay"] and result["owned_padding_checked"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--integer-build-dir", required=True)
    args, extra = parser.parse_known_args()
    raise SystemExit(
        pytest.main(
            [__file__, "--noconftest", "-q", "--integer-build-dir", args.integer_build_dir, *extra],
            plugins=[BuildOptions()],
        )
    )
