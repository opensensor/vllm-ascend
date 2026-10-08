# SPDX-License-Identifier: Apache-2.0
"""Selected-state replay with the serving cache's 327680-element page stride."""

import argparse
import importlib
from pathlib import Path

import pytest
import torch

from tools.glm_perf.state_rows_probe import load_helper


class BuildOptions:
    def pytest_addoption(self, parser):
        parser.addoption("--state-rows-build-dir")


@pytest.fixture(scope="module")
def bundle(pytestconfig):
    path = pytestconfig.getoption("--state-rows-build-dir", default=None)
    if path is None:
        pytest.skip("requires an independently compiled selected-state-copy bundle")
    native, provenance = load_helper(Path(path).resolve(strict=True))
    probe = importlib.import_module(provenance["_build"]["helper_package"] + ".state_rows_probe")
    return native, probe


@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("selected", [1, 4])
def test_serving_page_gap_changed_replay(bundle, index_dtype, selected):
    native, probe = bundle
    record = probe.case(native, (16, 128, 128), gap=65536, selected=selected, index_dtype=index_dtype)
    assert record["passed"] and record["changed_replay"] and record["all_backing_bytes_checked"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-rows-build-dir", required=True)
    args, extra = parser.parse_known_args()
    raise SystemExit(
        pytest.main(
            [__file__, "--noconftest", "-q", "--state-rows-build-dir", args.state_rows_build_dir, *extra],
            plugins=[BuildOptions()],
        )
    )
