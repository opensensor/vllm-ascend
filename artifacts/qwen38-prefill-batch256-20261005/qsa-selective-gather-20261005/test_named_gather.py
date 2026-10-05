# SPDX-License-Identifier: Apache-2.0
"""Apply the saved gather regression suite to the separately registered operator."""

import importlib.util
from functools import partial
from pathlib import Path

import pytest
import torch_npu
from benchmark_named_gather import gather, load_ops


@pytest.fixture(scope="module")
def suite():
    if not torch_npu.npu.is_available() or "310" not in torch_npu.npu.get_device_name(0):
        pytest.skip("requires an Ascend 310P NPU")
    experiment = Path(__file__).resolve().parent
    _, zero = load_ops(experiment / "binding-build/qsa_selective_probe.so")
    source = experiment / "test_gather_snapshot.py"
    spec = importlib.util.spec_from_file_location("qsa_named_regressions", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.qsa_gather_key_transposed_nz_310 = partial(gather, zero, transpose_output=True)
    module.qsa_gather_value_nz_310 = partial(gather, zero)
    return module


@pytest.mark.parametrize("num_groups,table_width", [(5, None), (5, 8), (512, None), (512, 2048)])
@pytest.mark.parametrize("transpose_output", [False, True])
def test_geometry(suite, num_groups, table_width, transpose_output):
    suite.test_gather_preserves_selected_values_with_paged_cache_and_tail(num_groups, table_width, transpose_output)


@pytest.mark.parametrize("num_groups", [5, 17, 512])
@pytest.mark.parametrize("head_dim", [16, 256])
@pytest.mark.parametrize("table_width", [None, 4096])
@pytest.mark.parametrize("transpose_output", [False, True])
def test_masked_groups(suite, num_groups, head_dim, table_width, transpose_output):
    suite.test_gather_zeroes_masked_groups_across_tile_boundaries(num_groups, head_dim, table_width, transpose_output)
