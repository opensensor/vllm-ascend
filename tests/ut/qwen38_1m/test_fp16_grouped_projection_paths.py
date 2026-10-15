# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from tools.qwen4exp.benchmark_fp16_grouped_projection_310 import projection_functions, select_graph_arms


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("columns", [4, 16, 32])
def test_grouped_projection_includes_dynamic_casts_and_padding(dtype, columns):
    generator = torch.Generator().manual_seed(1)
    # Small integers make host accumulation exact across MM/BMM backends.
    # Random real-weight NPU equality remains a separate hardware gate.
    inputs = torch.randint(-3, 4, (3, 64), generator=generator).to(dtype)
    weight = torch.randint(-3, 4, (columns, 64), generator=generator).half()
    padded = torch.nn.functional.pad(weight, (0, 0, 0, (-columns) % 16))
    bank = padded.t().contiguous().unsqueeze(0)
    ends = torch.tensor([3], dtype=torch.int64)
    calls = []

    def grouped_matmul(*, x, weight, group_list, split_item, group_type, group_list_type):
        assert x[0].dtype == torch.float16
        assert group_list is ends and split_item == 2 and group_type == group_list_type == 0
        calls.append((x[0].clone(), weight[0].shape))
        return [x[0] @ weight[0][0]]

    functions = projection_functions(inputs, weight, bank, ends, grouped_matmul, padded if columns % 16 else None)
    assert ("gmm_original_weight_view" in functions) == (columns % 16 == 0)
    assert ("linear_padded_nz_weight" in functions) == (columns % 16 != 0)
    with torch.no_grad():
        for _ in range(3):
            inputs.copy_(torch.randint(-3, 4, inputs.shape, generator=generator).to(dtype))
            expected = functions["linear_nz_weight"]()
            for function in functions.values():
                torch.testing.assert_close(expected, function(), rtol=0, atol=0)
    assert calls and not torch.equal(calls[0][0], calls[-1][0])


@pytest.mark.parametrize("skip", [False, True])
def test_known_eager_only_arms_do_not_remove_baseline_or_gmm(skip):
    functions = {
        name: object()
        for name in (
            "linear_nz_weight",
            "gmm_cached_bank",
            "gmm_original_weight_view",
            "bmm_cached_bank",
            "matmul_broadcast_bank",
        )
    }
    original = functions.copy()
    selected = select_graph_arms(functions, skip)
    assert functions == original
    assert set(selected) == (set(functions) - {"bmm_cached_bank", "matmul_broadcast_bank"} if skip else set(functions))
    assert all(selected[name] is functions[name] for name in selected)
