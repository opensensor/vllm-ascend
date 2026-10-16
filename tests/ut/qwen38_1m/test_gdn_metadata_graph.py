# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import pytest
import torch

from tools.qwen4exp.gdn_metadata_graph import fill_uniform_metadata, uniform_decode_rows


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 6])
@pytest.mark.parametrize("length", [3, 127, 128, 129, 255, 256, 257])
def test_changing_sequence_and_acceptance_matches_scalar_mamba_mapping(n, length):
    table = torch.arange(n * 12, dtype=torch.int32).reshape(n, 12) + 100
    outputs = dict(
        spec_state_indices_tensor=torch.empty((n, 3), dtype=torch.int32),
        spec_sequence_masks=torch.empty(n, dtype=torch.bool),
        spec_query_start_loc=torch.empty(n + 1, dtype=torch.int32),
        spec_token_indx=torch.empty(n * 3, dtype=torch.int32),
        num_accepted_tokens=torch.empty(n, dtype=torch.int32),
    )
    addresses = {key: value.data_ptr() for key, value in outputs.items()}
    for step in range(3):
        lengths = torch.arange(n, dtype=torch.int32) + length + step
        counts = (torch.arange(n, dtype=torch.int32) + step) % 3 + 1
        fill_uniform_metadata(
            table,
            lengths,
            counts,
            outputs,
            block_size=128,
            boundaries=torch.arange(n + 1, dtype=torch.int32) * 3,
            tokens=torch.arange(n * 3, dtype=torch.int32),
        )
        expected = torch.stack(
            [table[i, (int(value) - 1) // 128 : (int(value) - 1) // 128 + 3] for i, value in enumerate(lengths)]
        )
        assert torch.equal(outputs["spec_state_indices_tensor"], expected)
        assert torch.equal(outputs["num_accepted_tokens"], counts)
        assert outputs["spec_sequence_masks"].all()
        assert outputs["spec_query_start_loc"].tolist() == list(range(0, n * 3 + 1, 3))
        assert outputs["spec_token_indx"].tolist() == list(range(n * 3))
        assert addresses == {key: value.data_ptr() for key, value in outputs.items()}


def admitted():
    builder = SimpleNamespace(
        use_full_cuda_graph=True, use_spec_decode=True, num_spec=2, _USE_COMMON_KERNEL_METADATA=False
    )
    common = SimpleNamespace(
        num_reqs=2,
        num_actual_tokens=6,
        max_query_len=3,
        query_start_loc_cpu=torch.tensor([0, 3, 6]),
        is_prefilling=torch.tensor([False, False]),
    )
    return builder, common, torch.ones(2, dtype=torch.int32), torch.full((2,), 2), 2


def test_uniform_layout_admission():
    assert uniform_decode_rows(*admitted())


@pytest.mark.parametrize("change", ["padded", "prefill", "draft_count", "query", "tokens", "actual", "common_kernel"])
def test_unsupported_layout_requires_original_builder(change):
    builder, common, counts, drafts, actual = admitted()
    if change == "padded":
        common.num_reqs = 3
    elif change == "prefill":
        common.is_prefilling[0] = True
    elif change == "draft_count":
        drafts[0] = 1
    elif change == "query":
        common.query_start_loc_cpu[1] = 2
    elif change == "tokens":
        common.num_actual_tokens = 5
    elif change == "actual":
        actual = 1
    else:
        builder._USE_COMMON_KERNEL_METADATA = True
    assert not uniform_decode_rows(builder, common, counts, drafts, actual)
