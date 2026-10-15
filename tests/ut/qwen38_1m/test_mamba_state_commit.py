# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

from tools.qwen4exp.benchmark_mamba_commit_group_310 import host_group_copy
from tools.qwen4exp.benchmark_mamba_state_commit_310 import cases, host_driven_copy
from tools.qwen4exp.mamba_commit import geometry, group_geometry, reference


def inputs(source=0, computed=126, accepted=3):
    values = torch.zeros((5, 8), dtype=torch.int32)
    values[:, 0] = torch.tensor([source, computed, 3, 2, accepted])
    return values


def test_overlap_preserves_suffix_and_untouched_tail():
    state = torch.arange(4 * 5 * 16, dtype=torch.int32).reshape(4, 5, 16)
    table = torch.arange(4, dtype=torch.int32).reshape(1, 4)
    result, receipt = reference(state, inputs(), table, conv=True, block_size=128)
    assert torch.equal(result[0, :4], state[0, 1:])
    assert torch.equal(result[0, 4], state[0, 4])
    assert torch.equal(result[1:], state[1:])
    assert receipt[0, 0] == 1 and not receipt[1].any()


def test_temporal_copy_uses_accepted_block_offset():
    state = torch.arange(4 * 32, dtype=torch.int32).reshape(4, 32)
    table = torch.tensor([[2, 3, 0, 1]], dtype=torch.int32)
    result, receipt = reference(state, inputs(), table, conv=False, block_size=128)
    assert torch.equal(result[2], state[3])
    assert torch.equal(result[[0, 1, 3]], state[[0, 1, 3]])
    assert not receipt[1].any()


def test_inactive_copy_never_uses_invalid_block_table():
    state = torch.ones(4, 32, dtype=torch.int32)
    table = torch.full((1, 4), -1, dtype=torch.int32)
    result, receipt = reference(state, inputs(computed=10), table, conv=False, block_size=128)
    assert torch.equal(result, state) and not receipt[1].any()


@pytest.mark.parametrize(
    "error,values,table",
    [
        (1, inputs(accepted=4), [[0, 1, 2, 3]]),
        (2, inputs(source=3), [[0, 1, 2, 3]]),
        (3, inputs(), [[0, -1, 2, 3]]),
    ],
)
def test_invalid_rows_report_and_do_not_write(error, values, table):
    state = torch.arange(4 * 32, dtype=torch.int32).reshape(4, 32)
    result, receipt = reference(state, values, torch.tensor(table, dtype=torch.int32), conv=False, block_size=128)
    assert receipt[1, 0] == error and torch.equal(result, state)


def test_reject_unaligned_state_rows():
    with pytest.raises(ValueError, match="32 bytes"):
        geometry(torch.zeros(4, 3, 10), torch.zeros(1, 4, dtype=torch.int32), conv=True, block_size=128)


@pytest.mark.parametrize("conv", [False, True])
def test_benchmark_baseline_matches_oracle(conv):
    shape = (24, 5, 16) if conv else (24, 32)
    state = torch.arange(24 * 5 * 16 if conv else 24 * 32, dtype=torch.int32).reshape(shape)
    for values, table in list(cases())[:16]:
        expected, receipt = reference(state, values, table, conv=conv, block_size=128)
        assert not receipt[1].any()
        actual = state.clone()
        host_driven_copy(actual, values, table, conv=conv)
        assert torch.equal(actual, expected)


def test_parallel_banks_reject_aliases_but_allow_disjoint_views():
    storage = torch.zeros(8, 32)
    table = torch.zeros(1, 4, dtype=torch.int32)
    group_geometry([storage[:4], storage[4:]], [table, table], [False, False], block_size=128)
    with pytest.raises(ValueError, match="overlap"):
        group_geometry([storage[:4], storage[3:]], [table, table], [False, False], block_size=128)
    with pytest.raises(ValueError, match="alias"):
        group_geometry([storage], [storage.view(torch.int32)[:1, :4].contiguous()], [False], block_size=128)


def test_group_baseline_decides_once_and_skips_all_banks_when_idle():
    values, table = list(cases())[3]
    values[1, :4] = 10
    # Any bank access would fail: no-copy path must inspect request metadata only.
    host_group_copy([None] * 72, [True, False] * 36, values, table)


def test_group_baseline_matches_independent_bank_reference():
    banks = [
        torch.arange(24 * 5 * 16, dtype=torch.int32).reshape(24, 5, 16),
        torch.arange(24 * 32, dtype=torch.int32).reshape(24, 32),
    ]
    values, table = list(cases())[3]
    expected = [
        reference(bank, values, table, conv=conv, block_size=128)[0] for bank, conv in zip(banks, [True, False])
    ]
    host_group_copy(banks, [True, False], values, table)
    assert all(torch.equal(left, right) for left, right in zip(banks, expected))
