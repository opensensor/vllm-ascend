# SPDX-License-Identifier: Apache-2.0
"""Validate the private graph gate's independently calculated reference."""

from tools.qwen4exp.benchmark_mamba_commit_plan_310 import assert_plan, cases, scalar_reference
from tools.qwen4exp.mamba_commit_plan import plan_aligned_commit


def test_all_changing_cases_match_scalar_contract():
    changed_masks = set()
    changed_counts = set()
    for inputs in cases(128):
        expected = scalar_reference(inputs, 128)
        actual = plan_aligned_commit(*inputs, block_size=128)
        assert_plan(actual, expected)
        changed_masks.add(tuple(actual.copy_required.tolist()))
        changed_counts.add(tuple(actual.next_accepted_counts.tolist()))
    assert len(changed_masks) >= 3
    assert len(changed_counts) >= 4
