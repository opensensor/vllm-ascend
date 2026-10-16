# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import pytest

from vllm_ascend.core.prefix_mamba_leases import PrefixMambaStepLeases


class Pool:
    def __init__(self):
        self.blocks = [SimpleNamespace(block_id=i, is_null=i == 0, ref_cnt=1) for i in range(8)]

    def touch(self, blocks):
        for block in blocks:
            block.ref_cnt += 1

    def free_blocks(self, blocks):
        for block in blocks:
            block.ref_cnt -= 1
            assert block.ref_cnt >= 0


def test_aborted_request_does_not_retire_submitted_state():
    pool = Pool()
    leases = PrefixMambaStepLeases(pool, 2)
    first, second = object(), object()
    leases.acquire(first, {1: [1, 2], 2: [3]})
    leases.acquire(second, {1: [2, 4]})
    pool.free_blocks([pool.blocks[2]])  # request cancellation
    assert pool.blocks[2].ref_cnt == 2
    assert leases.retained(1) == {1, 2, 4}
    with pytest.raises(RuntimeError, match="in flight"):
        leases.acquire(object(), {1: [5]})
    leases.release(first)
    assert pool.blocks[2].ref_cnt == 1 and leases.retained(1) == {2, 4}
    assert not leases.drained
    leases.release(second)
    assert pool.blocks[2].ref_cnt == 0 and leases.drained
    assert not leases.retained(1)


def test_union_deduplicates_shared_blocks_without_conflating_outputs():
    pool = Pool()
    leases = PrefixMambaStepLeases(pool, 2)
    output = object()
    leases.acquire(output, {1: [1, 1], 2: [1]})
    assert pool.blocks[1].ref_cnt == 2
    with pytest.raises(RuntimeError, match="already owns"):
        leases.acquire(output, {1: [2]})
    with pytest.raises(RuntimeError, match="exact completed"):
        leases.release(object())
    assert pool.blocks[1].ref_cnt == 2
    leases.release(output)
    assert pool.blocks[1].ref_cnt == 1


@pytest.mark.parametrize("invalid", [0, -1, 8])
def test_invalid_lease_is_rejected_before_any_pool_changes(invalid):
    pool = Pool()
    leases = PrefixMambaStepLeases(pool, 2)
    with pytest.raises(ValueError):
        leases.acquire(object(), {1: [1, invalid]})
    assert leases.drained and all(block.ref_cnt == 1 for block in pool.blocks)


def test_partial_pool_failure_is_quarantined():
    pool = Pool()
    leases = PrefixMambaStepLeases(pool, 2)

    def fail(blocks):
        pool.blocks[1].ref_cnt += 1
        raise RuntimeError("partial pin")

    pool.touch = fail
    output = object()
    with pytest.raises(RuntimeError, match="partial pin"):
        leases.acquire(output, {1: [1, 2]})
    assert leases.failed and not leases.drained
    with pytest.raises(RuntimeError, match="exact completed"):
        leases.release(output)
    assert pool.blocks[1].ref_cnt == 2
