# SPDX-License-Identifier: Apache-2.0
import random

import pytest

from tools.qwen4exp.streaming_next_schedule import validate_prefetch


@pytest.mark.parametrize("groups", [1, 2, 3, 5, 10, 20])
def test_delayed_out_of_order_completion_preserves_all_resource_owners(groups):
    graph = validate_prefetch(groups)
    for seed in range(20):
        rng = random.Random(seed)
        pending = {c.name: c for c in graph}
        active, done = {}, set()
        while pending or active:
            ready = [c for c in pending.values() if c.dependencies <= done]
            rng.shuffle(ready)
            for c in ready:
                for prior in active.values():
                    assert not ((c.writes & (prior.reads | prior.writes)) | (prior.writes & c.reads))
                active[c.name] = c
                del pending[c.name]
            assert active, "deadlock"
            finished = rng.choice(list(active))
            done.add(finished)
            del active[finished]
        assert len(done) == 4 * groups


def test_two_metadata_slots_cannot_prefetch_before_previous_consumer():
    with pytest.raises(ValueError, match="not yet submitted"):
        validate_prefetch(3, metadata_slots=2)
