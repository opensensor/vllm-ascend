# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import numpy as np
import pytest

from tools.qwen4exp.iteration_transition_runtime import admit_all_acceptances
from tools.qwen4exp.runtime_geometry import request_capacity, state_geometry


@pytest.mark.parametrize("width,heads", [(2560, 12), (1920, 9), (1280, 6)])
@pytest.mark.parametrize("slots", [64, 105, 177])
def test_state_slots_are_independent_of_parallel_shard(width, heads, slots):
    assert (
        state_geometry(SimpleNamespace(shape=(slots, 5, width)), SimpleNamespace(shape=(slots, heads, 128, 128)))
        == slots
    )


@pytest.mark.parametrize(
    "conv,state",
    [
        ((105, 5, 1920), (105, 6, 128, 128)),
        ((105, 5, 1920), (64, 9, 128, 128)),
        ((105, 4, 1920), (105, 9, 128, 128)),
        ((0, 5, 1920), (0, 9, 128, 128)),
        ((105, 5, 1280), (105, 6, 64, 128)),
    ],
)
def test_reject_mismatched_pair_before_capture(conv, state):
    with pytest.raises(ValueError, match="paired QFN"):
        state_geometry(SimpleNamespace(shape=conv), SimpleNamespace(shape=state))


@pytest.mark.parametrize("count", [1, 4, 5, 6])
def test_capture_matches_configured_request_capacity(count):
    assert request_capacity(SimpleNamespace(scheduler_config=SimpleNamespace(max_num_seqs=count))) == count


@pytest.mark.parametrize("count", [0, 7, True, 6.0])
def test_reject_unqualified_request_capacity(count):
    with pytest.raises(ValueError, match="six request"):
        request_capacity(SimpleNamespace(scheduler_config=SimpleNamespace(max_num_seqs=count)))


def test_six_request_boundary_admission_checks_high_slots_and_every_row():
    columns = [np.full(6, value, dtype=np.int32) for value in (0, 126, 3, 2)]
    tables = {1: np.tile(np.array([102, 103, 104], dtype=np.int32), (6, 1))}
    packed = admit_all_acceptances(columns, tables, block_size=128, slots=105)
    assert packed.shape == (5, 8)
    assert (packed[1, :6] == 126).all()
    assert not packed[:, 6:].any()
    tables[1][5, 1] = 105
    with pytest.raises(ValueError, match="invalid block ID"):
        admit_all_acceptances(columns, tables, block_size=128, slots=105)
