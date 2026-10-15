# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

from tools.qwen4exp.iteration_transition import accept_step, advance_ple_history, prepare_next_step


def metadata(computed=(10, 126, 255, 262140)):
    values = torch.zeros((5, 8), dtype=torch.int32)
    values[0, :4] = torch.tensor([0, 0, 1, 2046])
    values[1, :4] = torch.tensor(computed)
    values[2, :4] = 3
    values[3, :4] = 2
    return values


@pytest.mark.parametrize("count", range(4))
def test_acceptance_and_next_input_match_scalar_reference(count):
    inputs = metadata()
    samples = torch.tensor([[7, 8, 9]] * 4, dtype=torch.int64)
    samples[:, count:] = -1
    drafts = torch.tensor([[10, 11]] * 4, dtype=torch.int64)
    accepted = accept_step(samples, inputs, block_size=128)
    next_step = prepare_next_step(samples, drafts, inputs, accepted, max_model_len=262144)
    for row in range(4):
        source, computed, scheduled, drafted, raw_count = inputs[:, row].tolist()
        running = computed + scheduled - drafted
        aligned = (running + count - 1) // 128 * 128
        next_count = 1 if aligned >= running and source == aligned // 128 - 1 else count
        assert raw_count == count and accepted.next_counts[row] == next_count
        assert accepted.rejected[row] == 3 - count and not accepted.error[row]
        new_computed = computed + count
        active = count > 0 and new_computed + 3 <= 262144
        assert bool(next_step.active[row]) == active
        assert next_step.input_ids[row].tolist() == ([6 + count, 10, 11] if active else [0, 0, 0])
        assert next_step.positions[row].tolist() == (
            list(range(new_computed, new_computed + 3)) if active else [0, 0, 0]
        )


def test_mamba_normalization_does_not_change_position_advance():
    inputs = metadata()
    samples = torch.tensor([[7, 8, 9]] * 4)
    accepted = accept_step(samples, inputs, block_size=128)
    assert accepted.next_counts[1] == 1 and accepted.counts[1] == 3
    output = prepare_next_step(samples, torch.ones(4, 2, dtype=torch.int64), inputs, accepted, max_model_len=262144)
    assert output.computed[1] == 129


@pytest.mark.parametrize("tokens", [[-1, 8, -1], [7, -1, 9], [7, -2, -1]])
def test_malformed_sampler_output_cannot_continue(tokens):
    inputs = metadata()
    samples = torch.tensor([tokens] * 4)
    accepted = accept_step(samples, inputs, block_size=128)
    assert accepted.error.all() and accepted.counts.eq(0).all()
    output = prepare_next_step(samples, torch.ones(4, 2, dtype=torch.int64), inputs, accepted, max_model_len=262144)
    assert not output.active.any() and output.input_ids.eq(0).all()


def test_eos_stops_only_rows_with_accepted_eos_and_invalid_draft_gates_row():
    inputs = metadata(computed=(10, 10, 10, 10))
    samples = torch.tensor([[7, 8, -1], [7, -1, -1], [8, 9, 10], [7, 9, -1]])
    drafts = torch.tensor([[11, 12], [8, 12], [11, 12], [-1, 12]])
    accepted = accept_step(samples, inputs, block_size=128)
    output = prepare_next_step(samples, drafts, inputs, accepted, max_model_len=262144, eos_token_id=8)
    assert output.active.tolist() == [False, True, False, False]
    assert output.error.tolist() == [False, False, False, True]


def test_no_host_value_observation(monkeypatch):
    inputs = metadata()
    samples = torch.tensor([[7, 8, -1]] * 4)
    drafts = torch.ones(4, 2, dtype=torch.int64)

    def forbidden(*args, **kwargs):
        raise AssertionError("device decision observed by host")

    with monkeypatch.context() as patch:
        for name in ("cpu", "item", "tolist", "numpy", "__bool__"):
            patch.setattr(torch.Tensor, name, forbidden)
        accepted = accept_step(samples, inputs, block_size=128)
        output = prepare_next_step(samples, drafts, inputs, accepted, max_model_len=262144)
    assert output.active.tolist() == [True, True, True, False]


def test_invalid_width_rejected_before_mutating_metadata():
    inputs = metadata()
    original = inputs.clone()
    with pytest.raises(ValueError):
        accept_step(torch.ones(4, 4, dtype=torch.int64), inputs, block_size=128)
    assert torch.equal(inputs, original)


@pytest.mark.parametrize("width", [1, 2, 3, 7])
def test_device_history_excludes_bonus_and_rejected_drafts(width):
    history = torch.arange(width).expand(4, -1).int().clone()
    samples = torch.tensor([[-1, -1, -1], [21, -1, -1], [22, 23, -1], [24, 25, 26]])
    first = torch.tensor([17, 18, 19, 20])
    accepted = accept_step(samples, metadata(), block_size=128)
    result = advance_ple_history(history, first, samples, accepted)
    for row, count in enumerate(range(4)):
        committed = [] if count == 0 else [int(first[row]), *samples[row, : count - 1].tolist()]
        expected = [*history[row].tolist(), *committed][-width:]
        assert result[row].tolist() == expected


def test_device_history_repeated_rejection_matches_full_token_history():
    history = torch.tensor([[1, 2, 3]], dtype=torch.int32)
    full = [1, 2, 3]
    first = torch.tensor([4])
    inputs = metadata()[:, :].clone()
    for step in range(24):
        count = step % 3 + 1
        samples = torch.arange(100 + 3 * step, 103 + 3 * step).reshape(1, 3)
        samples[:, count:] = -1
        accepted = accept_step(samples, inputs, block_size=128)
        history = advance_ple_history(history, first, samples, accepted)
        full.extend([int(first[0]), *samples[0, : count - 1].tolist()])
        assert history[0].tolist() == full[-3:]
        first = samples[:, count - 1]


def test_device_history_has_no_host_value_observation(monkeypatch):
    inputs = metadata()
    samples = torch.tensor([[7, 8, -1]] * 4)
    accepted = accept_step(samples, inputs, block_size=128)
    history = torch.tensor([[1, 2, 3]] * 4, dtype=torch.int32)

    def forbidden(*args, **kwargs):
        raise AssertionError("history dependency observed by host")

    with monkeypatch.context() as patch:
        for name in ("cpu", "item", "tolist", "numpy", "__bool__"):
            patch.setattr(torch.Tensor, name, forbidden)
        result = advance_ple_history(history, samples[:, 0], samples, accepted)
    assert result.tolist() == [[3, 7, 7]] * 4
