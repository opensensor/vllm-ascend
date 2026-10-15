# SPDX-License-Identifier: Apache-2.0
"""Model the executor's uneven reply queues without hardware or process hooks."""

from collections import deque

import pytest

from tools.sidecard.qfn_rpc_recovery import suppress_stale_replies


class Process:
    use_async_scheduling = False

    def __init__(self, debt):
        self.queue = deque(["stale"] * debt)

    def enqueue_output(self, output):
        self.queue.append(output)


@pytest.mark.parametrize("debts", [(0, 1, 2, 3), (0, 3, 6, 9), (0, 0, 0, 0)])
def test_measured_debt_is_drained_without_losing_next_control_reply(debts):
    processes = [Process(debt) for debt in debts]
    for p, debt in zip(processes, debts):
        suppress_stale_replies(p, debt, "repair", inference_drained=True)
    for wave in range(max(1, max(debts))):
        for p in processes:
            p.enqueue_output(("probe", wave))
            p.queue.popleft()
    for p, debt in zip(processes, debts):
        assert not p.queue
        assert "enqueue_output" not in vars(p)
        assert p._qfn_reply_recovery_receipt == dict(nonce="repair", requested=debt, suppressed=debt, complete=True)
        p.enqueue_output("next")
        assert p.queue.popleft() == "next"


def test_retry_is_idempotent_and_restores_existing_instance_override():
    p = Process(2)
    original = p.enqueue_output
    p.enqueue_output = original
    receipt = suppress_stale_replies(p, 2, "repair", inference_drained=True)
    p.enqueue_output(None)
    assert suppress_stale_replies(p, 2, "repair", inference_drained=True) is receipt
    with pytest.raises(RuntimeError):
        suppress_stale_replies(p, 1, "another", inference_drained=True)
    p.enqueue_output(None)
    assert p.enqueue_output is original
    p.enqueue_output("next")
    assert list(p.queue) == ["stale", "stale", "next"]


@pytest.mark.parametrize("count", [-1, 17, True, 1.5])
def test_bad_debt_does_not_install_hook(count):
    p = Process(0)
    with pytest.raises(ValueError):
        suppress_stale_replies(p, count, "repair", inference_drained=True)
    assert "enqueue_output" not in vars(p)


def test_async_or_undrained_workers_are_rejected():
    p = Process(0)
    with pytest.raises(ValueError):
        suppress_stale_replies(p, 1, "repair", inference_drained=False)
    p.use_async_scheduling = True
    with pytest.raises(ValueError):
        suppress_stale_replies(p, 1, "repair", inference_drained=True)
