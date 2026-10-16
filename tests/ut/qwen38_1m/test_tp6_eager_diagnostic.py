# SPDX-License-Identifier: Apache-2.0
import pytest

from tools.qwen4exp.tp6_eager_diagnostic import ForceEager


class Runner:
    def _determine_batch_execution_and_padding(self, tokens, uniform=None, force_eager=False):
        return tokens, uniform, force_eager


@pytest.mark.parametrize("owned", [False, True])
def test_forces_eager_and_restores_exact_attribute_state(owned):
    runner = Runner()
    if owned:
        runner._determine_batch_execution_and_padding = lambda tokens, uniform=None, force_eager=False: (
            tokens,
            uniform,
            force_eager,
        )
    original = runner._determine_batch_execution_and_padding
    owner = ForceEager(runner)
    owner.install()
    assert runner._determine_batch_execution_and_padding(3, True, False) == (3, True, True)
    assert runner._determine_batch_execution_and_padding(tokens=6) == (6, None, True)
    assert owner.calls == 2
    owner.restore()
    assert runner._determine_batch_execution_and_padding == original
    assert (owner.name in vars(runner)) == owned
    assert runner._determine_batch_execution_and_padding(3) == (3, None, False)


def test_refuses_stale_owner():
    runner = Runner()
    owner = ForceEager(runner)
    owner.install()
    runner._determine_batch_execution_and_padding = object()
    with pytest.raises(RuntimeError, match="ownership"):
        owner.restore()
