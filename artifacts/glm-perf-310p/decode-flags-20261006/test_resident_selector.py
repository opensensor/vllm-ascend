# SPDX-License-Identifier: Apache-2.0
"""Test temporary host dispatch selection, restoration and repeated switches."""

from types import SimpleNamespace

import pytest
from resident_selector import select_decode_flags


@pytest.mark.parametrize("offload", [False, True])
def test_selection_restores_bank_even_on_failure(offload):
    bank = SimpleNamespace(decode_swiglu=True, decode_combine=True, offload_to_cpu=offload)
    seen = []

    def original(self, grouped_op, experts, x, weights, ids, shared):
        seen.append((experts.decode_swiglu, experts.decode_combine))
        raise ValueError("operator failed")

    selected = select_decode_flags(original, (True, False))
    with pytest.raises(ValueError, match="operator failed"):
        selected(None, None, bank, None, None, None, None)
    assert seen == [(not offload, False)]
    assert (bank.decode_swiglu, bank.decode_combine) == (True, True)


def test_repeated_selection_uses_original_dispatch():
    bank = SimpleNamespace(decode_swiglu=True, decode_combine=True, offload_to_cpu=False)

    def original(self, grouped_op, experts, x, weights, ids, shared):
        return experts.decode_swiglu, experts.decode_combine

    first = select_decode_flags(original, (True, False))
    second = select_decode_flags(first, (False, True))
    assert second(None, None, bank, None, None, None, None) == (False, True)
    assert (bank.decode_swiglu, bank.decode_combine) == (True, True)
