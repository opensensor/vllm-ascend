# SPDX-License-Identifier: Apache-2.0
"""Host gates for mixed-prefill eligibility and reversible method ownership."""

from types import SimpleNamespace

import pytest

from tools.qwen4exp.mixed_qsa import install, matmul_dispatch, restore


@pytest.mark.parametrize(
    "lengths,capacity,expected",
    [
        ((3, 1920), 480, True),
        ((256, 3), 512, True),
        ((3, 129, 2), 512, True),
        ((3, 3), 512, False),
        ((3, 3), 2048, True),
        ((3, 32), 512, False),
        ((0, 128, 3), 512, True),
    ],
)
def test_prefill_justifies_matmul_without_changing_decode_crossover(lengths, capacity, expected):
    assert matmul_dispatch(lengths, sum(lengths), capacity, len(lengths), len(lengths) + 1, 8) is expected


@pytest.mark.parametrize("lengths,tokens,requests", [((3, 128), 130, 2), ((3, -1), 2, 2), ((128,), 128, 2)])
def test_matmul_rejects_wrong_request_boundaries(lengths, tokens, requests):
    with pytest.raises(ValueError, match="packed query lengths"):
        matmul_dispatch(lengths, tokens, 512, requests, requests + 1, 8)


def test_restore_preserves_static_descriptor_and_checks_every_owner_first():
    class Target:
        @staticmethod
        def original():
            return "original"

    descriptor = vars(Target)["original"]
    replacement = staticmethod(lambda: "new")
    Target.original = replacement
    worker = SimpleNamespace(
        _qfn_mixed_qsa=dict(
            mode="score",
            originals=[(Target, "original", True, descriptor)],
            replacements=[(Target, "original", replacement)],
        ),
        rank=0,
    )
    assert restore(worker)["restored"] == "score"
    assert vars(Target)["original"] is descriptor
    assert worker._qfn_mixed_qsa is None


def test_failed_attention_modes_cannot_be_installed():
    for mode in ("batched", "dense", "unknown"):
        with pytest.raises(ValueError, match="only score dispatch is qualified"):
            install(SimpleNamespace(), mode, "")


def test_restore_refuses_changed_ownership_without_partial_mutation():
    target = SimpleNamespace(first=object(), second=object())
    original_first, original_second = object(), object()
    first, second = target.first, target.second
    state = dict(
        mode="score",
        originals=[(target, "first", True, original_first), (target, "second", True, original_second)],
        replacements=[(target, "first", first), (target, "second", second)],
    )
    worker = SimpleNamespace(rank=0, _qfn_mixed_qsa=state)
    target.second = object()
    with pytest.raises(RuntimeError, match="ownership changed"):
        restore(worker)
    assert target.first is first
    assert worker._qfn_mixed_qsa is state
