# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import pytest
import torch

from tools.qwen4exp import hc_epilogues as hc


@pytest.fixture
def owner(monkeypatch):
    monkeypatch.setattr(hc, "_on_npu", lambda device: True)
    monkeypatch.setattr(hc, "_capturing", lambda: False)
    calls = []
    result = hc.HCEpilogues("affine", "mean", lambda *args: calls.append(args))
    return result, calls


def inputs(rows=3):
    return torch.empty(rows, hc.STREAMS * hc.WIDTH), torch.ones(hc.STREAMS * hc.WIDTH)


def test_affine_preserves_distinct_output_storage_and_reuses_descriptor(owner):
    op, calls = owner
    values, weight = inputs()
    first = op.affine(values, weight)
    second = op.affine(values, weight)
    assert first[0].dtype == torch.float32 and first[1].dtype == torch.float16
    assert len({t.data_ptr() for t in (*first, *second, values)}) == 5
    assert calls[0][1][-1] is calls[1][1][-1]
    assert calls[0][1][0] is values and calls[0][1][1] is weight


@pytest.mark.parametrize("bad", ["shape", "dtype", "strided"])
def test_invalid_gate_never_launches(owner, bad):
    op, calls = owner
    values, _ = inputs()
    gate = torch.empty_like(values, dtype=torch.float16)
    if bad == "shape":
        gate = gate[:, :-1]
    elif bad == "dtype":
        gate = gate.float()
    else:
        gate = torch.empty(3, hc.STREAMS * hc.WIDTH * 2, dtype=torch.float16)[:, ::2]
    with pytest.raises(ValueError):
        op.mean(values, gate)
    assert not calls


def test_capture_rejected_even_with_cached_descriptor(owner, monkeypatch):
    op, calls = owner
    values, weight = inputs()
    op.affine(values, weight)
    monkeypatch.setattr(hc, "_capturing", lambda: True)
    with pytest.raises(RuntimeError, match="eager-only"):
        op.affine(values, weight)
    assert len(calls) == 1


def test_failed_launch_retains_exception_and_prevents_reentry(owner):
    op, calls = owner
    values, weight = inputs()
    failure = RuntimeError("launch failed")

    def fail(*args):
        raise failure

    op.launch = fail
    with pytest.raises(RuntimeError, match="launch failed"):
        op.affine(values, weight)
    assert op.failed and op.failure is failure and failure.__traceback__ is not None
    with pytest.raises(RuntimeError, match="retain"):
        op.affine(values, weight)


def test_descriptor_capacity_never_evicts(owner, monkeypatch):
    op, calls = owner
    monkeypatch.setattr(hc, "MAX_CONFIGS", 1)
    op.affine(*inputs(1))
    with pytest.raises(RuntimeError, match="capacity"):
        op.affine(*inputs(2))
    assert len(op.configs) == len(calls) == 1


@pytest.mark.parametrize(
    "rows,allowed", [(3, False), (128, False), (255, False), (256, True), (515, True), (2048, True), (2049, False)]
)
def test_prefill_admission(owner, rows, allowed):
    module = SimpleNamespace(
        hc_count=4,
        hidden_size=2560,
        params_dtype=torch.float16,
        compute_dtype=torch.float32,
        use_combine=True,
        input_mix_weight_down=torch.empty(0, dtype=torch.float16),
        input_mix_weight_up=torch.empty(0, dtype=torch.float16),
        block_inject_weight=torch.empty(0, dtype=torch.float16),
        _rms_unit_weight=torch.empty(0),
    )
    values = torch.empty(rows, 10240, dtype=torch.float16)
    with torch.no_grad():
        assert hc.eligible(module, values) is allowed
    assert not hc.eligible(module, values)


def test_verification_returns_reference_and_disables_mismatch(owner, monkeypatch):
    op, _ = owner
    monkeypatch.setattr(hc, "eligible", lambda *args: True)
    expected = (torch.ones(1, 2).half(), (None, torch.ones(1, 2)))
    adapter = hc.HCEpilogueAdapter(op, lambda *args: expected)
    adapter.candidate = lambda *args: (torch.zeros(1, 2).half(), expected[1])
    assert adapter.replacement(None, None) is expected
    assert adapter.checked == adapter.mismatches == 1
    assert not adapter.enabled
    assert adapter.replacement(None, None) is expected
    assert adapter.calls == 1
