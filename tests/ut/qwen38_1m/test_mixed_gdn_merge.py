# SPDX-License-Identifier: Apache-2.0
"""Exact merge order, metadata lifetime and non-disruptive rejection gates."""

import ast
import inspect
import textwrap
from copy import copy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tools.qwen4exp.mixed_gdn_merge import MixedGDNMerge, build_forward, inverse_order, reference_merge


@pytest.mark.parametrize("rows,short", [(515, 3), (518, 6), (521, 9), (1923, 3), (17, 1)])
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_arbitrary_permutation_preserves_every_output_bit(rows, short, dtype):
    generator = torch.Generator().manual_seed(24)
    order = torch.randperm(rows, generator=generator).to(dtype)
    metadata = SimpleNamespace(spec_token_indx=order[:short], non_spec_token_indx=order[short:])
    # Include signed zero, infinities and NaN payloads; this is data movement.
    bits = torch.randint(-32768, 32768, (rows, 12, 128), generator=generator, dtype=torch.int16)
    first, second = bits[:short].view(torch.float16), bits[short:].view(torch.float16)
    owner = MixedGDNMerge()
    owner.verify = False
    expected = reference_merge((first, second), metadata)
    actual = owner.merge((first, second), metadata)
    assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))
    assert inverse_order(metadata, rows) is inverse_order(metadata, rows)
    first.fill_(1)
    second.fill_(-2)
    assert torch.equal(owner.merge((first, second), metadata), reference_merge((first, second), metadata))


def test_new_step_or_replaced_indices_get_a_new_inverse():
    metadata = SimpleNamespace(spec_token_indx=torch.tensor([2]), non_spec_token_indx=torch.tensor([0, 1]))
    first = inverse_order(metadata, 3)
    metadata.spec_token_indx = torch.tensor([0])
    metadata.non_spec_token_indx = torch.tensor([1, 2])
    second = inverse_order(metadata, 3)
    assert second is not first
    assert second.tolist() == [0, 1, 2]


def test_shadow_mismatch_returns_reference_and_disables_candidate(monkeypatch):
    from tools.qwen4exp import mixed_gdn_merge

    metadata = SimpleNamespace(spec_token_indx=torch.tensor([2]), non_spec_token_indx=torch.tensor([0, 1]))
    branches = (torch.ones(1, 2, dtype=torch.float16), torch.zeros(2, 2, dtype=torch.float16))
    owner = MixedGDNMerge()
    monkeypatch.setattr(mixed_gdn_merge, "inverse_order", lambda metadata, rows: torch.arange(rows))
    expected = reference_merge(branches, metadata)
    assert torch.equal(owner.merge(branches, metadata), expected)
    assert owner.failed and owner.mismatches == 1
    owner.verify = False
    assert torch.equal(owner.merge(branches, metadata), expected)
    assert owner.calls == owner.checked == 1


def test_bad_partition_geometry_is_rejected():
    metadata = SimpleNamespace(spec_token_indx=torch.tensor([0]), non_spec_token_indx=torch.tensor([1]))
    with pytest.raises(ValueError, match="partition"):
        inverse_order(metadata, 3)


def test_qualified_factory_keeps_state_updates_and_branch_order(monkeypatch):
    path = Path(__file__).resolve().parents[3] / "vllm_ascend/models/qwen4_exp/model.py"
    text = path.read_text()
    cls = next(n for n in ast.parse(text).body if isinstance(n, ast.ClassDef) and n.name == "_GDNAttention")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_native_mixed_attention")
    source = textwrap.dedent("".join(text.splitlines(keepends=True)[method.lineno - 1 : method.end_lineno]))
    namespace = dict(torch=torch, copy=copy)
    exec(compile(source, str(path), "exec"), namespace)
    original = namespace[method.name]
    monkeypatch.setattr(inspect, "getsource", lambda fn: source)
    owner = MixedGDNMerge()
    owner.verify = False
    candidate = build_forward(original, owner)
    calls = []

    def conv(mixed, state_indices, *args, **kwargs):
        calls.append(tuple(state_indices.tolist()))
        return mixed

    model = SimpleNamespace(
        num_v_heads=1,
        num_k_heads=1,
        key_dim=2,
        value_dim=2,
        compute_dtype=torch.float16,
        params=SimpleNamespace(head_k_dim=2, head_v_dim=2),
        _native_gating=lambda a, b: (a, b),
        _stateful_short_conv=conv,
        _native_delta_rule=lambda q, k, v, *args: v * 2,
    )
    metadata = SimpleNamespace(
        num_spec_decode_tokens=3,
        num_prefill_tokens=5,
        num_decode_tokens=0,
        spec_token_indx=torch.tensor([1, 4, 7]),
        non_spec_token_indx=torch.tensor([0, 2, 3, 5, 6]),
        spec_state_indices_tensor=torch.tensor([11]),
        non_spec_state_indices_tensor=torch.tensor([12]),
        spec_query_start_loc=torch.tensor([0, 3]),
        non_spec_query_start_loc=torch.tensor([0, 5]),
        has_initial_state=torch.tensor([True]),
    )
    mixed = torch.arange(48).reshape(8, 6).half()
    g = torch.zeros(1, 8, 1)
    expected = original(model, mixed, g, g, metadata)
    baseline_calls = list(calls)
    calls.clear()
    assert torch.equal(candidate(model, mixed, g, g, metadata), expected)
    assert calls == baseline_calls == [(11,), (12,)]
    monkeypatch.setattr(inspect, "getsource", lambda fn: source + "\n# changed\n")
    with pytest.raises(RuntimeError, match="qualified source"):
        build_forward(original, owner)
