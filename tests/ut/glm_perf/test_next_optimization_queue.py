# SPDX-License-Identifier: Apache-2.0
"""CPU contracts for the deferred five-experiment GLM queue."""

import ast
import importlib.util
import inspect
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tools.glm_perf.kda_gate_beta_native import GateBeta
from tools.glm_perf.optimization_queue import ExpertBatch, queue
from tools.glm_perf.resident_candidates.kda_input_preparation import (
    gate_beta_reference,
    make_preparer,
    normalize_qk,
    replace_input_prelude,
)
from tools.glm_perf.resident_candidates.kpool_decode_epilogue import select_and_expand

ROOT = Path(__file__).resolve().parents[3]


def load_source(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("tokens,routes", [(640, 6144), (1280, 10240), (2560, 20480)])
def test_batch_admission(tokens, routes):
    plan = ExpertBatch(tokens, routes).plan()
    assert plan["hf_override"]["ascend_glm_grouped_max_routes"] == routes
    assert plan["requires_runner_reallocation"]
    assert len(queue()["experiments"]) == 5


@pytest.mark.parametrize(
    "tokens,routes,context",
    [(1280, 6144, 311040), (2560, 10240, 311040), (1024, 10240, 311040), (640, 32769, 311040), (640, 6144, 131072)],
)
def test_batch_refuses_splitting_or_confounders(tokens, routes, context):
    with pytest.raises(ValueError):
        ExpertBatch(tokens, routes, context)


@pytest.mark.parametrize("rows", [1, 2, 8])
def test_qk_one_normalization_for_strided_inputs(rows):
    x = torch.randn(1, rows, 16, 256, dtype=torch.float16)
    q, k = x[..., ::2], x[..., 1::2]
    before = x.clone()
    calls = []

    def norm(value):
        calls.append((value.shape, value.is_contiguous()))
        return torch.nn.functional.normalize(value.float(), dim=-1).half()

    expected_q, expected_k = norm(q), norm(k)
    calls.clear()
    actual_q, actual_k = normalize_qk(q, k, norm)
    assert calls == [(torch.Size([2, rows, 16, 128]), True)]
    assert torch.equal(actual_q, expected_q) and torch.equal(actual_k, expected_k)
    assert torch.equal(x, before)


def test_qk_incompatible_dtype_falls_back():
    q, k = torch.ones(1, 2, 16, 128).half(), torch.ones(1, 2, 16, 128)
    calls = []
    normalize_qk(q, k, lambda x: calls.append(x.dtype) or x)
    assert calls == [torch.float16, torch.float32]


@pytest.mark.parametrize("accepted", [False, True])
def test_preparation_preserves_actual_recurrent_state_arguments(accepted):
    module = load_source("queue_kda", "vllm_ascend/models/glm5next_w2/kda_310.py")
    captured = []

    def kernel(**kwargs):
        captured.append(kwargs)
        return kwargs["value"]

    # Replace only this private test module's torch binding; no global monkeypatch.
    module.torch = SimpleNamespace(
        float16=torch.float16,
        int32=torch.int32,
        ops=SimpleNamespace(_C_ascend=SimpleNamespace(npu_recurrent_gated_delta_rule_310=kernel)),
    )
    module._l2norm_310p = lambda x: torch.nn.functional.normalize(x.float(), dim=-1).half()
    module._safe_gate_for_layer = lambda layer, raw: raw.float().sigmoid() * -5
    module._actual_lengths = lambda cu, n: (cu[1 : n + 1] - cu[:n]).int()
    module._flatten_spec_state_indices = lambda indices, lengths, tokens: indices[:, 0]
    original = module._run_recurrent
    candidate = replace_input_prelude(
        original, make_preparer(module._l2norm_310p, module._safe_gate_for_layer, batch_qk=True)
    )
    tensors = [torch.randn(1, 4, 16, 128).half() for _ in range(4)]
    state = torch.randn(12, 2)
    before = state.clone()
    args = (
        SimpleNamespace(head_dim=128),
        *tensors,
        torch.randn(1, 4, 16),
        state,
        torch.tensor([0, 2, 4]),
        torch.tensor([[3, 4, 5], [7, 8, 9]]),
    )
    counts = torch.tensor([3, 1]) if accepted else None
    original(*args, num_sequences=2, num_accepted_tokens=counts)
    candidate(*args, num_sequences=2, num_accepted_tokens=counts)
    assert captured[0].keys() == captured[1].keys()
    for name in captured[0]:
        a, b = captured[0][name], captured[1][name]
        assert torch.equal(a, b) if isinstance(a, torch.Tensor) else a == b
    assert captured[1]["state"] is state and torch.equal(state, before)
    if accepted:
        assert captured[1]["ssm_state_indices"].shape == (2, 3)
        bad_args = (*args[:-1], args[-1][:, 0])
        with pytest.raises(ValueError, match="full per-request"):
            candidate(*bad_args, num_sequences=2, num_accepted_tokens=counts)
    # The AST after preparation, including native kernel arguments, is untouched.
    tree = ast.parse(inspect.getsource(original)).body[0]
    assert any(isinstance(node, ast.If) for node in tree.body[6:])


def test_changed_prelude_rejected():
    def different(q):
        return q

    with pytest.raises(ValueError, match="re-audit"):
        replace_input_prelude(different, lambda *args: args)


@pytest.mark.parametrize("rows,capacity,budget", [(1, 0, 16), (2, 3, 16), (8, 77, 16), (8, 77760, 2048)])
def test_epilogue_exact_indices_ties_tails_and_padding(rows, capacity, budget):
    baseline = load_source("queue_kpool", "vllm_ascend/models/glm5next/kpool_ops.py")
    positions = torch.tensor([-1, 0, 2, 3, 4, 15, 1023, 300051][:rows])
    logits = torch.randint(-2, 3, (rows, capacity)).float()  # Includes boundary ties.
    logits.masked_fill_(torch.arange(capacity)[None, :] >= ((positions + 1) // 4)[:, None], -torch.inf)
    before = logits.clone()
    expected = []
    for row in range(rows):
        selected, _, starts, counts = baseline.select_kpool_groups(
            logits[row : row + 1], positions[row : row + 1], budget, 4, scores_are_causal=True
        )
        expected.append(baseline.expand_kpool_groups(selected, starts, counts, 4))
    assert torch.equal(select_and_expand(logits, positions, budget, 4), torch.cat(expected))
    assert torch.equal(logits, before)


def test_epilogue_preserves_topk_shape_and_masks_padding_alias(monkeypatch):
    calls = []

    def topk(value, count, dim):
        calls.append(tuple(value.shape))
        return SimpleNamespace(indices=torch.zeros(value.shape[0], count, dtype=torch.long))

    monkeypatch.setattr(torch, "topk", topk)
    result = select_and_expand(torch.zeros(2, 100), torch.tensor([3, -1]), 16, 4)
    assert calls == [(1, 100), (1, 100)]
    assert result[0, :4].tolist() == [0, 1, 2, 3]
    assert (result[0, 4:] == -1).all() and (result[1] == -1).all()


@pytest.mark.parametrize("cached_current", [True, False])
def test_gate_fusion_cached_weights_identity_and_fallback(cached_current):
    a_log, bias = torch.zeros(16), torch.randn(1, 1, 16, 128)
    layer = SimpleNamespace(A_log=a_log, dt_bias=bias, kda_lower_bound=-5)
    scale = torch.ones(1, 1, 16, 1)
    layer._kda_gate_weights = (a_log if cached_current else a_log.clone(), bias, scale, bias)
    calls = []

    class Native:
        def supports(self, *args):
            return True

        def __call__(self, *args):
            calls.append("native")
            return gate_beta_reference(*args)

    def safe(layer, raw):
        calls.append("fallback")
        return -5 * torch.sigmoid(scale * (raw.float() + bias))

    prepare = make_preparer(lambda x: x, safe, gate_beta=Native())
    raw = torch.linspace(-100, 100, 4096).reshape(1, 2, 16, 128).half()
    beta = torch.linspace(-100, 100, 32).reshape(1, 2, 16).half()
    output = prepare(layer, raw, raw, raw, raw, beta)
    expected = gate_beta_reference(raw, beta, scale, bias, -5)
    assert calls == ["native" if cached_current else "fallback"]
    assert output[3].dtype == torch.float32 and output[4].dtype == torch.float16
    assert torch.equal(output[3], expected[0]) and torch.equal(output[4], expected[1])


def test_native_wrapper_rejects_unsupported_without_launch():
    resource = GateBeta.__new__(GateBeta)
    resource.heads, resource.lower_bound, resource.lower = 16, -5, torch.tensor([-5.0])
    resource.tiling = {(2, torch.float16, torch.float16): None}
    raw, beta = torch.zeros(1, 2, 16, 128).half(), torch.zeros(1, 2, 16).half()
    scale, bias = torch.ones(1, 1, 16, 1), torch.zeros(1, 1, 16, 128)
    assert resource.supports(raw, beta, scale, bias, -5)
    for args in [
        (raw.bfloat16(), beta, scale, bias, -5),
        (raw, beta, scale, bias, -6),
        (raw[..., ::2], beta, scale, bias, -5),
    ]:
        assert not resource.supports(*args)
        with pytest.raises(ValueError, match="unsupported"):
            resource(*args)


def test_adaptive_teams_exact_coverage_and_hot_expert_uses_all_cores(tmp_path):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("host C++ compiler required")
    source = tmp_path / "teams.cpp"
    source.write_text(r"""
#include <cassert>
#include <vector>
#define __aicore__
#include "adaptive_expert_core_groups.h"
int main() {
  for (unsigned cores=1; cores<=8; ++cores)
  for (unsigned lanes=0; lanes<=16; ++lanes)
  for (unsigned total : {1, 32, 64, 65, 5120, 10240})
  for (unsigned tiles : {1, 3, 16, 32, 128}) {
    const std::vector<unsigned> counts = {0, 1, 127, 0, 128, 129, 640, 2, 0, 63};
    std::vector<unsigned> writes(counts.size()*tiles, 0);
    for (unsigned core=0; core<cores; ++core) {
      unsigned small=0;
      for (unsigned e=0; e<counts.size(); ++e) {
        if (!counts[e]) continue;
        auto t=GlmAdaptive::Plan(core, cores, lanes, total, counts[e]);
        assert(t.lane<t.lanes && t.group<t.groups);
        if (counts[e]>=128 || total<=64) assert(t.lanes==cores && t.groups==1);
        if (t.groups!=1 && small++%t.groups!=t.group) continue;
        for (unsigned tile=t.lane; tile<tiles; tile+=t.lanes) ++writes[e*tiles+tile];
      }
    }
    for (unsigned e=0; e<counts.size(); ++e)
      for (unsigned tile=0; tile<tiles; ++tile) assert(writes[e*tiles+tile]==(counts[e]?1u:0u));
  }
}
""")
    executable = tmp_path / "teams"
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-O2",
            "-Wall",
            "-Werror",
            "-I",
            str(ROOT / "artifacts/glm-perf-310p/next-queue-20261005"),
            str(source),
            "-o",
            str(executable),
        ],
        check=True,
    )
    subprocess.run([str(executable)], check=True)
