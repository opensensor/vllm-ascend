# SPDX-License-Identifier: Apache-2.0
import json
import sys

import pytest

from tools.sidecard import architecture_benchmark
from tools.sidecard.architecture_benchmark import cases, summarize, validate
from tools.sidecard.architecture_experts import EXPERTS, FIRST_EXPERT, route_fixture


def test_actual_glm_shapes_and_decode_only_head():
    matrix = cases([1, 512])
    by_name = {case["name"]: case for case in matrix if case["m"] == 1}
    assert (by_name["kda_qkv_bfg_tp2"]["n"], by_name["kda_qkv_bfg_tp2"]["k"]) == (12576, 4096)
    assert (by_name["kda_qkv_bfg_full"]["n"], by_name["kda_qkv_bfg_full"]["k"]) == (24896, 4096)
    assert by_name["lm_head_tp2"]["n"] == 154880 // 2
    assert not any(case["name"] == "lm_head_tp2" and case["m"] == 512 for case in matrix)
    assert by_name["mhc_projection"]["dtype"] == "fp32"
    assert by_name["mhc_projection"]["k"] == 4 * 4096


@pytest.mark.parametrize("rows", [[0], [True], [640]])
def test_reject_unqualified_rows(rows):
    with pytest.raises(ValueError, match="rows"):
        cases(rows)


def test_timing_summary_preserves_variation():
    result = summarize([4.0, 1.0, 2.0])
    assert result == {"median_ms": 2.0, "min_ms": 1.0, "max_ms": 4.0, "samples_ms": [4.0, 1.0, 2.0]}
    with pytest.raises(ValueError):
        summarize([0.0])


def test_arithmetic_gate_detects_bad_orientation():
    torch = pytest.importorskip("torch")
    case = {"op": "gemm", "m": 4, "n": 4}
    x = torch.arange(12).reshape(4, 3).float()
    w = torch.arange(12, 24).reshape(4, 3).float()
    expected = x @ w.T
    assert validate(case, (x, w), expected, torch)["passed"]
    with pytest.raises(AssertionError):
        validate(case, (x, w), expected.T, torch)


@pytest.mark.parametrize("tokens", [1, 4, 128, 512])
def test_matched_routing_maps_cuda_rows_back_to_native_tokens(tokens):
    torch = pytest.importorskip("torch")
    hidden, ids, weights, routes, row_map = route_fixture(tokens, torch)
    assert hidden.shape == (tokens, 4096)
    assert len(routes) == tokens * 2
    assert routes == sorted(routes)
    for index, (expert, token, slot) in enumerate(routes):
        assert 0 <= expert < EXPERTS
        assert ids[token, slot] == FIRST_EXPERT + expert
        assert row_map[token, slot] == index
        assert weights[token, slot] == 1 / 8
    assert bool((row_map[:, 2:] == -1).all())
    assert bool((ids[:, 2:] == -1).all())
    assert bool((weights[:, 2:] == 0).all())


@pytest.mark.parametrize("interrupt", [False, True])
def test_partial_hardware_report_does_not_claim_completion(tmp_path, monkeypatch, interrupt):
    pytest.importorskip("torch")
    case = {"name": "test", "op": "gemm", "m": 2, "n": 3, "k": 4, "dtype": "fp16"}
    output = tmp_path / "results.json"
    monkeypatch.setattr(architecture_benchmark, "cases", lambda rows: [case, case])
    monkeypatch.setattr(sys, "argv", ["benchmark", "--backend", "cuda", "--library", "unused", "--output", str(output)])

    class Backend:
        calls = 0

        def measure(self, case, inputs, output, warmups, samples):
            self.calls += 1
            if interrupt and self.calls == 2:
                raise RuntimeError("hardware failure")
            output.copy_((inputs[0].float() @ inputs[1].float().T).half())
            return output, [1.0] * samples, [2.0] * samples

    monkeypatch.setattr(architecture_benchmark, "CudaProbe", lambda library: Backend())
    if interrupt:
        with pytest.raises(RuntimeError, match="hardware failure"):
            architecture_benchmark.main()
    else:
        architecture_benchmark.main()
    report = json.loads(output.read_text())
    assert report["completed"] is not interrupt
    assert len(report["cases"]) == (1 if interrupt else 2)
    assert report["cases"][0]["validation"]["passed"]
