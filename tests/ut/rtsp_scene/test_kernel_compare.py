import json
from contextlib import nullcontext
from types import SimpleNamespace

import torch

from tools.rtsp_scene.kernel_compare import compare_case


def test_nonfinite_candidate_fails_without_invalid_json():
    x = torch.empty(1, 4, dtype=torch.float16)
    result = compare_case(
        lambda x: x.clone(),
        (lambda x: x.clone(), lambda x: torch.full_like(x, float("nan"))),
        x,
        iterations=1,
        repeats=1,
    )
    assert not result["passed"]
    assert "graph_control" not in result
    assert all(c["bitwise_equal"] == [True, False] and c["max_abs_error"][1] is None for c in result["checks"])
    json.dumps(result, allow_nan=False)


def test_check_only_does_not_capture_or_report_timings():
    x = torch.empty(1, 4, dtype=torch.float16)
    result = compare_case(
        lambda x: x.clone(),
        (lambda x: x.clone(), lambda x: x.clone()),
        x,
        iterations=1,
        repeats=1,
        check_only=True,
    )
    assert result["passed"] and len(result["checks"]) == 5
    assert "graph_checks" not in result and "reduction_percent" not in result


def test_stale_graph_outputs_never_receive_a_speedup_claim(monkeypatch):
    class StaleGraph:
        def replay(self):
            pass

    monkeypatch.setattr(
        torch,
        "npu",
        SimpleNamespace(NPUGraph=StaleGraph, graph=lambda *a, **kw: nullcontext(), graph_pool_handle=object),
        raising=False,
    )
    x = torch.empty(1, 4, dtype=torch.float16)
    result = compare_case(lambda x: x.clone(), (lambda x: x.clone(), lambda x: x.clone()), x, iterations=1, repeats=1)
    assert result["passed"] and not result["graph_passed"]
    assert "reduction_percent" not in result and "graph_control" not in result
