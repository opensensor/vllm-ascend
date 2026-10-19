# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace as NS

import pytest
import torch

from tools.dflash_research.qfn_pair_bench import collect_oracle, collect_profile, install_oracle, install_profile


@pytest.mark.parametrize("fail", [False, True])
def test_profile_restores_original_methods_after_success_or_failure(monkeypatch, fail):
    monkeypatch.setattr(torch, "npu", NS(synchronize=lambda: None), raising=False)

    class Runner:
        drafter = None

        def _model_forward(self, x):
            if fail:
                raise ValueError("injected")
            return x + 2

    runner = Runner()
    worker = NS(rank=0, model_runner=runner)
    original = runner._model_forward
    install_profile(worker)
    with pytest.raises(RuntimeError, match="already installed"):
        install_profile(worker)
    try:
        if fail:
            with pytest.raises(ValueError, match="injected"):
                runner._model_forward(7)
        else:
            assert runner._model_forward(7) == 9
    finally:
        result = collect_profile(worker)
    assert runner._model_forward == original
    assert "_model_forward" not in vars(runner)
    assert not hasattr(worker, "_qfn_pair_profile")
    assert len(result["seconds"]["target_forward"]) == 1
    assert result["seconds"]["target_forward"][0] >= 0


def test_oracle_uses_post_draft_anchor_and_restores_method():
    class Drafter:
        positions = torch.zeros(3, dtype=torch.int32)

        def _propose(self, position):
            self.positions[0] = position
            return torch.zeros(1, 3, dtype=torch.int64)

    drafter = Drafter()
    worker = NS(rank=2, model_runner=NS(drafter=drafter))
    original = drafter._propose
    install_oracle(worker, [50, 60, 70, 80, 90, 100], 19, [0, 2, 3])
    assert drafter._propose(19).tolist() == [[61, 71, 81]]
    assert drafter._propose(20).tolist() == [[70, 80, 91]]
    assert drafter._propose(21).tolist() == [[80, 90, 100]]
    receipt = collect_oracle(worker)
    assert receipt["rank"] == 2 and len(receipt["calls"]) == 3
    assert "_propose" not in vars(drafter) and drafter._propose == original


def test_profile_times_runnable_inside_eager_graph_wrapper(monkeypatch):
    monkeypatch.setattr(torch, "npu", NS(synchronize=lambda: None), raising=False)

    class Draft(torch.nn.Module):
        def forward(self, value):
            return value + 1

    class Wrapper:
        def __init__(self, runnable):
            self.runnable = runnable

        def unwrap(self):
            return self.runnable

        def __call__(self, value):
            return self.runnable(value)

    draft = Draft()
    wrapper = Wrapper(draft)
    drafter = NS(model=wrapper, _propose=lambda value: wrapper(value))
    worker = NS(rank=0, model_runner=NS(drafter=drafter, _model_forward=lambda value: value))
    install_profile(worker)
    assert drafter._propose(2) == 3
    result = collect_profile(worker)
    assert len(result["seconds"]["draft_forward"]) == 1
    assert len(result["seconds"]["draft_total_inclusive"]) == 1
    assert "forward" not in vars(wrapper) and "forward" not in vars(draft)


def test_oracle_logit_trace_preserves_target_output_and_local_methods():
    logits = torch.arange(128).float().repeat(3, 1)
    model = NS(compute_logits=lambda hidden: logits)
    runner = NS(
        model=model,
        _model_forward=lambda n, ids, positions: torch.ones(n, 4),
        drafter=NS(_propose=lambda: torch.zeros(1, 3, dtype=torch.int64), positions=torch.tensor([19, 20, 21])),
    )
    original_forward, original_logits = runner._model_forward, model.compute_logits
    worker = NS(rank=0, model_runner=runner)
    install_oracle(worker, [50, 60, 70, 80, 90], 19, [2])
    hidden = runner._model_forward(3, None, torch.tensor([19, 20, 21]))
    assert model.compute_logits(hidden) is logits
    result = collect_oracle(worker)
    assert result["logits"][0]["output_offsets"] == [1, 2, 3]
    assert result["logits"][0]["reference_scores"] == [[60.0], [70.0], [80.0]]
    assert runner._model_forward is original_forward and model.compute_logits is original_logits
