# SPDX-License-Identifier: Apache-2.0
"""Host-only tests for the 310P Qwen4Exp MTP runner inputs."""

import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm_ascend._310p.prefix_mamba_state import (
    PREFIX_MAMBA_MIN_SLOTS,
    prefix_mamba_slot_count,
    supports_compact_live_mamba_state,
    uses_bounded_prefix_mamba_scheduler,
)
from vllm_ascend._310p.qwen4exp_mtp import (
    is_qwen4exp_mtp_config,
    is_qwen4exp_speculative_target,
    qwen4exp_mtp_hidden_width,
    stage_ple_history,
    validate_qwen4exp_speculative_scheduling,
)


@pytest.mark.parametrize("arch", ["Qwen4ExpForCausalLM", "Qwen4ExpForConditionalGeneration"])
def test_only_qwen4exp_mtp_uses_310p_v1_path(arch):
    model_config = SimpleNamespace(architectures=[arch])
    assert is_qwen4exp_mtp_config(model_config, SimpleNamespace(method="mtp"))
    assert not is_qwen4exp_mtp_config(model_config, SimpleNamespace(method="ngram"))
    assert not is_qwen4exp_mtp_config(
        SimpleNamespace(architectures=["Qwen3ForCausalLM"]), SimpleNamespace(method="mtp")
    )


def test_qwen4exp_draft_uses_all_hyperconnection_streams():
    hf_config = SimpleNamespace(architectures=["Qwen4ExpMTP"], hc_count=4)
    draft = SimpleNamespace(hf_config=hf_config, get_hidden_size=lambda: 2560)
    assert qwen4exp_mtp_hidden_width(draft, "mtp") == 10240
    assert qwen4exp_mtp_hidden_width(draft, "ngram") is None
    hf_config.hc_count = 0
    with pytest.raises(ValueError, match="hc_count"):
        qwen4exp_mtp_hidden_width(draft, "mtp")


def test_ple_history_follows_request_position_and_speculative_rollback():
    context = torch.empty((3, 3), dtype=torch.int32)
    boundaries = torch.empty(4, dtype=torch.int32)
    token_ids = np.array([[10, 11, 12, 13, 14], [20, 21, 22, 23, 24]], dtype=np.int32)
    computed = np.array([5, 2], dtype=np.int32)
    stage_ple_history(context, boundaries, token_ids, computed, np.array([0, 1, 3]), 2, 4, 99)
    assert context.tolist() == [[12, 13, 14], [99, 20, 21], [99, 99, 99]]
    assert boundaries.tolist() == [0, 1, 3, 4]

    # Rejection rolls request 0 back to its previous accepted position.
    computed[0] = 3
    stage_ple_history(context, boundaries, token_ids, computed, np.array([0, 1, 2]), 2, 2, 99)
    assert context[0].tolist() == [10, 11, 12]
    assert boundaries.tolist() == [0, 1, 2, 2]


def test_ple_history_rejects_invalid_positions():
    context = torch.empty((2, 2), dtype=torch.int32)
    boundaries = torch.empty(3, dtype=torch.int32)
    token_ids = np.zeros((1, 4), dtype=np.int32)
    with pytest.raises(ValueError, match="history exceeds"):
        stage_ple_history(context, boundaries, token_ids, np.array([5]), np.array([0, 1]), 1, 1, 99)
    with pytest.raises(ValueError, match="query length exceeds"):
        stage_ple_history(context, boundaries, token_ids, np.array([0]), np.array([0, 2]), 1, 1, 99)


@pytest.mark.parametrize("method", ["mtp", "dspark", "ngram", "dflash", None])
@pytest.mark.parametrize("arch", ["Qwen4ExpForCausalLM", "Qwen4ExpForConditionalGeneration", "Qwen3ForCausalLM"])
def test_target_state_does_not_enable_mtp_draft_optimizations(method, arch):
    model, spec = SimpleNamespace(architectures=[arch]), SimpleNamespace(method=method)
    expected = arch.startswith("Qwen4Exp") and method in {"mtp", "dspark"}
    assert is_qwen4exp_speculative_target(model, spec) == expected
    assert is_qwen4exp_mtp_config(model, spec) == (expected and method == "mtp")
    validate_qwen4exp_speculative_scheduling(model, spec, False)
    if expected:
        with pytest.raises(ValueError, match="no-async-scheduling"):
            validate_qwen4exp_speculative_scheduling(model, spec, True)
    else:
        validate_qwen4exp_speculative_scheduling(model, spec, True)


@pytest.mark.parametrize("k", [1, 3, 7])
def test_every_accepted_prefix_discards_rejected_ple_history(k):
    context = torch.empty((3, 3), dtype=torch.int32)
    boundaries = torch.empty(4, dtype=torch.int32)
    # Request 0 is verified speculatively; request 1's history is independent.
    table = np.array([np.arange(20), np.arange(100, 120)], dtype=np.int32)
    for accepted in range(k + 1):
        computed = np.array([4 + accepted, 3])
        stage_ple_history(context, boundaries, table, computed, np.array([0, k + 1, k + 2]), 2, k + 2, 99)
        assert context[0].tolist() == table[0, computed[0] - 3 : computed[0]].tolist()
        assert context[1].tolist() == [100, 101, 102]
        assert boundaries.tolist() == [0, k + 1, k + 2, k + 2]


@pytest.mark.parametrize("method", ["mtp", "dspark"])
@pytest.mark.parametrize("requests", [1, 4])
@pytest.mark.parametrize("prefix", [False, True])
def test_prefix_tier_reserves_all_speculative_target_states(method, requests, prefix):
    path = Path(__file__).resolve().parents[3] / "vllm_ascend/_310p/model_runner_310p.py"
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "NPUModelRunner310")
    init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
    fields = {"supports_prefix_mamba_state_tier", "supports_compact_mamba_state", "num_compact_mamba_blocks"}
    assignments = [
        n
        for n in init.body
        if isinstance(n, ast.Assign) and any(isinstance(t, ast.Attribute) and t.attr in fields for t in n.targets)
    ]
    assert len(assignments) == len(fields)
    runner = SimpleNamespace(
        cache_config=SimpleNamespace(enable_prefix_caching=prefix, mamba_cache_mode="align"),
        model_config=SimpleNamespace(hf_text_config=SimpleNamespace(model_type="qwen4_exp_text")),
        speculative_config=SimpleNamespace(method=method, num_speculative_tokens=2),
        scheduler_config=SimpleNamespace(
            scheduler_cls="vllm_ascend.core.qwen_prefill_scheduler.QwenPrefillPacedScheduler"
        ),
        vllm_config=SimpleNamespace(additional_config={}),
        max_num_reqs=requests,
        _qwen4exp_speculative_state=True,
        _qwen4exp_mtp_ple=method == "mtp",
    )
    scope = dict(
        self=runner,
        PREFIX_MAMBA_MIN_SLOTS=PREFIX_MAMBA_MIN_SLOTS,
        prefix_mamba_slot_count=prefix_mamba_slot_count,
        supports_compact_live_mamba_state=supports_compact_live_mamba_state,
        uses_bounded_prefix_mamba_scheduler=uses_bounded_prefix_mamba_scheduler,
    )
    exec(compile(ast.Module(body=assignments, type_ignores=[]), str(path), "exec"), scope)
    assert runner.supports_prefix_mamba_state_tier == prefix
    assert runner.supports_compact_mamba_state
    expected = prefix_mamba_slot_count(requests, 2, bounded_checkpoints=True) if prefix else 3
    assert runner.num_compact_mamba_blocks == expected
