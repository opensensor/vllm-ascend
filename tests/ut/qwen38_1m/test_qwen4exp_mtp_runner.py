# SPDX-License-Identifier: Apache-2.0
"""Host-only tests for the 310P Qwen4Exp MTP runner inputs."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm_ascend._310p.qwen4exp_mtp import (
    is_qwen4exp_mtp_config,
    qwen4exp_mtp_hidden_width,
    stage_ple_history,
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
