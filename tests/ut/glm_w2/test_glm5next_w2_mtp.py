# SPDX-License-Identifier: Apache-2.0
"""CPU contracts for the packed GLM MTP candidate; no device imports."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm_ascend.models.glm5next_w2.kda_310 import _promote_accepted_recurrent_state
from vllm_ascend.models.glm5next_w2.model import Glm5NextW2MTP, _prepare_dsa_indexer_weights
from vllm_ascend.models.glm5next_w2.mtp_config import normalize_glm_mtp_hf_config, validate_packed_glm_mtp


def config():
    return SimpleNamespace(
        speculative_config=SimpleNamespace(method="mtp", num_speculative_tokens=1),
        cache_config=SimpleNamespace(mamba_cache_mode="align"),
        scheduler_config=SimpleNamespace(async_scheduling=False, max_num_seqs=4),
        model_config=SimpleNamespace(enforce_eager=True, hf_config=SimpleNamespace()),
        parallel_config=SimpleNamespace(pipeline_parallel_size=1),
    )


def test_indexer_projection_copies_are_prepared_and_refreshed_after_loading():
    wk = nn.Parameter(torch.arange(24, dtype=torch.float16).reshape(6, 4))
    gate = nn.Parameter(torch.ones(4, 4, dtype=torch.bfloat16))
    indexer = SimpleNamespace(wk_weights_proj=SimpleNamespace(weight=wk), index_kpool_compress_gate=gate)
    layers = [SimpleNamespace(), SimpleNamespace(self_attn=SimpleNamespace(indexer=indexer))]
    _prepare_dsa_indexer_weights(layers)
    assert indexer._wk_weight_f32.dtype == torch.float32
    assert indexer._gate_weight_f32.dtype == torch.float32
    assert not indexer._wk_weight_f32.requires_grad
    torch.testing.assert_close(indexer._wk_weight_f32, wk.float())
    wk.data.fill_(7)
    gate.data.fill_(3)
    _prepare_dsa_indexer_weights(layers)
    assert torch.equal(indexer._wk_weight_f32, torch.full((6, 4), 7.0))
    assert torch.equal(indexer._gate_weight_f32, torch.full((4, 4), 3.0))


@pytest.mark.parametrize("packed", [True, False])
def test_normalization_flattens_without_mutating_target(packed):
    text = SimpleNamespace(model_type="glm5_next_text", num_nextn_predict_layers=1, hidden_size=4096)
    target = SimpleNamespace(text_config=text, ascend_glm_nz_packed_codes=True, quantization_config={"method": "fp8"})
    result = normalize_glm_mtp_hf_config(target, packed=packed)
    assert result is not text and text.model_type == "glm5_next_text"
    assert result.model_type == "glm5_next_mtp" and result.n_predict == 1
    assert result.architectures == ["Glm5NextW2MTPModel" if packed else "Glm5NextMTPModel"]
    assert result.ascend_glm_nz_packed_codes and result.hidden_size == 4096
    assert result.quantization_config == {"method": "fp8"}


@pytest.mark.parametrize("count", [None, 0, -1])
def test_normalization_rejects_absent_prediction_layer(count):
    with pytest.raises(ValueError, match="positive"):
        normalize_glm_mtp_hf_config(SimpleNamespace(num_nextn_predict_layers=count))


@pytest.mark.parametrize("drafts", [1, 2, 7])
def test_eager_aligned_candidate_is_admitted(drafts):
    candidate = config()
    candidate.speculative_config.num_speculative_tokens = drafts
    validate_packed_glm_mtp(candidate)


@pytest.mark.parametrize(
    "section,field,value,match",
    [
        ("speculative_config", "method", "ngram", "method"),
        ("speculative_config", "num_speculative_tokens", 8, "1..7"),
        ("cache_config", "mamba_cache_mode", "none", "align"),
        ("model_config", "enforce_eager", False, "graph replay"),
        ("scheduler_config", "async_scheduling", True, "synchronous"),
        ("parallel_config", "pipeline_parallel_size", 2, "pipeline"),
    ],
)
def test_unqualified_modes_fail_before_allocation(section, field, value, match):
    candidate = config()
    setattr(getattr(candidate, section), field, value)
    with pytest.raises(ValueError, match=match):
        validate_packed_glm_mtp(candidate)


@pytest.mark.parametrize("accepted", [1, 2, 3])
def test_recurrent_rollback_handles_shorter_queries_and_padding(accepted):
    state = torch.arange(12 * 4, dtype=torch.float16).reshape(12, 2, 2)
    saved = state.clone()
    # Requests have moved rows. Padding must neither read nor overwrite slot 0.
    indices = torch.tensor([[6, 7, 8], [-1, -1, -1], [0, 1, 2]], dtype=torch.int32)
    lengths = torch.tensor([1, 0, 2], dtype=torch.int32)
    _promote_accepted_recurrent_state(state, indices, lengths, torch.tensor([accepted, 0, 2]))
    torch.testing.assert_close(state[6], saved[6 + accepted - 1], rtol=0, atol=0)
    torch.testing.assert_close(state[0], saved[1], rtol=0, atol=0)
    for slot in [1, 2, 3, 4, 5, 7, 8, 9, 10, 11]:
        torch.testing.assert_close(state[slot], saved[slot], rtol=0, atol=0)


def test_recurrent_rollback_does_not_consume_rejected_suffix():
    state = torch.full((4, 1, 1), float("nan"), dtype=torch.float16)
    state[1] = 5
    _promote_accepted_recurrent_state(state, torch.tensor([[0, 1, 2, 3]]), torch.tensor([1]), torch.tensor([2]))
    assert state[0].item() == 5
    assert torch.isnan(state[2:]).all()


def test_missing_head_is_shared_by_identity_and_owned_head_preserved():
    model = Glm5NextW2MTP.__new__(Glm5NextW2MTP)
    nn.Module.__init__(model)
    model.model = nn.Module()
    layer = nn.Module()
    layer.shared_head = nn.Module()
    layer.shared_head.head = None
    model.model.layers = nn.ModuleDict({"45": layer})
    model.has_own_lm_head = False
    head = nn.Linear(4, 8, bias=False)
    assert model.share_target_lm_head_if_identical(SimpleNamespace(lm_head=head))
    assert layer.shared_head.head is head
    model.has_own_lm_head = True
    assert not model.share_target_lm_head_if_identical(SimpleNamespace(lm_head=nn.Linear(4, 8)))
    assert layer.shared_head.head is head
    model.has_own_lm_head = False
    with pytest.raises(ValueError, match="no lm_head"):
        model.share_target_lm_head_if_identical(SimpleNamespace())


def test_constructor_reuses_shipped_predictor_and_suppresses_fp8(monkeypatch):
    import sys
    from types import ModuleType

    from vllm_ascend import envs
    from vllm_ascend.models.glm5next_w2 import model as packed

    shipped = ModuleType("vllm_ascend.models.glm5next.model")
    original_factory = object()
    shipped.FusedMoEFactory = original_factory
    mtp = ModuleType("vllm_ascend.models.glm5next.mtp")

    class Predictor(nn.Module):
        def __init__(self, *, vllm_config, prefix):
            super().__init__()
            assert shipped.FusedMoEFactory is packed._NoFp8FusedMoEExperts
            assert prefix == "model"
            layer = nn.Module()
            layer.mtp_block = nn.Module()
            self.layers = nn.ModuleDict({"45": layer})
            self.num_mtp_layers = 1

    mtp.Glm5NextMultiTokenPredictor = Predictor
    monkeypatch.setitem(sys.modules, shipped.__name__, shipped)
    monkeypatch.setitem(sys.modules, mtp.__name__, mtp)
    monkeypatch.setattr(envs, "VLLM_ASCEND_310P_GLM_HOST_KV", False)
    monkeypatch.setattr(packed, "_install_dsa_indexer", lambda *a: 1)
    monkeypatch.setattr(packed, "_install_w2_moe", lambda *a: 1)
    candidate = config()
    candidate.quant_config = None
    candidate.model_config.hf_text_config = SimpleNamespace(ascend_glm_nz_packed_codes=True, num_nextn_predict_layers=1)
    model = Glm5NextW2MTP(vllm_config=candidate)
    assert model.model.layers["45"].use_310p_eh_norm
    assert model._nz_packed_codes
    assert shipped.FusedMoEFactory is original_factory


def test_recurrent_kernel_receives_full_slot_table_without_state_copy(monkeypatch):
    import sys

    from vllm_ascend.models.glm5next_w2 import kda_310

    monkeypatch.setitem(
        sys.modules, "vllm_ascend.ascend_forward_context", SimpleNamespace(_EXTRA_CTX=SimpleNamespace(capturing=False))
    )
    monkeypatch.setattr(kda_310, "_l2norm_310p", lambda value: value)
    monkeypatch.setattr(kda_310, "_safe_gate_for_layer", lambda layer, value: value)
    state = torch.arange(8 * 4, dtype=torch.float16).reshape(8, 1, 2, 2)
    before = state.clone()
    seen = []

    def kernel(**kwargs):
        seen.append(kwargs)
        torch.testing.assert_close(kwargs["state"], before, rtol=0, atol=0)
        return kwargs["value"]

    monkeypatch.setattr(torch.ops._C_ascend, "npu_recurrent_gated_delta_rule_310", kernel, raising=False)
    inputs = torch.ones(1, 3, 1, 2, dtype=torch.float16)
    kda_310._run_recurrent(
        SimpleNamespace(head_dim=2),
        inputs,
        inputs,
        inputs,
        inputs,
        torch.zeros(1, 3, 1),
        state,
        torch.tensor([0, 1, 3]),
        torch.tensor([[0, 1, 2], [4, 5, 6]]),
        num_sequences=2,
        num_accepted_tokens=torch.tensor([3, 2]),
    )
    assert seen[0]["ssm_state_indices"].tolist() == [[0, 1, 2], [4, 5, 6]]
    assert seen[0]["actual_seq_lengths"].tolist() == [1, 2]
    assert seen[0]["num_accepted_tokens"].tolist() == [3, 2]


def test_full_graph_candidate_requires_explicit_native_state_profile():
    candidate = config()
    candidate.model_config.enforce_eager = False
    candidate.model_config.hf_config.ascend_glm_mtp_full_graph = True
    validate_packed_glm_mtp(candidate)
