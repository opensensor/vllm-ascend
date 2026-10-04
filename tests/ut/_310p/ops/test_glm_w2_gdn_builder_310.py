# SPDX-License-Identifier: Apache-2.0
"""GLM W2 needs the 310P GDN builder's nested KDA metadata."""

from types import SimpleNamespace

import torch

from vllm_ascend._310p.ops.gdn_attn_builder_310 import (
    AscendGDNAttentionBackend310,
    GDNAttentionMetadataBuilder310,
    GlmW2GDNAttentionBackend310,
    GlmW2GDNAttentionMetadataBuilder310,
)
from vllm_ascend.models.glm5next_w2 import model as glm_w2_model
from vllm_ascend.models.glm5next_w2.dtype_policy import ASCEND_GLM5NEXT_W2_DTYPE_POLICY


def test_glm_w2_builder_is_separate_from_shared_310p_gdn():
    assert GlmW2GDNAttentionBackend310.get_builder_cls() is GlmW2GDNAttentionMetadataBuilder310
    assert AscendGDNAttentionBackend310.get_builder_cls() is GDNAttentionMetadataBuilder310
    assert GlmW2GDNAttentionMetadataBuilder310._USE_COMMON_KERNEL_METADATA
    assert not GDNAttentionMetadataBuilder310._USE_COMMON_KERNEL_METADATA


def test_glm_w2_kda_installer_selects_nested_metadata_builder(monkeypatch):
    monkeypatch.setattr(glm_w2_model, "_bind_eager_kda_forward", lambda *_args: None)
    self_attn = SimpleNamespace(
        in_proj_qkvbfg_a=object(),
        get_state_dtype=lambda: (torch.float16,) * 4,
    )
    layer = SimpleNamespace(layer_kind="kda", self_attn=self_attn)

    installed = glm_w2_model._install_310p_kda([layer], SimpleNamespace(), ASCEND_GLM5NEXT_W2_DTYPE_POLICY)

    assert installed == 1
    assert self_attn.get_attn_backend().get_builder_cls() is GlmW2GDNAttentionMetadataBuilder310


def test_glm_w2_decode_metadata_has_causal_conv_fields():
    builder = object.__new__(GlmW2GDNAttentionMetadataBuilder310)
    builder.use_full_cuda_graph = False
    cache_indices = torch.tensor([7], dtype=torch.int32)
    query_start_loc = torch.tensor([0, 1], dtype=torch.int32)
    metadata = SimpleNamespace(
        num_decodes=1,
        num_prefills=0,
        num_spec_decodes=0,
        non_spec_query_start_loc=query_start_loc,
        non_spec_decode_metadata=None,
    )

    builder._attach_non_spec_decode_metadata(metadata, cache_indices)

    conv = metadata.non_spec_decode_metadata.causal_conv1d
    assert conv.query_start_loc is query_start_loc
    assert conv.cache_indices.tolist() == [7]
    assert conv.initial_state_mode is None
    assert metadata.non_spec_decode_metadata.actual_seq_lengths.tolist() == [0, 1]


def test_glm_w2_prefill_metadata_has_chunk_and_conv_fields():
    builder = object.__new__(GlmW2GDNAttentionMetadataBuilder310)
    builder.vllm_config = SimpleNamespace(parallel_config=SimpleNamespace(prefill_context_parallel_size=1))
    cache_indices = torch.tensor([7], dtype=torch.int32)
    query_start_loc = torch.tensor([0, 4], dtype=torch.int32)
    has_initial_state = torch.tensor([False])
    chunk = SimpleNamespace(cu_seqlens_host=(0, 4), chunk_indices_chunk64_host=(0, 0), keep_meta=None)
    metadata = SimpleNamespace(
        num_prefills=1,
        non_spec_query_start_loc=query_start_loc,
        prefill_query_start_loc=query_start_loc,
        has_initial_state=has_initial_state,
        non_spec_prefill_metadata=None,
    )

    builder._attach_non_spec_prefill_metadata(metadata, chunk, cache_indices)

    prefill = metadata.non_spec_prefill_metadata
    assert prefill.chunk is chunk
    assert prefill.causal_conv1d.query_start_loc is query_start_loc
    assert prefill.causal_conv1d.cache_indices.tolist() == [7]
    assert prefill.causal_conv1d.initial_state_mode is has_initial_state
