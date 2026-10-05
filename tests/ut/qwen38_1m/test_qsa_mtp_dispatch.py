# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Device-routed W4/W8 MTP dispatch and precreated graph metadata."""

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from vllm_ascend.models.qwen4_exp.dtype_policy import ASCEND_QWEN4EXP_DTYPE_POLICY
from vllm_ascend.models.qwen4_exp.model import (
    _QSA_PREFILL_BATCHED_GATHER,
    _QSA_PREFILL_GROUP_MAJOR_UNION,
    _qsa_prefill_policy,
    _qsa_selection_policies,
    _qsa_step_selection_policy,
    _QSAAttention,
)
from vllm_ascend.models.qwen4_exp.ops.qsa_batched_attention_310 import QSAPrefillGatherStreams, _request_slices
from vllm_ascend.models.qwen4_exp.ops.qsa_indexer import (
    QSA_SELECTION_FAST_TOPK,
    QSA_SELECTION_STABLE_ARGSORT,
    _use_qsa_matmul_score,
    _use_qsa_matmul_score_batch,
)
from vllm_ascend.models.qwen4_exp.w4_moe import CUBE_DEVICE_ROUTED_BACKENDS, FORMAT


def test_qsa_selection_policy_defaults_to_bounded_prefill_and_exact_decode() -> None:
    expected = (QSA_SELECTION_STABLE_ARGSORT, QSA_SELECTION_FAST_TOPK)
    assert _qsa_selection_policies(SimpleNamespace()) == expected
    config = SimpleNamespace(ascend_qsa_selection={})
    assert _qsa_selection_policies(config) == expected


def test_qsa_selection_policy_accepts_explicit_fast_topk() -> None:
    config = SimpleNamespace(ascend_qsa_selection={"policy": QSA_SELECTION_FAST_TOPK})
    assert _qsa_selection_policies(config) == (QSA_SELECTION_FAST_TOPK, QSA_SELECTION_FAST_TOPK)


def test_qsa_selection_policy_accepts_explicit_prefill_override() -> None:
    config = SimpleNamespace(
        ascend_qsa_selection={
            "policy": QSA_SELECTION_FAST_TOPK,
            "prefill_policy": QSA_SELECTION_STABLE_ARGSORT,
        }
    )
    assert _qsa_selection_policies(config) == (QSA_SELECTION_FAST_TOPK, QSA_SELECTION_STABLE_ARGSORT)


def test_qsa_step_selection_policy_uses_prefill_policy_for_mixed_step() -> None:
    decode_policy = QSA_SELECTION_STABLE_ARGSORT
    prefill_policy = QSA_SELECTION_FAST_TOPK
    assert (
        _qsa_step_selection_policy(decode_policy, prefill_policy, SimpleNamespace(num_prefills=1, num_decodes=0))
        == prefill_policy
    )
    assert (
        _qsa_step_selection_policy(decode_policy, prefill_policy, SimpleNamespace(num_prefills=1, num_decodes=1))
        == prefill_policy
    )
    assert (
        _qsa_step_selection_policy(decode_policy, prefill_policy, SimpleNamespace(num_prefills=0, num_decodes=2))
        == decode_policy
    )


@pytest.mark.parametrize(
    "metadata",
    [
        QSA_SELECTION_FAST_TOPK,
        {"policy": [QSA_SELECTION_FAST_TOPK]},
        {"prefill_policy": [QSA_SELECTION_FAST_TOPK]},
        {"policy": "unvalidated"},
        {"policy": QSA_SELECTION_STABLE_ARGSORT, "extra": True},
    ],
)
def test_qsa_selection_policy_rejects_invalid_metadata(metadata) -> None:
    config = SimpleNamespace(ascend_qsa_selection=metadata)
    with pytest.raises(ValueError, match="ascend_qsa_selection|QSA selection policy"):
        _qsa_selection_policies(config)


def test_qsa_prefill_policy_defaults_to_retained_backend() -> None:
    assert _qsa_prefill_policy(SimpleNamespace()) == (_QSA_PREFILL_BATCHED_GATHER, 8, True)
    config = SimpleNamespace(ascend_qsa_prefill={})
    assert _qsa_prefill_policy(config) == (_QSA_PREFILL_BATCHED_GATHER, 8, True)


def test_qsa_prefill_policy_accepts_explicit_group_major_tile() -> None:
    config = SimpleNamespace(ascend_qsa_prefill={"backend": _QSA_PREFILL_GROUP_MAJOR_UNION, "query_tile": 16})
    assert _qsa_prefill_policy(config) == (_QSA_PREFILL_GROUP_MAJOR_UNION, 16, True)


def test_qsa_prefill_can_disable_parallel_gather_for_service_comparison() -> None:
    config = SimpleNamespace(ascend_qsa_prefill={"parallel_gather": False})
    assert _qsa_prefill_policy(config) == (_QSA_PREFILL_BATCHED_GATHER, 8, False)


@pytest.mark.parametrize(
    "metadata",
    [
        _QSA_PREFILL_GROUP_MAJOR_UNION,
        {"backend": "unknown"},
        {"query_tile": 0},
        {"query_tile": 17},
        {"query_tile": True},
        {"parallel_gather": 0},
        {"parallel_gather": "false"},
        {"extra": True},
    ],
)
def test_qsa_prefill_policy_rejects_invalid_metadata(metadata) -> None:
    with pytest.raises(ValueError, match="ascend_qsa_prefill|QSA prefill|query_tile"):
        _qsa_prefill_policy(SimpleNamespace(ascend_qsa_prefill=metadata))


@pytest.mark.parametrize("backend", [None, "eager_dequant", "cube_310", "cube_310_tiled", *CUBE_DEVICE_ROUTED_BACKENDS])
@pytest.mark.parametrize("tp_size", [1, 2, 4])
def test_batched_qsa_limit_and_group_list_cover_w8_and_routed_w4(backend, tp_size):
    config = SimpleNamespace(
        hidden_size=256,
        moe_intermediate_size=256,
        num_attention_heads=8,
        num_key_value_heads=2,
        head_dim=16,
        indexer_n_heads=2,
        indexer_head_dim=16,
        indexer_budget=2048,
        indexer_compress_ratio=4,
    )
    if backend is not None:
        config.ascend_expert_quantization = {
            "format": FORMAT,
            "bits": 4,
            "symmetric": False,
            "group_size": 128,
            "packing": "signed_int4_low_nibble_first_in_axis",
            "backend": backend,
            "activation_quantization": "int8_per_group" if backend == "cube_310_int4_a8" else "float16",
            "scale_dtype": "float16",
            "offset_dtype": "int8",
        }
    module = _QSAAttention(
        config=config, layer_idx=0, dtype_policy=ASCEND_QWEN4EXP_DTYPE_POLICY, expert_sharding=(0, tp_size)
    )
    limit = 8 if backend is None or backend in CUBE_DEVICE_ROUTED_BACKENDS else 2
    assert module.reuse_query_rope is (limit == 8)
    assert module.qsa_selection_policy == QSA_SELECTION_STABLE_ARGSORT
    assert module.qsa_prefill_selection_policy == QSA_SELECTION_FAST_TOPK
    assert module.qsa_prefill_backend == _QSA_PREFILL_BATCHED_GATHER
    assert module.qsa_group_major_query_tile == 8
    assert module._batched_qsa_max_decode_tokens == limit
    expected = torch.arange(1, limit * module.num_kv_heads + 1, dtype=torch.int64)
    expected *= module.num_heads // module.num_kv_heads
    torch.testing.assert_close(module._qsa_decode_group_list, expected)
    assert "_qsa_decode_group_list" not in module.state_dict()
    metadata = SimpleNamespace(num_decodes=1, num_prefills=0, query_lens_cpu=None)
    for tokens in (1, 2, 3, 5, 8, 9):
        assert module._can_use_batched_qsa_decode(metadata, tokens, 512) == (tokens <= limit)
        assert not module._can_use_batched_qsa_decode(metadata, tokens, 255)
        assert _use_qsa_matmul_score(tokens, 5856, 1, 2, module._batched_qsa_max_decode_tokens) == (tokens <= limit)
    assert module._can_use_batched_qsa_decode(metadata, limit, 256)
    for prefills, decodes in ((1, 0), (1, 1), (0, 0)):
        metadata.num_prefills, metadata.num_decodes = prefills, decodes
        assert not module._can_use_batched_qsa_decode(metadata, 2, 512)


def test_batched_qsa_decode_uses_largest_per_request_query() -> None:
    config = SimpleNamespace(
        hidden_size=256,
        moe_intermediate_size=256,
        num_attention_heads=8,
        num_key_value_heads=2,
        head_dim=16,
        indexer_n_heads=2,
        indexer_head_dim=16,
        indexer_budget=2048,
        indexer_compress_ratio=4,
    )
    module = _QSAAttention(
        config=config,
        layer_idx=0,
        dtype_policy=ASCEND_QWEN4EXP_DTYPE_POLICY,
        expert_sharding=(0, 4),
    )
    metadata = SimpleNamespace(
        num_decodes=4,
        num_prefills=0,
        query_lens_cpu=torch.tensor([3, 3, 3, 3], dtype=torch.int32),
    )
    assert module._can_use_batched_qsa_decode(metadata, 12, 512)
    metadata.query_lens_cpu[-1] = 9
    assert not module._can_use_batched_qsa_decode(metadata, 18, 512)


def test_batched_qsa_prefill_requires_host_request_boundaries() -> None:
    metadata = SimpleNamespace(block_tables=torch.zeros((4, 8), dtype=torch.int32), query_lens_cpu=None)
    assert not _QSAAttention._has_qsa_request_boundaries(metadata)
    metadata.query_lens_cpu = torch.tensor([512, 512, 512, 512], dtype=torch.int32)
    assert _QSAAttention._has_qsa_request_boundaries(metadata)
    metadata.query_lens_cpu = metadata.query_lens_cpu[:3]
    assert not _QSAAttention._has_qsa_request_boundaries(metadata)
    metadata.block_tables = metadata.block_tables[:1]
    assert not _QSAAttention._has_qsa_request_boundaries(metadata)
    metadata.query_lens_cpu = None
    assert _QSAAttention._has_qsa_request_boundaries(metadata)


def test_batched_qsa_group_list_covers_maximum_request_batch() -> None:
    config = SimpleNamespace(
        hidden_size=256,
        moe_intermediate_size=256,
        num_attention_heads=8,
        num_key_value_heads=2,
        head_dim=16,
        indexer_n_heads=2,
        indexer_head_dim=16,
        indexer_budget=2048,
        indexer_compress_ratio=4,
    )
    runtime_config = SimpleNamespace(
        device_config=SimpleNamespace(device=torch.device("cpu")),
        scheduler_config=SimpleNamespace(max_num_seqs=4),
    )
    with (
        patch("vllm_ascend.models.qwen4_exp.model.get_current_vllm_config_or_none", return_value=runtime_config),
        patch("vllm_ascend.models.qwen4_exp.model._resolve_attn_backend", return_value=None),
    ):
        module = _QSAAttention(
            config=config,
            layer_idx=0,
            dtype_policy=ASCEND_QWEN4EXP_DTYPE_POLICY,
            expert_sharding=(0, 4),
        )
    expected_groups = module._batched_qsa_max_decode_tokens * 4 * module.num_kv_heads
    assert module._qsa_decode_group_list.numel() == expected_groups
    assert module._qsa_decode_group_list[-1] == expected_groups * (module.num_heads // module.num_kv_heads)


def test_multi_request_qsa_matmul_dispatch_uses_per_request_lengths() -> None:
    assert _use_qsa_matmul_score_batch(None, 128, 2048, 1, 2, 8)
    assert _use_qsa_matmul_score_batch([3, 3, 3, 3], 12, 5856, 4, 5, 8)
    # A decode request must not force a long prefill through the slow native
    # scorer when the shared visible cache is still below 2,048 groups.
    assert _use_qsa_matmul_score_batch([1920, 3], 1923, 1920, 2, 3, 8)
    assert not _use_qsa_matmul_score_batch([1920, 9], 1929, 1920, 2, 3, 8)
    assert not _use_qsa_matmul_score_batch([0, 0], 0, 5856, 2, 3, 8)
    assert not _use_qsa_matmul_score_batch([3, 9], 12, 5856, 2, 3, 8)
    assert not _use_qsa_matmul_score_batch([3, 3], 6, 1024, 2, 3, 8)
    with pytest.raises(ValueError, match="one length"):
        _use_qsa_matmul_score_batch([3], 6, 5856, 2, 3, 8)
    with pytest.raises(ValueError, match="sum"):
        _use_qsa_matmul_score_batch([3, 2], 6, 5856, 2, 3, 8)


def test_request_slices_validate_packed_query_boundaries() -> None:
    assert _request_slices([3, 1, 2], 6, 3) == ((0, 3), (3, 4), (4, 6))
    with pytest.raises(ValueError, match="one length"):
        _request_slices([3, 3], 6, 3)
    with pytest.raises(ValueError, match="sum"):
        _request_slices([3, 2], 6, 2)


def test_qsa_prefill_gather_streams_are_lazy_and_persistent() -> None:
    streams = QSAPrefillGatherStreams()
    assert streams._streams is None
    make_stream = Mock(side_effect=(object(), object()))
    with patch.object(torch, "npu", SimpleNamespace(Stream=make_stream), create=True):
        first = streams.get()
        second = streams.get()
    assert first is second
    assert make_stream.call_count == 2
