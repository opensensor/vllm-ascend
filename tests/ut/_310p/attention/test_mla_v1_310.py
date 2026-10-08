from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from vllm_ascend._310p.attention.mla_v1_310 import (
    AscendMLABackend310,
    AscendMLAImpl310,
    AscendMLAMetadataBuilder310,
    _grouped_absorbed_key_projection,
    _qsa_cache_block_table,
    _write_nz_latent_cache,
)
from vllm_ascend.attention.mla_v1 import DecodeMLAPreprocessResult


def test_qsa_block_table_maps_split_kernel_pages_to_shared_scheduler_pages() -> None:
    expanded = torch.tensor(
        [
            [36 + offset for offset in range(12)] + [84 + offset for offset in range(12)],
            [-1] * 12 + [120 + offset for offset in range(12)],
        ],
        dtype=torch.int32,
    )
    torch.testing.assert_close(
        _qsa_cache_block_table(expanded, 384),
        torch.tensor([[3, 7], [-1, 10]], dtype=torch.int32),
    )
    assert _qsa_cache_block_table(expanded, 32) is expanded
    with pytest.raises(ValueError, match="incompatible"):
        _qsa_cache_block_table(expanded, 48)


def test_native_mla_impl_retains_glm_kpool_indexer() -> None:
    indexer = SimpleNamespace(index_kpool=4)

    def set_minimum_parent_state(impl, *args, **kwargs):
        impl.enable_mlapo = False
        impl.qk_rope_head_dim = 0

    with patch(
        "vllm_ascend._310p.attention.mla_v1_310.AscendMLAImpl.__init__",
        set_minimum_parent_state,
    ):
        impl = AscendMLAImpl310(indexer=indexer)
    assert impl.glm_indexer is indexer


def test_grouped_absorbed_key_projection_flattens_heads_for_gmm() -> None:
    query = torch.randn(2, 3, 4)
    weight = torch.randn(3, 4, 5)
    group_list = torch.tensor([2, 4, 6])
    grouped_output = torch.randn(6, 5)

    with patch(
        "vllm_ascend._310p.attention.mla_v1_310.torch_npu.npu_grouped_matmul",
        return_value=[grouped_output],
    ) as grouped_matmul:
        actual = _grouped_absorbed_key_projection(query, weight, group_list)

    args, kwargs = grouped_matmul.call_args
    assert args == ()
    assert kwargs["x"][0].shape == (6, 4)
    assert kwargs["x"][0].is_contiguous()
    assert kwargs["weight"] == [weight]
    assert kwargs["group_list"] is group_list
    assert kwargs["split_item"] == 2
    assert kwargs["group_type"] == 0
    torch.testing.assert_close(actual, grouped_output.view(3, 2, 5).transpose(0, 1))


def test_native_mla_backend_publishes_nz_latent_cache_shape() -> None:
    assert AscendMLABackend310.get_kv_cache_shape(8, 32, 1, 512) == (
        8,
        32,
        32,
        16,
    )
    with pytest.raises(ValueError, match="divisible by 16"):
        AscendMLABackend310.get_kv_cache_shape(8, 32, 1, 510)


def test_metadata_builder_reuses_one_native_prefill_mask() -> None:
    builder = AscendMLAMetadataBuilder310.__new__(AscendMLAMetadataBuilder310)
    builder.device = torch.device("cpu")
    builder._native_prefill_mask = None
    builder._native_prefill_mask_size = 0
    first_native = torch.ones(1)
    second_native = torch.ones(2)

    with (
        patch(
            "vllm_ascend._310p.attention.mla_v1_310.AttentionMaskBuilder310.gen_causal_additive_mask",
            side_effect=(torch.zeros(32, 32), torch.zeros(64, 64)),
        ) as generate,
        patch(
            "vllm_ascend._310p.attention.mla_v1_310.torch_npu.npu_format_cast",
            side_effect=(first_native, second_native),
        ),
    ):
        assert builder._get_native_prefill_mask(31) is first_native
        assert builder._get_native_prefill_mask(32) is first_native
        assert builder._get_native_prefill_mask(33) is second_native

    assert generate.call_count == 2


def test_metadata_builder_keeps_replay_updated_decode_lengths() -> None:
    builder = AscendMLAMetadataBuilder310.__new__(AscendMLAMetadataBuilder310)
    builder.num_decodes = 2
    metadata = SimpleNamespace(seq_lens=torch.tensor([100, 100]))
    common = SimpleNamespace(seq_lens=torch.tensor([7, 11, 0], dtype=torch.int32))

    with patch(
        "vllm_ascend._310p.attention.mla_v1_310.AscendMLAMetadataBuilder.build_decode_metadata",
        return_value=metadata,
    ):
        actual = builder.build_decode_metadata(0, common)

    assert actual is metadata
    assert torch.equal(actual.seq_lens, common.seq_lens[:2])


def test_native_prefill_uses_host_lengths_and_310p_flash_attention() -> None:
    impl = AscendMLAImpl310.__new__(AscendMLAImpl310)
    impl.num_heads = 2
    impl.v_head_dim = 4
    impl.scale = 0.5
    mask = torch.zeros(16, 16, dtype=torch.float16)

    query = torch.randn(6, 2, 4, dtype=torch.float16)
    key = torch.randn_like(query)
    value = torch.randn_like(query)
    metadata = SimpleNamespace(
        prefill=SimpleNamespace(
            actual_seq_lengths_q=[2, 5],
            chunked_context=None,
            attn_mask=mask,
            query_lens=torch.tensor([2, 3], dtype=torch.int32),
        )
    )

    def flash_attention(**kwargs) -> None:
        kwargs["out"].copy_(kwargs["value"])

    with (
        patch("vllm_ascend._310p.attention.mla_v1_310.record_attention_compute_start"),
        patch(
            "vllm_ascend._310p.attention.mla_v1_310.torch_npu._npu_flash_attention",
            side_effect=flash_attention,
        ) as flash,
    ):
        actual = impl._forward_prefill_naive(
            query,
            torch.empty(6, 2, 0),
            key,
            torch.empty(6, 2, 0),
            value,
            (torch.empty(0), torch.empty(0)),
            metadata,
        )

    kwargs = flash.call_args.kwargs
    assert kwargs["seq_len"].device.type == "cpu"
    assert kwargs["seq_len"].tolist() == [2, 4]
    assert kwargs["mask"] is mask
    assert kwargs["num_heads"] == 2
    assert kwargs["num_kv_heads"] == 2
    torch.testing.assert_close(actual, value.reshape(6, 8))


def test_native_prefill_rejects_unmerged_chunked_context() -> None:
    impl = AscendMLAImpl310.__new__(AscendMLAImpl310)
    metadata = SimpleNamespace(
        prefill=SimpleNamespace(
            actual_seq_lengths_q=[1],
            chunked_context=SimpleNamespace(seq_tot=[1]),
        )
    )

    with pytest.raises(NotImplementedError, match="chunked-prefill context merge"):
        impl._forward_prefill_naive(
            torch.empty(1, 1, 1),
            torch.empty(1, 1, 0),
            torch.empty(1, 1, 1),
            torch.empty(1, 1, 0),
            torch.empty(1, 1, 1),
            (torch.empty(0), torch.empty(0)),
            metadata,
        )


@pytest.mark.parametrize("position_padding", [0, 1, 2])
def test_continued_prefill_uses_visible_paged_latent_prefix(position_padding: int) -> None:
    impl = AscendMLAImpl310.__new__(AscendMLAImpl310)
    impl.host_kv_layer = None
    impl.W_UK_T = torch.eye(4, dtype=torch.float16).expand(2, -1, -1).contiguous()
    impl._decode_constant_buffers = {}
    impl.scale = 0.5
    impl._v_up_proj = lambda value: value.transpose(0, 1).reshape(4, 8)
    query = torch.randn(5, 2, 4, dtype=torch.float16)
    cache = torch.empty(3, 1, 32, 16, dtype=torch.float16)
    metadata = SimpleNamespace(
        prefill=SimpleNamespace(
            chunked_context=SimpleNamespace(seq_tot=[6]),
            input_positions=torch.tensor([5, 6, 1, 2] + [0] * position_padding),
            actual_seq_lengths_q=[2, 4],
            max_seq_lens=7,
            block_table=torch.tensor([[2, 1], [0, 2]], dtype=torch.int32),
            query_start_loc=torch.tensor([0, 2, 4], dtype=torch.int32),
        )
    )
    captured = {}

    def paged_latent_op(*args):
        captured["args"] = args
        return args[0].clone()

    impl._get_paged_latent_op = lambda: paged_latent_op
    with patch("vllm_ascend._310p.attention.mla_v1_310.record_attention_compute_start"):
        actual = impl._forward_prefill(
            query,
            torch.empty(5, 2, 0),
            query,
            torch.empty(5, 2, 0),
            query,
            (cache, cache),
            metadata,
        )

    args = captured["args"]
    assert args[1] is cache and args[2] is cache
    assert args[4].tolist() == [6, 7, 2, 3]
    assert args[6].tolist() == [-1] * 4
    torch.testing.assert_close(args[7], metadata.prefill.block_table)
    torch.testing.assert_close(args[8], metadata.prefill.query_start_loc)
    assert args[9:] == (0.5, 4)
    torch.testing.assert_close(actual[:4], query[:4].reshape(4, 8))
    assert torch.count_nonzero(actual[4]) == 0


def test_continued_prefill_rejects_short_query() -> None:
    impl = AscendMLAImpl310.__new__(AscendMLAImpl310)
    metadata = SimpleNamespace(
        prefill=SimpleNamespace(
            input_positions=torch.zeros(4, dtype=torch.int32),
            actual_seq_lengths_q=[4],
        )
    )
    with pytest.raises(RuntimeError, match="shorter than the actual prefill"):
        impl._forward_prefill_paged_latent(
            torch.empty(3, 1, 4, dtype=torch.float16),
            (torch.empty(0), torch.empty(0)),
            metadata,
        )


def test_write_nz_latent_cache_maps_slots_and_ignores_padding() -> None:
    cache = torch.zeros(3, 2, 4, 16, dtype=torch.float16)
    rows = torch.arange(4 * 32, dtype=torch.float16).view(4, 2, 16)
    slots = torch.tensor([0, 7, -1, 9], dtype=torch.int64)

    _write_nz_latent_cache(cache, rows, slots)

    torch.testing.assert_close(cache[0, :, 0], rows[0])
    torch.testing.assert_close(cache[1, :, 3], rows[1])
    torch.testing.assert_close(cache[2, :, 1], rows[3])
    assert torch.count_nonzero(cache) == torch.count_nonzero(rows[[0, 1, 3]])


def test_nope_cache_write_stores_latent_once_in_aliased_native_pages() -> None:
    impl = AscendMLAImpl310.__new__(AscendMLAImpl310)
    impl.host_kv_layer = None
    impl.kv_lora_rank = 32
    impl.kv_a_layernorm = lambda value: value + 1
    source = torch.arange(64, dtype=torch.float16).view(2, 1, 1, 32)
    key_cache = torch.zeros(1, 2, 4, 16, dtype=torch.float16)
    slots = torch.tensor([0, 3], dtype=torch.int64)

    empty_rope, latent = impl._exec_kv_mla_nope(
        source,
        (key_cache, key_cache),
        slots,
        is_prefill=True,
    )

    expected = source + 1
    torch.testing.assert_close(key_cache[0, :, 0], expected[0].reshape(2, 16))
    torch.testing.assert_close(key_cache[0, :, 3], expected[1].reshape(2, 16))
    assert empty_rope.shape == (2, 1, 1, 0)
    torch.testing.assert_close(latent, expected)


def test_native_decode_accepts_shared_mla_forward_interface() -> None:
    impl = AscendMLAImpl310.__new__(AscendMLAImpl310)
    query = torch.randn(1, 2, 4)
    key_cache = torch.empty(2, 1, 32, 16)
    value_cache = torch.empty_like(key_cache)
    metadata = SimpleNamespace(decode=object())
    result = DecodeMLAPreprocessResult(
        ql_nope=query,
        q_pe=torch.empty(1, 2, 0),
        k_nope=key_cache,
        k_pe=value_cache,
    )
    expected = torch.randn(1, 2, 4)
    calls = []

    def fused(*args):
        calls.append(args)
        return expected

    impl._forward_decode_fused = fused

    assert impl._forward_decode(result, 32, metadata) is expected
    assert calls == [(query, key_cache, value_cache, metadata)]


def test_native_decode_rejects_missing_latent_cache() -> None:
    impl = AscendMLAImpl310.__new__(AscendMLAImpl310)
    result = DecodeMLAPreprocessResult(ql_nope=torch.randn(1, 2, 4))

    with pytest.raises(ValueError, match="requires query and latent KV cache"):
        impl._forward_decode(result, 32, SimpleNamespace())


def test_fused_decode_uses_constant_size_dense_prefix_metadata() -> None:
    impl = AscendMLAImpl310.__new__(AscendMLAImpl310)
    impl.host_kv_layer = None
    impl.scale = 0.25
    impl._decode_constant_buffers = {}
    impl._v_up_proj = lambda value: value
    query = torch.randn(2, 2, 16, dtype=torch.float16)
    key_cache = torch.empty(4, 1, 32, 16, dtype=torch.float16)
    value_cache = torch.empty_like(key_cache)
    block_table = torch.tensor([[3, 1, 0], [2, 0, 1]], dtype=torch.int32)
    metadata = SimpleNamespace(
        num_decodes=2,
        decode=SimpleNamespace(
            seq_lens=torch.tensor([9, 6], dtype=torch.int32),
            block_table=block_table,
        ),
    )
    captured = {}

    def paged_latent_op(*args):
        captured["args"] = args
        return args[0].clone()

    impl._get_paged_latent_op = lambda: paged_latent_op
    actual = impl._forward_decode_fused(
        query,
        key_cache,
        value_cache,
        metadata,
    )

    args = captured["args"]
    assert args[1] is key_cache
    assert args[2] is value_cache
    assert args[3].shape == (2, 1)
    assert torch.equal(args[4], metadata.decode.seq_lens)
    assert args[4].data_ptr() == metadata.decode.seq_lens.data_ptr()
    assert args[4].tolist() == [9, 6]
    assert args[5].tolist() == [0, 0]
    assert args[6].tolist() == [-1, -1]
    assert torch.equal(args[7], block_table)
    assert args[8].tolist() == [0, 1, 2]
    assert args[9:] == (0.25, 4, 1)
    torch.testing.assert_close(actual, query.transpose(0, 1))


def test_glm_kpool_decode_passes_selected_pools_and_tail_to_qsa() -> None:
    impl = AscendMLAImpl310.__new__(AscendMLAImpl310)
    impl.host_kv_layer = None
    impl.scale = 0.25
    impl._decode_constant_buffers = {}
    impl._v_up_proj = lambda value: value
    selected_tokens = torch.full((2, 12), -1, dtype=torch.int32)
    selected_tokens[0, :8] = torch.tensor([8, 9, 10, 11, 0, 1, 2, 3])
    selected_tokens[1, :4] = torch.tensor([0, 1, 2, 3])
    impl.glm_indexer = SimpleNamespace(
        index_kpool=4,
        topk_tokens=8,
        topk_indices_buffer=selected_tokens,
    )
    query = torch.randn(2, 2, 16, dtype=torch.float16)
    cache = torch.empty(4, 1, 32, 16, dtype=torch.float16)
    metadata = SimpleNamespace(
        num_decodes=2,
        decode=SimpleNamespace(
            seq_lens=torch.tensor([14, 7], dtype=torch.int32),
            input_positions=torch.tensor([13, 6], dtype=torch.int32),
            block_table=torch.tensor([[0, 1, 2, 3], [3, 2, 1, 0]], dtype=torch.int32),
        ),
    )
    captured = {}

    def paged_latent_op(*args):
        captured["args"] = args
        return args[0].clone()

    impl._get_paged_latent_op = lambda: paged_latent_op
    impl._forward_decode_fused(query, cache, cache, metadata)
    args = captured["args"]
    assert args[3].tolist() == [[2, 0], [0, -1]]
    assert args[4].tolist() == [2, 7]
    assert args[5].tolist() == [12, 4]
    assert args[6].tolist() == [2, -1]


def test_fused_decode_rejects_non_glm_multi_token_request() -> None:
    impl = AscendMLAImpl310.__new__(AscendMLAImpl310)
    impl.glm_indexer = None
    query = torch.randn(2, 2, 4, dtype=torch.float16)
    cache = torch.empty(4, 1, 4, 16, dtype=torch.float16)
    metadata = SimpleNamespace(
        num_decodes=1,
        decode=SimpleNamespace(
            seq_lens=torch.tensor([9], dtype=torch.int32),
            block_table=torch.tensor([[3, 1, 0]], dtype=torch.int32),
        ),
    )

    with pytest.raises(NotImplementedError, match="one decode token per request"):
        impl._forward_decode_fused(query, cache, cache, metadata)


def test_decomposed_decode_gathers_paged_latent_and_masks_context() -> None:
    impl = AscendMLAImpl310.__new__(AscendMLAImpl310)
    impl.num_heads = 2
    impl.kv_lora_rank = 4
    impl.scale = 0.5
    impl._v_up_proj = lambda value: value

    cache = torch.arange(4 * 2 * 4, dtype=torch.float32).view(4, 2, 1, 4)
    query = torch.tensor(
        [
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
            [[0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
        ]
    )
    decode = SimpleNamespace(
        block_table=torch.tensor([[0, 2], [1, 3]], dtype=torch.int32),
        seq_lens=torch.tensor([3, 2], dtype=torch.int32),
        seq_lens_list=[3, 2],
    )
    metadata = SimpleNamespace(decode=decode)

    actual = impl._forward_decode_naive(
        query,
        torch.empty(2, 2, 0),
        cache,
        torch.empty(0),
        2,
        metadata,
    )

    request_keys = torch.stack(
        (
            torch.stack((cache[0, 0, 0], cache[0, 1, 0], cache[2, 0, 0])),
            torch.stack((cache[1, 0, 0], cache[1, 1, 0], torch.zeros(4))),
        )
    )
    scores = torch.bmm(query, request_keys.transpose(1, 2)) * impl.scale
    scores[1, :, 2] = float("-inf")
    expected = torch.bmm(torch.softmax(scores, dim=-1), request_keys).transpose(0, 1)

    torch.testing.assert_close(actual, expected)


def test_value_up_projection_uses_head_batched_matmul() -> None:
    impl = AscendMLAImpl310.__new__(AscendMLAImpl310)
    impl.num_heads = 2
    impl.kv_lora_rank = 3
    impl.v_head_dim = 2
    impl.W_UV = torch.arange(2 * 3 * 2, dtype=torch.float32).view(2, 3, 2)
    latent = torch.arange(2 * 4 * 3, dtype=torch.float32).view(2, 4, 3)

    actual = impl._v_up_proj(latent)
    expected = torch.bmm(latent, impl.W_UV).transpose(0, 1).reshape(4, 4)

    torch.testing.assert_close(actual, expected)


def test_glm_mtp_decode_passes_causal_rows_and_device_request_boundaries():
    impl = AscendMLAImpl310.__new__(AscendMLAImpl310)
    impl.host_kv_layer = None
    impl.scale = 0.25
    impl._decode_constant_buffers = {}
    impl._v_up_proj = lambda value: value
    impl.glm_indexer = SimpleNamespace(
        index_kpool=4, topk_tokens=8, topk_indices_buffer=torch.zeros(4, 12, dtype=torch.int32)
    )
    query = torch.randn(4, 2, 16, dtype=torch.float16)
    cache = torch.empty(4, 1, 32, 16, dtype=torch.float16)
    boundaries = torch.tensor([0, 2, 4], dtype=torch.int32)
    table = torch.tensor([[0, 1], [2, 3]], dtype=torch.int32)
    metadata = SimpleNamespace(
        num_decodes=2,
        query_start_loc=boundaries,
        decode=SimpleNamespace(
            seq_lens=torch.tensor([7, 6], dtype=torch.int32),
            input_positions=torch.tensor([5, 6, 4, 5]),
            block_table=table,
        ),
    )
    captured = []
    impl._get_paged_latent_op = lambda: lambda *args: captured.append(args) or query.clone()
    impl._forward_decode_fused(query, cache, cache, metadata)
    args = captured[0]
    assert args[4].tolist() == [6, 7, 5, 6]
    assert args[7].shape[0] == 2
    assert args[8].data_ptr() == boundaries.data_ptr()
    assert args[8].tolist() == [0, 2, 4]
