# SPDX-License-Identifier: Apache-2.0
"""Native MTP carry selection, including changed inputs on ACL graph replay."""

import pytest
import torch
import torch_npu

from vllm_ascend.utils import enable_custom_op


def _state_with_page_guards(initial, paged):
    if not paged:
        return initial.npu(), None
    # Both an offset within each physical page and trailing space are shared
    # with other cache components. Neither may be touched by the state kernel.
    payload = initial[0].numel()
    guard = 256
    backing = torch.full((initial.shape[0], payload + 2 * guard), 7, dtype=initial.dtype, device="npu")
    state = backing[:, guard : guard + payload].view(initial.shape)
    state.copy_(initial)
    return state, backing


def _assert_page_guards(backing):
    if backing is not None:
        torch.testing.assert_close(backing[:, :256].cpu(), torch.full_like(backing[:, :256], 7).cpu(), rtol=0, atol=0)
        torch.testing.assert_close(backing[:, -256:].cpu(), torch.full_like(backing[:, -256:], 7).cpu(), rtol=0, atol=0)


@pytest.mark.parametrize("paged", [False, True])
def test_speculative_convolution_replays_acceptance_and_request_slots(paged):
    torch.npu.set_device(0)
    torch_npu.npu.set_compile_mode(jit_compile=False)
    enable_custom_op()
    torch.manual_seed(31049)
    tokens, channels, width = 6, 6144, 4
    inputs_cpu = torch.randn(tokens, channels).half() * 0.1
    weights_cpu = torch.randn(width, channels).half()
    initial = torch.randn(8, width, channels).half() * 0.1
    inputs, weights = inputs_cpu.npu(), weights_cpu.npu()
    state, backing = _state_with_page_guards(initial, paged)
    boundaries = torch.tensor([0, 2, 4, 6], dtype=torch.int32, device="npu")
    slots = torch.tensor([[0, 1], [2, 3], [4, 5]], dtype=torch.int32, device="npu")
    accepted = torch.tensor([2, 1, 2], dtype=torch.int32, device="npu")

    def run():
        return torch.ops._C_ascend.npu_causal_conv1d_310(
            inputs,
            weights,
            bias=None,
            conv_states=state,
            query_start_loc=boundaries,
            cache_indices=slots,
            initial_state_mode=None,
            num_accepted_tokens=accepted,
            activation_mode=1,
            pad_slot_id=-1,
            run_mode=1,
        )

    for _ in range(2):
        run()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        output = run()
    expected_state = initial.clone()
    state.copy_(initial)
    for lengths, indices, counts in [
        ([2, 2, 2], [[0, 1], [2, 3], [4, 5]], [2, 1, 2]),
        ([2, 2, 0], [[4, 5], [0, 1], [-1, -1]], [1, 2, 0]),
        ([2, 2, 2], [[2, 3], [4, 5], [0, 1]], [1, 2, 1]),
    ]:
        boundaries.copy_(torch.tensor([0] + lengths, dtype=torch.int32).cumsum(0))
        slots.copy_(torch.tensor(indices, dtype=torch.int32))
        accepted.copy_(torch.tensor(counts, dtype=torch.int32))
        expected_output = []
        start = 0
        for length, row, count in zip(lengths, indices, counts):
            if not length:
                continue
            slot = row[0]
            history = expected_state[slot, count - 1 : count + width - 2].clone()
            sequence = torch.cat((history, inputs_cpu[start : start + length]))
            for offset in range(length):
                value = (sequence[offset : offset + width].float() * weights_cpu.float()).sum(0)
                expected_output.append(torch.nn.functional.silu(value).half())
            expected_state[slot, : width - 2 + length] = sequence[1:]
            start += length
        graph.replay()
        torch.npu.synchronize()
        torch.testing.assert_close(output[:start].cpu(), torch.stack(expected_output), rtol=3e-3, atol=2e-3)
        torch.testing.assert_close(state.cpu(), expected_state, rtol=0, atol=0)
        _assert_page_guards(backing)


@pytest.mark.parametrize("paged", [False, True])
def test_speculative_kda_matches_independent_cpu_recurrence(paged):
    torch.npu.set_device(0)
    torch_npu.npu.set_compile_mode(jit_compile=False)
    enable_custom_op()
    torch.manual_seed(31048)
    tokens, heads, width = 6, 8, 128
    q = torch.nn.functional.normalize(torch.randn(tokens, heads, width), dim=-1).half()
    k = torch.nn.functional.normalize(torch.randn(tokens, heads, width), dim=-1).half()
    v = torch.randn(tokens, heads, width).half()
    beta = torch.rand(tokens, heads).half()
    gate = -torch.rand(tokens, heads, width)
    initial = torch.randn(6, heads, width, width).half()
    indices = [[0, 1], [2, 3], [4, 5]]
    accepted = [2, 1, 2]
    expected_state = initial.float().clone()
    expected_output = torch.empty_like(v)
    for batch, (slots, count) in enumerate(zip(indices, accepted)):
        carry = initial[slots[count - 1]].float().clone()
        for offset, slot in enumerate(slots):
            row = batch * 2 + offset
            carry = carry * gate[row].exp()[:, None, :]
            residual = (v[row].float() - (carry * k[row].float()[:, None, :]).sum(-1)) * beta[row].float()[:, None]
            carry = carry + residual[:, :, None] * k[row].float()[:, None, :]
            expected_output[row] = (carry * q[row].float()[:, None, :]).sum(-1) * width**-0.5
            expected_state[slot] = carry
    state, backing = _state_with_page_guards(initial, paged)
    output = torch.ops._C_ascend.npu_recurrent_gated_delta_rule_310(
        query=q.npu(),
        key=k.npu(),
        value=v.npu(),
        beta=beta.npu(),
        state=state,
        actual_seq_lengths=torch.tensor([2, 2, 2], dtype=torch.int32, device="npu"),
        ssm_state_indices=torch.tensor(indices, dtype=torch.int32, device="npu"),
        num_accepted_tokens=torch.tensor(accepted, dtype=torch.int32, device="npu"),
        g=None,
        gk=gate.npu(),
        scale_value=width**-0.5,
    )
    torch.testing.assert_close(output.cpu(), expected_output, rtol=3e-3, atol=2e-3)
    torch.testing.assert_close(state.cpu(), expected_state.half(), rtol=3e-3, atol=2e-3)
    _assert_page_guards(backing)


@pytest.mark.parametrize("graph", [False, True])
@pytest.mark.parametrize("paged", [False, True])
def test_native_state_table_matches_legacy_promotion(graph, paged):
    torch.npu.set_device(0)
    torch_npu.npu.set_compile_mode(jit_compile=False)
    enable_custom_op()
    torch.manual_seed(31045)
    tokens, heads, width = 6, 2, 128
    query = torch.nn.functional.normalize(torch.randn(tokens, heads, width), dim=-1).half().npu()
    key = torch.nn.functional.normalize(torch.randn(tokens, heads, width), dim=-1).half().npu()
    value = torch.randn(tokens, heads, width).half().npu()
    beta = torch.rand(tokens, heads).half().npu()
    gate = (-torch.rand(tokens, heads, width)).npu()
    initial = torch.randn(12, heads, width, width).half().npu()
    state, backing = _state_with_page_guards(initial, paged)
    lengths = torch.tensor([2, 2, 2], dtype=torch.int32, device="npu")
    slots = torch.tensor([[0, 1, 2], [4, 5, 6], [8, 9, 10]], dtype=torch.int32, device="npu")
    accepted = torch.tensor([1, 2, 3], dtype=torch.int32, device="npu")

    def run(target, table, counts):
        return torch.ops._C_ascend.npu_recurrent_gated_delta_rule_310(
            query=query,
            key=key,
            value=value,
            beta=beta,
            state=target,
            actual_seq_lengths=lengths,
            ssm_state_indices=table,
            g=None,
            gk=gate,
            num_accepted_tokens=counts,
            scale_value=width**-0.5,
        )

    for _ in range(2):
        run(state, slots, accepted)
    torch.npu.synchronize()
    if graph:
        captured = torch.npu.NPUGraph()
        with torch.npu.graph(captured):
            output = run(state, slots, accepted)

    profiles = [
        ([2, 2, 2], [[0, 1, 2], [4, 5, 6], [8, 9, 10]], [1, 2, 3]),
        ([1, 2, 0], [[8, 9, 10], [0, 1, 2], [-1, -1, -1]], [3, 2, 0]),
        ([0, 1, 1], [[-1, -1, -1], [4, 5, 6], [0, 1, 2]], [0, 3, 1]),
    ]
    for lens, indices, counts in profiles:
        lengths.copy_(torch.tensor(lens, dtype=torch.int32))
        slots.copy_(torch.tensor(indices, dtype=torch.int32))
        accepted.copy_(torch.tensor(counts, dtype=torch.int32))
        state.copy_(initial)
        reference = initial.clone()
        packed = []
        for n, row, count in zip(lens, indices, counts):
            if n:
                reference[row[0]].copy_(initial[row[count - 1]])
                packed.extend(row[:n])
        packed.extend([0] * (tokens - len(packed)))
        expected = run(
            reference,
            torch.tensor(packed, dtype=torch.int32, device="npu"),
            torch.tensor([int(n > 0) for n in lens], dtype=torch.int32, device="npu"),
        )
        if graph:
            captured.replay()
        else:
            output = run(state, slots, accepted)
        torch.npu.synchronize()
        assert torch.equal(output[: sum(lens)].cpu(), expected[: sum(lens)].cpu())
        assert torch.equal(state.cpu(), reference.cpu())
        _assert_page_guards(backing)


def test_kpool_selector_replays_new_positions_and_request_pages():
    from types import SimpleNamespace

    from vllm_ascend.models.glm5next.kpool_ops import score_and_select_kpool_tokens
    from vllm_ascend.models.glm5next.sparse_attn_indexer_kpool import SparseAttnIndexerKpool

    torch.npu.set_device(0)
    torch_npu.npu.set_compile_mode(jit_compile=False)
    selector = SparseAttnIndexerKpool.__new__(SparseAttnIndexerKpool)
    torch.nn.Module.__init__(selector)
    torch.manual_seed(31046)
    cache = torch.randn(8, 4, 1, 128).half().npu()
    selector.k_cache = SimpleNamespace(kv_cache=cache)
    selector.max_model_len = 32
    selector.topk_tokens = 8
    selector.topk_indices_buffer = torch.empty(3, 12, dtype=torch.int32, device="npu")
    query = torch.randn(3, 2, 128).half().npu()
    weights = torch.rand(3, 2).npu()
    positions = torch.tensor([3, 5, 7], device="npu")
    boundaries = torch.tensor([1, 3], dtype=torch.int32, device="npu")
    table = torch.tensor([[2, 3], [0, 1]], dtype=torch.int32, device="npu")
    metadata = SimpleNamespace(cum_query_lens=boundaries, block_table=table)

    def run():
        return selector._select_tokens_fixed(query, weights, positions, 4, metadata)

    for _ in range(2):
        run()
    torch.npu.synchronize()
    captured = torch.npu.NPUGraph()
    with torch.npu.graph(captured):
        output = run()
    for pos, ends, pages in [([3, 5, 7], [1, 3], [[2, 3], [0, 1]]), ([19, 20, 23], [2, 3], [[0, 1], [2, 3]])]:
        positions.copy_(torch.tensor(pos))
        boundaries.copy_(torch.tensor(ends, dtype=torch.int32))
        table.copy_(torch.tensor(pages, dtype=torch.int32))
        captured.replay()
        torch.npu.synchronize()
        actual = output.cpu()
        start = 0
        for req, end in enumerate(ends):
            keys = cache[torch.tensor(pages[req], device="npu")].reshape(-1, 128)
            expected = score_and_select_kpool_tokens(
                query[start:end], weights[start:end], keys, positions[start:end], 8, 4
            ).cpu()
            assert torch.equal(actual[start:end, :11].sort(-1).values, expected.sort(-1).values)
            start = end


def test_glm_mtp_mla_graph_preserves_causal_query_boundaries():
    from types import SimpleNamespace

    from vllm_ascend._310p.attention.mla_v1_310 import AscendMLAImpl310, AscendMLAMetadataBuilder310

    torch.npu.set_device(0)
    torch_npu.npu.set_compile_mode(jit_compile=False)
    enable_custom_op()
    torch.manual_seed(31047)
    tokens, heads, width, block = 4, 16, 512, 640
    cache_cpu = torch.randn(2, width // 16, block, 16).half() * 0.1
    # Logical dimensions already describe physical NZ order in the shared cache.
    cache = cache_cpu.npu()
    query_cpu = torch.randn(tokens, heads, width).half() * 0.1
    query = query_cpu.npu()
    positions = torch.tensor([2, 3, 5, 6], device="npu")
    boundaries = torch.tensor([0, 2, 4], dtype=torch.int32, device="npu")
    table = torch.arange(40, dtype=torch.int32, device="npu").reshape(2, 20)
    impl = AscendMLAImpl310.__new__(AscendMLAImpl310)
    impl.host_kv_layer = None
    impl.scale = 256**-0.5
    impl._decode_constant_buffers = {}
    impl._v_up_proj = lambda value: value
    impl.glm_indexer = SimpleNamespace(
        index_kpool=4, topk_tokens=8, topk_indices_buffer=torch.zeros(tokens, 12, dtype=torch.int32, device="npu")
    )
    # Build metadata outside capture, as serving does. Handwritten metadata
    # would miss zero-length padding that copies the runner's live buffers.
    builder = AscendMLAMetadataBuilder310.__new__(AscendMLAMetadataBuilder310)
    builder.num_actual_tokens = builder.num_decode_tokens = builder.graph_pad_size = tokens
    builder.num_decodes = 2
    builder.use_mla_rope = False
    builder.device = positions.device
    builder.seq_lens = torch.tensor([4, 7], dtype=torch.int32)
    builder.speculative_config = SimpleNamespace(disable_padded_drafter_batch=False)
    builder.attn_mask_builder = SimpleNamespace(get_splitfuse_attn_mask=lambda: None)
    builder.decode_metadata_cls = SimpleNamespace
    builder.nope_zero_rope_cache = {}
    builder.block_table = table
    builder.slot_mapping = torch.arange(tokens, dtype=torch.int32, device="npu")
    common = SimpleNamespace(
        num_reqs=2,
        positions=positions,
        query_start_loc_cpu=boundaries.cpu(),
        seq_lens=builder.seq_lens.npu(),
        decode_token_per_req=2,
        actual_seq_lengths_q=[2, 4],
    )
    metadata = SimpleNamespace(
        num_decodes=2,
        query_start_loc=boundaries,
        decode=builder.build_decode_metadata(0, common),
    )

    def run():
        return impl._forward_decode_fused(query, cache, cache, metadata)

    for _ in range(2):
        run()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        result = run()
    for pos, ends, pages in [([2, 3, 5, 6], [0, 2, 4], [0, 1]), ([4, 3, 4, 5], [0, 1, 4], [1, 0])]:
        positions.copy_(torch.tensor(pos))
        boundaries.copy_(torch.tensor(ends, dtype=torch.int32))
        table.copy_(torch.tensor([[page * 20 + i for i in range(20)] for page in pages], dtype=torch.int32))
        graph.replay()
        actual = result.transpose(0, 1).cpu()
        for req, (start, end) in enumerate(zip(ends[:-1], ends[1:])):
            for row in range(start, end):
                keys = cache_cpu[pages[req], :, : pos[row] + 1, :].permute(1, 0, 2).reshape(-1, width).float()
                expected = (torch.softmax(query_cpu[row].float() @ keys.T * impl.scale, dim=-1) @ keys).half()
                torch.testing.assert_close(actual[row], expected, rtol=5e-3, atol=3e-3)


def test_secondary_draft_slots_replay_device_positions():
    from vllm_ascend.spec_decode.multi_kv_cache_group_proposer import compute_packed_draft_slots

    torch.npu.set_device(0)
    torch_npu.npu.set_compile_mode(jit_compile=False)
    table = torch.tensor([[5, 8], [2, 3], [-1, -1]], dtype=torch.int32, device="npu")
    boundaries = torch.tensor([0, 2, 3, 3], dtype=torch.int32, device="npu")
    positions = torch.tensor([3, 4, 6, 0], device="npu")
    rows = torch.arange(4, dtype=torch.int32, device="npu")
    for _ in range(2):
        compute_packed_draft_slots(table, boundaries, positions, 4, rows)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        slots = compute_packed_draft_slots(table, boundaries, positions, 4, rows)
    graph.replay()
    assert slots.cpu().tolist() == [23, 32, 14, -1]
    table.copy_(torch.tensor([[-1, -1], [2, 3], [5, 8]], dtype=torch.int32))
    boundaries.copy_(torch.tensor([0, 0, 2, 3], dtype=torch.int32))
    positions.copy_(torch.tensor([1, 2, 7, -1]))
    graph.replay()
    assert slots.cpu().tolist() == [9, 10, 35, -1]


def test_draft_indexer_first_forward_captures_prepared_nz_weights():
    from types import SimpleNamespace

    from vllm_ascend.models.glm5next.attention import Indexer
    from vllm_ascend.models.glm5next_w2.model import _prepare_dsa_indexer_weights

    torch.npu.set_device(0)
    torch_npu.npu.set_compile_mode(jit_compile=False)
    indexer = Indexer.__new__(Indexer)
    torch.nn.Module.__init__(indexer)
    indexer.n_head, indexer.head_dim, indexer.rope_dim = 2, 128, 0
    indexer.softmax_scale, indexer.index_kpool = 128**-0.5, 4
    indexer.wq_b = lambda qr: (qr, None)
    indexer.wk_weights_proj = SimpleNamespace(weight=torch_npu.npu_format_cast(torch.randn(130, 16).half().npu(), 29))
    indexer.index_kpool_compress_gate = torch.randn(128, 16).half().npu()
    indexer.index_kpool_compress_ape = torch.zeros(4, 128, device="npu")
    indexer.k_norm = torch.nn.LayerNorm(128).npu()
    indexer.indexer_op = lambda hidden, q, k, weights, **kw: torch.cat(
        (q.float().flatten(1), k, weights, kw["gate_score"]), dim=-1
    )
    _prepare_dsa_indexer_weights([SimpleNamespace(self_attn=SimpleNamespace(indexer=indexer))])
    hidden = torch.randn(2, 16).half().npu()
    qr = torch.randn(2, 256).half().npu()
    positions = torch.tensor([3, 4], device="npu")
    graph = torch.npu.NPUGraph()
    # No eager indexer forward: the draft's first call may be captured.
    with torch.npu.graph(graph):
        output = indexer(hidden, qr, positions, None)
    for factor in (0.5, 2.0):
        hidden.mul_(factor)
        expected = indexer(hidden, qr, positions, None)
        graph.replay()
        torch.testing.assert_close(output.cpu(), expected.cpu(), rtol=1e-5, atol=1e-5)
