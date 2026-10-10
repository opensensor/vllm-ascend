# SPDX-License-Identifier: Apache-2.0
"""Queued coherent-OPP parity for live GDN views inside uniform cache pages."""

import pytest

from vllm_ascend.models.qwen4_exp.head_partition import gdn_execution_caches, gdn_execution_shard, gdn_head_shard


@pytest.mark.parametrize("rank", [0, 5])
@pytest.mark.parametrize("graph_replay", [False, True])
def test_compact_conv_view_matches_dense_state_and_preserves_slack(gdn_operators, rank, graph_replay):
    torch = gdn_operators
    torch.manual_seed(625)
    allocated = gdn_head_shard(16, 48, 128, 128, rank, 6, "padded_compact")
    live = gdn_execution_shard(allocated)
    raw_conv = torch.full((3, 3, allocated.conv_dim), -91.0, dtype=torch.float16, device="npu")
    raw_state = torch.zeros(3, 9, 128, 128, dtype=torch.float32, device="npu")
    conv, _ = gdn_execution_caches((raw_conv, raw_state), allocated, live)
    dense = torch.randn(3, 3, live.conv_dim, dtype=torch.float16, device="npu") * 0.05
    conv.copy_(dense)
    untouched = conv[1].clone()
    x = torch.randn(3, live.conv_dim, dtype=torch.float16, device="npu") * 0.05
    weight = torch.randn(4, live.conv_dim, dtype=torch.float16, device="npu") * 0.05
    starts = torch.tensor([0, 2, 3], dtype=torch.int32, device="npu")
    ids = torch.tensor([0, 2], dtype=torch.int32, device="npu")
    initialized = torch.tensor([True, False], device="npu")

    def call(state):
        return torch.ops._C_ascend.npu_causal_conv1d_310(
            x,
            weight,
            bias=None,
            conv_states=state,
            query_start_loc=starts,
            cache_indices=ids,
            initial_state_mode=initialized,
            num_accepted_tokens=None,
            activation_mode=1,
            pad_slot_id=-1,
            run_mode=0,
        )

    expected, actual = call(dense), call(conv)
    torch.npu.synchronize()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(conv, dense, rtol=0, atol=0)
    torch.testing.assert_close(conv[1], untouched, rtol=0, atol=0)
    assert (raw_conv.flatten(1)[:, 3 * live.conv_dim :] == -91).all().item()
    if graph_replay:
        captured = torch.npu.NPUGraph()
        with torch.npu.graph(captured):
            output = call(conv)
        captured.replay()
        torch.testing.assert_close(output, call(dense), rtol=0, atol=0)
        x.mul_(0.7)
        captured.replay()
        torch.testing.assert_close(output, call(dense), rtol=0, atol=0)
        torch.testing.assert_close(conv, dense, rtol=0, atol=0)
        torch.testing.assert_close(conv[1], untouched, rtol=0, atol=0)
        assert (raw_conv.flatten(1)[:, 3 * live.conv_dim :] == -91).all().item()


@pytest.mark.parametrize("rank", [0, 5])
@pytest.mark.parametrize("graph_replay", [False, True])
def test_compact_recurrent_view_matches_dense_state_and_preserves_slack(gdn_operators, rank, graph_replay):
    torch = gdn_operators
    torch.manual_seed(626)
    allocated = gdn_head_shard(16, 48, 128, 128, rank, 6, "padded_compact")
    live = gdn_execution_shard(allocated)
    # Include physical page alignment slack beyond the nine-head descriptor.
    page_elements = 10 * 128 * 128
    raw = torch.full((3, page_elements), -93.0, dtype=torch.float32, device="npu")
    state = raw[:, : 9 * 128 * 128].view(3, 9, 128, 128)
    conv = torch.zeros(3, 3, 1920, dtype=torch.float16, device="npu")
    _, state_view = gdn_execution_caches((conv, state), allocated, live)
    dense = torch.randn(3, live.value_heads, 128, 128, dtype=torch.float32, device="npu") * 0.01
    state_view.copy_(dense)
    untouched = state_view[1].clone()
    q = torch.randn(3, live.key_heads, 128, dtype=torch.float16, device="npu") * 0.05
    k = torch.randn_like(q) * 0.05
    v = torch.randn(3, live.value_heads, 128, dtype=torch.float16, device="npu") * 0.05
    beta = torch.full((3, live.value_heads), 0.5, dtype=torch.float16, device="npu")
    g = torch.full((3, live.value_heads), -0.1, dtype=torch.float32, device="npu")
    lengths = torch.tensor([2, 1], dtype=torch.int32, device="npu")
    ids = torch.tensor([0, 0, 2], dtype=torch.int32, device="npu")

    def call(cache):
        return torch.ops._C_ascend.npu_recurrent_gated_delta_rule_310(
            query=q,
            key=k,
            value=v,
            beta=beta,
            state=cache,
            actual_seq_lengths=lengths,
            ssm_state_indices=ids,
            g=g,
            gk=None,
            num_accepted_tokens=None,
            scale_value=128**-0.5,
        )

    expected, actual = call(dense), call(state_view)
    torch.npu.synchronize()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(state_view, dense, rtol=0, atol=0)
    torch.testing.assert_close(state_view[1], untouched, rtol=0, atol=0)
    assert (raw[:, live.value_heads * 128 * 128 :] == -93).all().item()
    if graph_replay:
        captured = torch.npu.NPUGraph()
        with torch.npu.graph(captured):
            output = call(state_view)
        captured.replay()
        torch.testing.assert_close(output, call(dense), rtol=0, atol=0)
        q.mul_(0.7)
        v.mul_(0.8)
        captured.replay()
        torch.testing.assert_close(output, call(dense), rtol=0, atol=0)
        torch.testing.assert_close(state_view, dense, rtol=0, atol=0)
        torch.testing.assert_close(state_view[1], untouched, rtol=0, atol=0)
        assert (raw[:, live.value_heads * 128 * 128 :] == -93).all().item()
