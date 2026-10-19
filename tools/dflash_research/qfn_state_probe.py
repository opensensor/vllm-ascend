# SPDX-License-Identifier: Apache-2.0
"""Compare QFN speculative convolution/GDN state with sequential native steps."""

import argparse
import json
import time
from pathlib import Path

import torch


@torch.inference_mode()
def probe(device, rank, draft_lengths=(1, 3)):
    import torch_npu

    from vllm_ascend.models.qwen4_exp.head_partition import gdn_execution_shard, gdn_head_shard
    from vllm_ascend.utils import enable_custom_op

    torch.npu.set_device(device)
    torch_npu.npu.set_compile_mode(jit_compile=False)
    enable_custom_op()
    torch.manual_seed(701 + rank)
    live = gdn_execution_shard(gdn_head_shard(16, 48, 128, 128, rank, 6, "padded_compact"))
    dev = f"npu:{device}"
    reports = []

    def tensor(values):
        return torch.tensor(values, dtype=torch.int32, device=dev)

    def recurrent(q, k, v, beta, g, state, lengths, ids, accepted):
        return torch.ops._C_ascend.npu_recurrent_gated_delta_rule_310(
            query=q,
            key=k,
            value=v,
            beta=beta,
            g=g,
            gk=None,
            state=state,
            actual_seq_lengths=lengths,
            ssm_state_indices=ids,
            num_accepted_tokens=accepted,
            scale_value=128**-0.5,
        )

    def conv(x, weights, state, starts, accepted):
        return torch.ops._C_ascend.npu_causal_conv1d_310(
            x,
            weights,
            bias=None,
            conv_states=state,
            query_start_loc=starts,
            cache_indices=tensor([0, 2]),
            initial_state_mode=None,
            num_accepted_tokens=accepted,
            activation_mode=1,
            pad_slot_id=-1,
            run_mode=1,
        )

    for draft_k in draft_lengths:
        width = draft_k + 1
        for accepted_drafts in range(width):
            # Acceptance counts include the anchor: zero accepted drafts is 1.
            counts = [accepted_drafts + 1, width - accepted_drafts]
            slots = [list(range(width)), list(range(width + 1, 2 * width + 1))]
            flat_slots = tensor([slot for row in slots for slot in row])
            raw = torch.full((2 * width + 1, 10 * 128 * 128), -93.0, dtype=torch.float32, device=dev)
            state = raw[:, : live.value_heads * 128 * 128].view(-1, live.value_heads, 128, 128)
            state.copy_(torch.randn_like(state) * 0.01)
            unused_state = state[width].clone()
            raw_conv = torch.full((3, (3 + draft_k) * 1920), -91.0, dtype=torch.float16, device=dev)
            history = raw_conv[:, : (3 + draft_k) * live.conv_dim].view(3, 3 + draft_k, live.conv_dim)
            history.copy_(torch.randn_like(history) * 0.05)
            unused_conv = history[1].clone()
            weights = torch.randn(4, live.conv_dim, dtype=torch.float16, device=dev) * 0.05
            for iteration in range(2):
                if iteration:
                    counts.reverse()
                q = torch.randn(2 * width, live.key_heads, 128, dtype=torch.float16, device=dev) * 0.05
                k = torch.randn_like(q) * 0.05
                v = torch.randn(2 * width, live.value_heads, 128, dtype=torch.float16, device=dev) * 0.05
                beta = torch.full((2 * width, live.value_heads), 0.5, dtype=torch.float16, device=dev)
                g = torch.full((2 * width, live.value_heads), -0.1, dtype=torch.float32, device=dev)
                x = torch.randn(2 * width, live.conv_dim, dtype=torch.float16, device=dev) * 0.05
                before = state.clone()
                before_conv = history.clone()
                seq_state = torch.stack([before[slots[r][counts[r] - 1]] for r in range(2)])
                seq_conv = torch.zeros(3, 3, live.conv_dim, dtype=torch.float16, device=dev)
                for r, row in enumerate((0, 2)):
                    seq_conv[row].copy_(before_conv[row, counts[r] - 1 : counts[r] + 2])
                actual = recurrent(q, k, v, beta, g, state, tensor([width, width]), flat_slots, tensor(counts))
                actual_conv = conv(x, weights, history, tensor([0, width, 2 * width]), tensor(counts))
                for step in range(width):
                    indices = tensor([step, width + step]).long()
                    expected = recurrent(
                        q[indices],
                        k[indices],
                        v[indices],
                        beta[indices],
                        g[indices],
                        seq_state,
                        tensor([1, 1]),
                        tensor([0, 1]),
                        None,
                    )
                    expected_conv = conv(x[indices], weights, seq_conv, tensor([0, 1, 2]), None)
                    torch.testing.assert_close(actual[indices], expected, rtol=0, atol=0)
                    torch.testing.assert_close(actual_conv[indices], expected_conv, rtol=0, atol=0)
                    for r in range(2):
                        torch.testing.assert_close(state[slots[r][step]], seq_state[r], rtol=0, atol=0)
                for r, row in enumerate((0, 2)):
                    expected_history = torch.cat(
                        (before_conv[row, counts[r] : counts[r] + 2], x[r * width : (r + 1) * width])
                    )
                    torch.testing.assert_close(history[row], expected_history, rtol=0, atol=0)
                torch.testing.assert_close(state[width], unused_state, rtol=0, atol=0)
                torch.testing.assert_close(history[1], unused_conv, rtol=0, atol=0)
                assert (raw[:, live.value_heads * 128 * 128 :] == -93).all().item()
                assert (raw_conv[:, (3 + draft_k) * live.conv_dim :] == -91).all().item()
                reports.append(
                    {"k": draft_k, "accepted_drafts": [n - 1 for n in counts], "iteration": iteration, "exact": True}
                )
    return {
        "device": device,
        "rank_geometry": rank,
        "key_heads": live.key_heads,
        "value_heads": live.value_heads,
        "cases": reports,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--draft-lengths", type=int, nargs="+", choices=(1, 3, 7), default=[1, 3])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("use a fresh output path")
    start = time.monotonic()
    result = probe(args.device, args.rank, args.draft_lengths)
    result.update(
        passed=True, elapsed_s=time.monotonic() - start, scope="native state kernels, no full target generation"
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
