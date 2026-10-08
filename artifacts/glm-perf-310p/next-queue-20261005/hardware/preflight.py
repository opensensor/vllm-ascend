# SPDX-License-Identifier: Apache-2.0
"""Run independent parity gates in paused resident GLM workers."""

import torch

from tools.glm_perf.resident_worker import ResidentWorkerExtension


def audit(worker):
    from tools.glm_perf.resident_candidates import direct_route_tokens as direct
    from tools.glm_perf.resident_candidates import indexer_projection as projection
    from tools.glm_perf.resident_candidates import kda_input_preparation as qk
    from tools.glm_perf.resident_candidates import kpool_decode_epilogue as epilogue
    from tools.glm_perf.resident_candidates import moe_half_unpermute as combine
    from vllm_ascend.models.glm5next import kpool_ops
    from vllm_ascend.models.glm5next.attention import Indexer
    from vllm_ascend.models.glm5next_w2 import kda_310
    from vllm_ascend.models.qwen4_exp.grouped_expert_dispatch import build_grouped_expert_dispatch

    generator = torch.Generator().manual_seed(310)
    result = {}

    def exact(a, b):
        torch.testing.assert_close(a, b, rtol=0, atol=0)

    def run(name, fn):
        try:
            details = fn()
            torch.npu.synchronize()
            result[name] = {"passed": True, "details": details}
        except Exception as exc:
            result[name] = {"passed": False, "error": str(exc)}

    def check_qk():
        for rows in (1, 2, 4, 8):
            data = torch.randn(1, rows, 16, 256, generator=generator).half().npu()
            q, k = data[..., ::2], data[..., 1::2]
            actual = qk.normalize_qk(q, k, kda_310._l2norm_310p)
            for a, b in zip(actual, (kda_310._l2norm_310p(q), kda_310._l2norm_310p(k))):
                exact(a, b)
        return {"cases": 4}

    def check_epilogue():
        for rows, capacity, budget in ((2, 3, 16), (8, 77, 16), (8, 77760, 2048)):
            positions = torch.tensor([-1, 0, 2, 3, 4, 15, 1023, 300051][:rows]).npu()
            logits = torch.randint(-2, 3, (rows, capacity), generator=generator).float().npu()
            logits.masked_fill_(
                torch.arange(capacity, device=logits.device)[None, :] >= ((positions + 1) // 4)[:, None], -torch.inf
            )
            expected = []
            for row in range(rows):
                selected, _, starts, counts = kpool_ops.select_kpool_groups(
                    logits[row : row + 1], positions[row : row + 1], budget, 4, scores_are_causal=True
                )
                expected.append(kpool_ops.expand_kpool_groups(selected, starts, counts, 4))
            exact(epilogue.select_and_expand(logits, positions, budget, 4), torch.cat(expected))
        return {"cases": 3}

    def check_combine():
        for tokens in (2, 8, 640):
            routes, hidden = tokens * 8, 4096
            order = torch.randperm(routes, generator=generator).npu()
            inverse = order.argsort()
            routed = torch.randn(routes, hidden, generator=generator).half().npu()
            weights = torch.randn(routes, 1, generator=generator).npu()
            reference = routed.float() * weights.index_select(0, order)
            reference = reference.index_select(0, inverse).reshape(tokens, 8, hidden).sum(1)
            exact(combine.combine_routes(routed, inverse, weights, tokens, 8, hidden), reference)
        return {"cases": 3}

    def check_direct():
        for tokens in (2, 8, 640, 1280):
            routes = tokens * 8
            order = torch.randperm(routes, generator=generator).npu()
            expected = torch.arange(tokens, device=order.device).repeat_interleave(8).index_select(0, order)
            actual = direct.sorted_token_ids(order, 8)
            exact(actual.long(), expected)
            hidden = torch.randn(tokens, 128, generator=generator).half().npu()
            exact(hidden.index_select(0, actual), hidden.index_select(0, expected))
        # Also enforce runtime source compatibility, without applying patches.
        direct.make_dispatch(build_grouped_expert_dispatch)
        return {"cases": 4}

    def check_projection():
        bank = projection.ProjectionBank()
        projection.prepare_runner(bank, worker.model_runner, Indexer)
        differences = []
        for index, entry in list(bank.entries.items()):
            for rows in (2, 8, 640):
                x = torch.randn(rows, entry.key_weight.shape[1], generator=generator).half().float().npu()
                actual = bank.project(index, x)
                expected = (torch.mm(x, entry.key_weight.t()), torch.mm(x, entry.gate_weight.t()))
                for a, b in zip(actual, expected):
                    changed = int((a != b).sum().item())
                    if changed:
                        differences.append({"rows": rows, "changed": changed, "max_abs": (a - b).abs().max().item()})
        if differences:
            raise ValueError(f"projection exact gate failed: {differences[:8]}")
        return {"indexers": len(bank.entries), "extra_bytes": bank.prepared_bytes}

    for name, fn in [
        ("kda_batched_qk", check_qk),
        ("kpool_decode_epilogue", check_epilogue),
        ("moe_half_unpermute", check_combine),
        ("direct_route_tokens", check_direct),
        ("indexer_projection", check_projection),
    ]:
        run(name, fn)
    return result


def replacements():
    cached = {}

    def status(self):
        if not cached:
            cached.update(audit(self))
        receipt = ResidentWorkerExtension.resident_status(self)
        receipt["queue_preflight"] = cached
        return receipt

    return {"vllm_ascend._310p.worker_310p:NPUWorker310.resident_status": status}
