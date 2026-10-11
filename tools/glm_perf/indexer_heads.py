# SPDX-License-Identifier: Apache-2.0
"""Prefill-only head partition: exact score sum before any pool selection.

Decode retains the original loaded implementation. Pool/tail writes remain
replicated; this candidate partitions Q and head-weight projection and scoring.
FP32 head reduction changes summation order and requires a numerical gate.
"""

from types import FunctionType, MethodType

import torch
import torch.distributed as dist

from .indexer_owner import MAX_CAPTURE_ROWS, validate_config

HEAD_OPTIONS = ("indexer_head_sharding", "indexer_head_budget_bytes", "indexer_score_elements")
TOTAL_HEADS = 32
HEAD_DIM = 128


def read_options(options):
    if not any(key in options for key in HEAD_OPTIONS):
        return None
    if options.get(HEAD_OPTIONS[0]) is not True or any(key not in options for key in HEAD_OPTIONS[1:]):
        raise ValueError("head sharding requires explicit enablement, storage budget, and score tile elements")
    budget, elements = (options[key] for key in HEAD_OPTIONS[1:])
    if type(budget) is not int or budget < 1 or type(elements) is not int or elements < 1:
        raise ValueError("head sharding budgets require positive integers")
    if "indexer_owner_rank" in options:
        raise ValueError("head sharding and full indexer ownership are alternative candidates")
    return dict(budget_bytes=budget, score_elements=elements)


def memory_plan(
    *, size, max_tokens, max_model_len, score_elements, query_rank=1536, hidden=4096, indexers=1, weight_itemsize=2
):
    if size not in (2, 4) or min(max_tokens, max_model_len, score_elements, indexers) < 1:
        raise ValueError("head partition requires TP2/TP4 and positive memory bounds")
    heads = TOTAL_HEADS // size
    pools = (max_model_len + 3) // 4
    # One query must fit: pool columns cannot be truncated before global scores.
    if score_elements < heads * pools:
        raise ValueError("score budget cannot hold one complete pooled-context row")
    rows = min(max_tokens, max(1, score_elements // (heads * pools)))
    weights = indexers * ((heads * HEAD_DIM * query_rank) * weight_itemsize + (2 * HEAD_DIM + heads) * hidden * 4)
    # Local head scores, reduced scores, widened keys, Q/FWHT and head weights.
    scratch = 4 * (
        rows * (heads + 1) * pools
        + pools * HEAD_DIM
        + max_tokens * heads * HEAD_DIM * 8
        + max_tokens * hidden
        + max_tokens * (HEAD_DIM + heads) * 4
    )
    return dict(
        local_heads=heads,
        score_rows=rows,
        prepared_weight_bytes=weights,
        score_scratch_bound_bytes=scratch,
        required_bytes=weights + scratch,
        numerical_contract="FP32 partial sums then TP reduction; not bitwise full-head sum",
    )


class Proxy:
    def __init__(self, original, **values):
        self.original = original
        self.__dict__.update(values)

    def __getattr__(self, name):
        return getattr(self.original, name)


class LocalQuery:
    def __init__(self, weight):
        self.weight = weight

    def __call__(self, value):
        return torch.nn.functional.linear(value, self.weight), None


def clone_function(original, **bindings):
    result = FunctionType(
        original.__code__,
        dict(original.__globals__, **bindings),
        original.__name__,
        original.__defaults__,
        original.__closure__,
    )
    result.__kwdefaults__ = original.__kwdefaults__
    return result


def reduced_score(original, *, group, backend=dist):
    def score(queries, weights, keys):
        logits = original(queries, weights, keys)
        # All heads must be summed before causal mask/top-k. Never merge local
        # top-k: a pool below each partition's cutoff can win the global sum.
        backend.all_reduce(logits, group=group)
        return logits

    return score


def bind_indexer(indexer, *, rank, size, group, score_elements, backend=dist):
    from vllm_ascend.models.glm5next import kpool_ops

    if hasattr(indexer, "_glm_head_partition") or hasattr(indexer, "_glm_indexer_result_owner"):
        raise ValueError("indexer head partition is cold-load only and excludes full ownership")
    if size not in (2, 4) or not 0 <= rank < size or indexer.n_head != TOTAL_HEADS or indexer.rope_dim != 0:
        raise ValueError("head partition initially requires 32 heads, zero indexer RoPE, and TP2/TP4")
    if indexer.wq_b.weight.device.type == "npu" and not hasattr(indexer, "_native_bf16_cast"):
        raise ValueError("head partition requires the permanent AI-Core BF16 conversion bundle on NPU")
    heads = TOTAL_HEADS // size
    start, stop = rank * heads, (rank + 1) * heads
    query = LocalQuery(indexer.wq_b.weight.detach()[start * HEAD_DIM : stop * HEAD_DIM].clone())
    wk = indexer.wk_weights_proj.weight.detach()
    local_wk = torch.cat((wk[:HEAD_DIM].float(), wk[HEAD_DIM + start : HEAD_DIM + stop].float())).contiguous()
    # Preserve the full-head normalization exactly: original multiplies
    # softmax_scale * n_head**-0.5. Expose the precomputed factor to the facade.
    scale = indexer.softmax_scale * TOTAL_HEADS**-0.5
    original = indexer.forward
    operator = indexer.indexer_op
    convert = getattr(indexer, "_native_bf16_cast", None)

    def local_score(queries, weights, keys):
        rotated = kpool_ops.hadamard128(queries)
        rotated = convert.round(rotated, torch.float32) if convert is not None else rotated.bfloat16().float()
        logits = (rotated.reshape(-1, HEAD_DIM) @ keys.float().t()).reshape(queries.shape[0], heads, -1)
        return logits.relu_().mul_(weights.float().unsqueeze(-1)).sum(dim=1)

    score = reduced_score(local_score, group=group, backend=backend)
    select = clone_function(
        kpool_ops.score_and_select_kpool_tokens, score_kpool=score, MAX_KPOOL_SCORE_ELEMENTS=score_elements
    )
    facade_op = Proxy(operator)
    for name in ("_select_tokens", "_select_tokens_fixed"):
        function = getattr(operator, name).__func__
        facade_op.__dict__[name] = MethodType(
            clone_function(
                function,
                score_kpool=score,
                score_and_select_kpool_tokens=select,
                supports_live_kpool_score=lambda *args: False,
            ),
            facade_op,
        )

    def call_operator(*args, **kwargs):
        return operator.forward_oot.__func__(facade_op, *args, **kwargs)

    facade = Proxy(
        indexer,
        n_head=heads,
        wq_b=query,
        _wk_weight_f32=local_wk,
        _gate_weight_f32=indexer.index_kpool_compress_gate.detach().float(),
        indexer_op=call_operator,
    )

    def forward(self, hidden_states, qr, positions, rotary_emb):
        if hidden_states.shape[0] <= MAX_CAPTURE_ROWS:
            return original(hidden_states, qr, positions, rotary_emb)
        q, _ = query(qr)
        q = q.view(-1, heads, HEAD_DIM)
        hidden_f32 = hidden_states.float()
        kw = torch.mm(hidden_f32, local_wk.t())
        k = self.k_norm(kw[:, :HEAD_DIM])
        q = convert(q, torch.bfloat16) if convert is not None else q.bfloat16()
        weights = kw[:, HEAD_DIM:] * scale
        gate_score = torch.mm(hidden_f32, facade._gate_weight_f32.t())
        return call_operator(
            hidden_states,
            q,
            k,
            weights,
            gate_score=gate_score,
            compress_ape=self.index_kpool_compress_ape,
            index_kpool=self.index_kpool,
            positions=positions,
        )

    indexer.forward = MethodType(forward, indexer)
    indexer._glm_head_partition = facade


def plan_install(model, *, budget_bytes, score_elements):
    from vllm_ascend.models.glm5next.attention import Indexer

    validate_config(model.vllm_config, 0)
    if model.vllm_config.scheduler_config.max_num_seqs > MAX_CAPTURE_ROWS:
        raise ValueError("prefill head partition initially requires at most eight decode sequences")
    indexers = [module for module in model.modules() if isinstance(module, Indexer)]
    if not indexers or any(
        module.n_head != TOTAL_HEADS
        or module.head_dim != HEAD_DIM
        or module.rope_dim != 0
        or hasattr(module, "_glm_head_partition")
        or hasattr(module, "_glm_indexer_result_owner")
        for module in indexers
    ):
        raise ValueError("head partition requires unbound zero-RoPE GLM indexers")
    config = model.vllm_config
    plan = memory_plan(
        size=config.parallel_config.tensor_parallel_size,
        max_tokens=config.scheduler_config.max_num_batched_tokens,
        max_model_len=config.model_config.max_model_len,
        score_elements=score_elements,
        query_rank=indexers[0].wq_b.weight.shape[1],
        hidden=indexers[0].wk_weights_proj.weight.shape[1],
        indexers=len(indexers),
        weight_itemsize=indexers[0].wq_b.weight.element_size(),
    )
    if type(budget_bytes) is not int or plan["required_bytes"] > budget_bytes:
        raise ValueError("head partition exceeds separate prepared weights and score reservation")
    return indexers, plan


def install(model, **options):
    from vllm.distributed import get_tp_group

    indexers, plan = plan_install(model, **options)
    tp = get_tp_group()
    if tp.world_size != model.vllm_config.parallel_config.tensor_parallel_size:
        raise ValueError("head partition TP group differs from configuration")
    for indexer in indexers:
        bind_indexer(
            indexer,
            rank=tp.rank_in_group,
            size=tp.world_size,
            group=tp.device_group,
            score_elements=options["score_elements"],
        )
    return dict(**plan, indexers=len(indexers), hardware_qualified=False, selection_replicated=True)
