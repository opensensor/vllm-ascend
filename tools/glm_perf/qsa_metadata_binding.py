# SPDX-License-Identifier: Apache-2.0
"""Bind a fused plan to each loaded GLM attention instance, reversibly."""

from types import FunctionType


def bind_attention(bindings, runner, native):
    roots = [runner.model]
    draft = getattr(getattr(runner, "drafter", None), "model", None)
    if draft is not None:
        roots.append(draft)
    count = 0
    seen = set()
    for root in roots:
        for module in root.modules():
            owner = getattr(module, "impl", None)
            if owner is None or id(owner) in seen or getattr(owner, "glm_indexer", None) is None:
                continue
            if not hasattr(owner, "_forward_decode_fused"):
                continue
            seen.add(id(owner))
            bind_one(bindings, owner, native)
            count += 1
    if not count:
        raise ValueError("no loaded GLM QSA attention instance found")
    return count


def bind_one(bindings, owner, native):
    active = []
    original_plan = owner._get_kpool_qsa_plan

    def plan(self, positions, token_start, num_tokens):
        return active[-1][0] if active else original_plan(positions, token_start, num_tokens)

    bindings.bind(owner, "_get_kpool_qsa_plan", plan)
    for method, prefill in (("_forward_decode_fused", False), ("_forward_prefill_paged_latent", True)):
        original = getattr(owner, method).__func__
        table_original = original.__globals__["_qsa_cache_block_table"]

        def table_hook(table, block_size, fallback=table_original):
            return active[-1][1] if active else fallback(table, block_size)

        namespace = dict(original.__globals__, _qsa_cache_block_table=table_hook)
        clone = FunctionType(
            original.__code__, namespace, original.__name__, original.__defaults__, original.__closure__
        )
        clone.__kwdefaults__ = original.__kwdefaults__

        def forward(self, query, *args, implementation=clone, is_prefill=prefill):
            metadata = args[-1]
            if is_prefill:
                cache, attention = args
                meta = attention.prefill
                rows = meta.actual_seq_lengths_q[-1]
                start = attention.num_decode_tokens
                table = meta.block_table
                block_size = cache[0].shape[2]
            else:
                key_cache, _, attention = args
                meta = attention.decode
                rows, start = query.shape[0], 0
                table = meta.block_table[: metadata.num_decodes]
                block_size = key_cache.shape[2]
            if self.host_kv_layer is not None:
                return implementation(self, query, *args)
            indexer = self.glm_indexer
            inputs = (indexer.topk_indices_buffer, meta.input_positions, table)
            options = dict(
                rows=rows, token_start=start, budget=indexer.topk_tokens // indexer.index_kpool, block_size=block_size
            )
            if indexer.index_kpool != 4 or not native.available(*inputs, **options):
                native.fallbacks += 1
                return implementation(self, query, *args)
            result = native.plan(*inputs, **options)
            active.append(result)
            try:
                return implementation(self, query, *args)
            finally:
                active.pop()

        bindings.bind(owner, method, forward)
