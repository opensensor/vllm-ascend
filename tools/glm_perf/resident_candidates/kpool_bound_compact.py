# SPDX-License-Identifier: Apache-2.0
"""Compact pool writes on permanent indexer instances, retaining their casts."""

from pathlib import Path

import torch

from ..instance_bindings import extend_bindings
from ..integer_divide import DivisionTorch, prepare_counts, private_divisions, rewrite_pool_remainders
from ..resident_rpc_guard import WORKER_PREFIX, guard_replacements
from .kpool_completed_prefill import MAX_GRAPH_TOKENS, MAX_SPECULATIVE_TOKENS, POOL_SIZE, wrap_builder, wrap_writer

MAX_PREFILL_ROWS = 640
CAST_MODES = (0, 1, 2, 3, 4, 5)


def prepare_converters(runner):
    """Rebase tiny immutable descriptors outside retired graph allocator pools."""
    roots = [runner.model]
    draft = getattr(getattr(runner, "drafter", None), "model", None)
    if draft is not None:
        roots.append(draft)
    indexers = [
        module
        for root in roots
        for module in root.modules()
        if hasattr(module, "indexer_op") and hasattr(module, "_native_bf16_cast")
    ]
    converters = {id(module._native_bf16_cast): module._native_bf16_cast for module in indexers}
    if not converters:
        raise ValueError("no permanent converters found for compact pool writes")
    device = torch.device("npu", torch.npu.current_device())
    for converter in converters.values():
        keys = set(converter.configs)
        for indexer in indexers:
            if indexer._native_bf16_cast is converter:
                widths = (
                    indexer.rope_dim,
                    indexer.head_dim,
                    POOL_SIZE * indexer.head_dim,
                    indexer.n_head * indexer.head_dim,
                )
                keys.update(
                    (device, rows * width, mode)
                    for rows in range(1, MAX_PREFILL_ROWS + 1)
                    for width in widths
                    for mode in CAST_MODES
                )
        if any(key[0] != device for key in keys):
            raise ValueError("permanent converter descriptors belong to another device")
        keys = sorted(keys, key=lambda key: (key[1], key[2]))
        backing = torch.tensor([(count, mode) for _, count, mode in keys], dtype=torch.int64, device=device)
        converter.configs = {key: backing[index] for index, key in enumerate(keys)}
    return len(converters)


def extend_replacements(changes, native):
    from vllm_ascend.attention.indexer_kpool import AscendIndexerKPoolMetadataBuilder
    from vllm_ascend.models.glm5next import sparse_attn_indexer_kpool as module

    prepare_counts(native, range(1, MAX_PREFILL_ROWS + 1))
    authoritative_writer_source = Path(module.__file__).read_text()
    proxy = DivisionTorch(native)
    state = {"compact_calls": 0, "fallback_calls": 0, "prepared_converters": 0}

    def writer_factory(original):
        compressor = original.__globals__.get("compress_kpool")
        if not callable(compressor):
            raise ValueError("permanent pool writer has no bound compressor")
        integer_writer = rewrite_pool_remainders(original, authoritative_writer_source, proxy)
        compact = wrap_writer(
            integer_writer,
            module._cache_tensor,
            module._masked_storage_write,
            compressor,
            preserve_cache_dtype=True,
            integer_ops=proxy,
        )

        def write(self, keys, gates, ape, positions, pool_size, indexer_metadata, state_metadata):
            eligible = (
                getattr(indexer_metadata, "_glm_completed_pool_plan", None) is not None
                and keys.shape[0] > MAX_GRAPH_TOKENS
                and pool_size == POOL_SIZE
                and 0 <= getattr(self.tail_cache, "num_speculative_tokens", 0) <= MAX_SPECULATIVE_TOKENS
            )
            state["compact_calls" if eligible else "fallback_calls"] += 1
            return compact(self, keys, gates, ape, positions, pool_size, indexer_metadata, state_metadata)

        return write

    def selector_factory(original):
        selected = private_divisions(original, proxy)
        for name in ("dense_kpool_token_indices", "select_kpool_groups"):
            if name in selected.__globals__:
                selected.__globals__[name] = private_divisions(selected.__globals__[name], proxy)
        return selected

    result = dict(changes)
    target = "vllm_ascend.attention.indexer_kpool:AscendIndexerKPoolMetadataBuilder.build"
    result[target] = wrap_builder(result.get(target, AscendIndexerKPoolMetadataBuilder.build))
    result = extend_bindings(result, writer_factory, selector_factory)
    capture_base = result[WORKER_PREFIX + "resident_capture"]

    def capture(self):
        try:
            state["prepared_converters"] = prepare_converters(self.model_runner)
            return capture_base(self)
        except Exception as error:
            self._resident_session().graphs_dirty = True
            return self._resident_error(error)

    result[WORKER_PREFIX + "resident_capture"] = capture
    parent_status = result[WORKER_PREFIX + "resident_status"]

    def status(self):
        receipt = parent_status(self)
        receipt["compact_bound_pools"] = dict(state, retained_cache_dtype=True)
        receipt["integer_divide"] = dict(
            calls=native.calls, descriptors=len(native.configs), prepared_outside_capture=True
        )
        return receipt

    result[WORKER_PREFIX + "resident_status"] = status
    return guard_replacements(result)
