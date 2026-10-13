# SPDX-License-Identifier: Apache-2.0
"""Bounded aligned prompt prefixes with complete recurrent and KV snapshots."""

import json

MAX_PREFIXES = 2
MAX_PREFIX_BYTES = 2 << 30


class PromptPrefix:
    """Restore prompt boundaries without changing captured tensor addresses.

    Independent KV/pool copies survive unrelated requests. Recurrent attention
    also needs state/conv copies and compressed-pool tails. Chunk-aligned
    boundaries preserve the original prefill arithmetic and snapshot memory
    is bounded. The caller can retain the stable context before a new message.
    """

    def __init__(self, engine):
        self.engine = engine
        self.entries = []

    @property
    def entry(self):
        return self.entries[-1] if self.entries else None

    def snapshot(self, tokens, logits, sparse):
        layers = []
        for layer in self.engine.layers.values():
            if layer.linear:
                values = (layer.state.clone(), layer.conv.clone())
            else:
                indexer = layer.indexer
                values = (
                    indexer.pool_count,
                    indexer.tail_length,
                    indexer.tail_keys.clone(),
                    indexer.tail_gates.clone(),
                    layer.cache[: layer.length].clone(),
                    indexer.pools[: indexer.pool_count].clone(),
                )
            layers.append((layer, layer.length, values))
        return (tuple(tokens), sparse, logits.clone(), layers)

    @staticmethod
    def size(entry):
        tensors = [entry[2]] + [value for _, _, values in entry[3] for value in values if hasattr(value, "numel")]
        return sum(value.numel() * value.element_size() for value in tensors)

    def restore(self, entry):
        tokens, _, logits, layers = entry
        for layer, length, values in layers:
            layer.length = length
            if layer.linear:
                layer.state.copy_(values[0])
                layer.conv.copy_(values[1])
            else:
                indexer = layer.indexer
                indexer.pool_count, indexer.tail_length = values[:2]
                indexer.tail_keys.copy_(values[2])
                indexer.tail_gates.copy_(values[3])
                layer.cache[:length].copy_(values[4])
                indexer.pools[: indexer.pool_count].copy_(values[5])
        self.engine.position = len(tokens)
        return logits

    def prefill(self, tokens, context_limit, *, cancelled=None, boundary=None):
        engine = self.engine
        chunk = engine.prefill_chunk_size
        boundary = len(tokens) // chunk * chunk if boundary is None else boundary
        if type(boundary) is not int or not 0 <= boundary <= len(tokens) or boundary % chunk:
            raise ValueError("prefix boundary must be an admitted complete prefill chunk")
        sparse = context_limit > engine.config["index_topk"]
        matches = [entry for entry in self.entries if entry[1] == sparse and tuple(tokens[: len(entry[0])]) == entry[0]]
        entry = max(matches, key=lambda value: len(value[0])) if matches else None
        cached = len(entry[0]) if entry is not None else 0
        common = max(
            (
                next(
                    (index for index, (left, right) in enumerate(zip(value[0], tokens)) if left != right),
                    min(len(value[0]), len(tokens)),
                )
                for value in self.entries
            ),
            default=0,
        )
        print(
            json.dumps(
                {
                    "kind": "prefill_started",
                    "prompt_tokens": len(tokens),
                    "cached_prompt_tokens": cached,
                    "common_prefix_tokens": common,
                    "snapshot_boundary": boundary,
                    "retained_prefixes": len(self.entries),
                }
            ),
            flush=True,
        )
        try:
            # Reset also checks process health and applies the new admission/mode.
            engine.reset(context_limit=context_limit)
            logits = self.restore(entry) if cached else None
            candidate = entry
            overlap = engine.spec.get("prefill_overlap", True)
            if boundary > cached:
                logits = engine.prefill(tokens[cached:boundary], overlap=overlap, cancelled=cancelled)
                candidate = self.snapshot(tokens[:boundary], logits, sparse)
            if max(boundary, cached) < len(tokens):
                logits = engine.prefill(tokens[max(boundary, cached) :], overlap=overlap, cancelled=cancelled)
            if candidate is not None:
                self.entries = [value for value in self.entries if value is not entry] + [candidate]
                while len(self.entries) > MAX_PREFIXES or sum(map(self.size, self.entries)) > MAX_PREFIX_BYTES:
                    # Preserve expensive contexts when a short background request arrives.
                    # Equal-length entries retain their least-recently-used ordering.
                    oldest_shortest = min(range(len(self.entries)), key=lambda index: len(self.entries[index][0]))
                    self.entries.pop(oldest_shortest)
            return logits, cached
        except BaseException:
            self.entries.clear()
            raise
