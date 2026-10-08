# SPDX-License-Identifier: Apache-2.0
"""Append to the candidate source after renaming its factory to base_replacements."""

from tools.glm_perf.resident_worker import ResidentWorkerExtension


def replacements(native_resources=None):
    changes = base_replacements(native_resources)  # noqa: F821 - appended candidate factory
    target = "vllm_ascend.models.glm5next.sparse_attn_indexer_kpool:SparseAttnIndexerKpool._write_pools"
    write = changes[target]
    counts = {"compact_calls": 0, "compact_tokens": 0, "fallback_calls": 0, "max_compact_tokens": 0}

    def audited(self, keys, gates, ape, positions, pool_size, indexer_metadata, state_metadata):
        plan = getattr(indexer_metadata, "_glm_completed_pool_plan", None)
        speculative = getattr(self.tail_cache, "num_speculative_tokens", 0)
        if plan is not None and keys.shape[0] > 8 and pool_size == 4 and 0 <= speculative <= 1:
            counts["compact_calls"] += 1
            counts["compact_tokens"] += keys.shape[0]
            counts["max_compact_tokens"] = max(counts["max_compact_tokens"], keys.shape[0])
        else:
            counts["fallback_calls"] += 1
        return write(self, keys, gates, ape, positions, pool_size, indexer_metadata, state_metadata)

    def status(self):
        receipt = ResidentWorkerExtension.resident_status(self)
        receipt["completed_pool_audit"] = dict(counts)
        return receipt

    changes[target] = audited
    changes["vllm_ascend._310p.worker_310p:NPUWorker310.resident_status"] = status
    return changes
