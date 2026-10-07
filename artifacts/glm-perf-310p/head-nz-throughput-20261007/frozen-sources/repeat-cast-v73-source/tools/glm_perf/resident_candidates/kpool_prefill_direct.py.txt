# SPDX-License-Identifier: Apache-2.0
"""Prepared direct-kernel dispatch; apply only after load-native validates it."""

from functools import partial
from types import FunctionType

from tools.glm_perf.kpool_prefill import select_prefill_request
from tools.glm_perf.resident_candidates.kpool_prefill_tiled import select_tokens


def replacements(native_resources):
    op = native_resources["prefill_v1"]
    scope = dict(select_tokens.__globals__)
    scope["select_prefill_request"] = partial(select_prefill_request, native_op=op)
    replacement = FunctionType(select_tokens.__code__, scope, select_tokens.__name__)
    return {"vllm_ascend.models.glm5next.sparse_attn_indexer_kpool:SparseAttnIndexerKpool._select_tokens": replacement}
