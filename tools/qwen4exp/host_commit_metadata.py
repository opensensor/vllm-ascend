# SPDX-License-Identifier: Apache-2.0
"""Avoid unused legacy metadata uploads for qualified 310P commit consumers."""

import hashlib
import inspect
from types import FunctionType

QUALIFIED_STAGE_SHA256 = "0b447e32547a5690f5207d6b4410682624a15773336ea3bd1d17d9e8491927ac"
FIELDS = ("mamba_state_idx_buf", "num_scheduled_tokens_buf", "num_computed_tokens_buf", "num_draft_tokens_buf")


def without_uploads(original, *, source=None):
    """Retain all upstream host validation/packing, remove four unused copies."""
    source = inspect.getsource(original) if source is None else source
    if hashlib.sha256(source.encode()).hexdigest() != QUALIFIED_STAGE_SHA256:
        raise RuntimeError("host commit staging requires the exact qualified source")
    for name in FIELDS:
        statement = f"    ctx.{name}.copy_to_gpu(num_reqs)\n"
        if source.count(statement) != 1:
            raise RuntimeError("qualified commit upload seam differs")
        source = source.replace(statement, "")
    namespace = dict(original.__globals__)
    exec(compile(source, "<qfn-host-commit-metadata>", "exec"), namespace)
    compiled = namespace[original.__name__]
    return FunctionType(compiled.__code__, original.__globals__, compiled.__name__, compiled.__defaults__)


class HostCommitMetadata:
    def __init__(self, context, original):
        self.context, self.original = context, original
        self.host_stage = without_uploads(original)
        self.enabled = False
        self.calls = self.avoided_uploads = 0

    def stage(self, ctx, *args, **kwargs):
        if not self.enabled or ctx is not self.context:
            return self.original(ctx, *args, **kwargs)
        result = self.host_stage(ctx, *args, **kwargs)
        self.calls += 1
        self.avoided_uploads += len(FIELDS)
        return result

    def status(self):
        return dict(
            enabled=self.enabled, calls=self.calls, avoided_uploads=self.avoided_uploads, host_metadata_retained=True
        )
