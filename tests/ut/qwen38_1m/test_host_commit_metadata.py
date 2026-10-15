# SPDX-License-Identifier: Apache-2.0
import hashlib
from types import SimpleNamespace

import pytest

from tools.qwen4exp import host_commit_metadata as module


def test_source_drift_is_rejected():
    with pytest.raises(RuntimeError, match="exact qualified"):
        module.without_uploads(lambda: None, source="def changed(): pass\n")


def test_transformation_preserves_host_work_and_errors(monkeypatch):
    # A small source fixture exercises removal independently of NPU libraries.
    source = "def stage(ctx, values, num_reqs):\n    ctx.validate(values)\n    ctx.values = values[:num_reqs]\n"
    source += "".join(f"    ctx.{field}.copy_to_gpu(num_reqs)\n" for field in module.FIELDS)
    namespace = {}
    exec(source, namespace)
    monkeypatch.setattr(module, "QUALIFIED_STAGE_SHA256", hashlib.sha256(source.encode()).hexdigest())
    transformed = module.without_uploads(namespace["stage"], source=source)
    uploads = []
    ctx = SimpleNamespace(
        validate=lambda value: None, **{name: SimpleNamespace(copy_to_gpu=uploads.append) for name in module.FIELDS}
    )
    transformed(ctx, [10, 20, 30], 2)
    assert ctx.values == [10, 20] and uploads == []
    namespace["stage"](ctx, [10, 20, 30], 2)
    assert uploads == [2] * 4

    def invalid(value):
        raise ValueError("invalid request metadata")

    ctx.validate = invalid
    with pytest.raises(ValueError, match="invalid request"):
        transformed(ctx, [], 2)


def test_foreign_context_and_disabled_owner_keep_original_uploads():
    owner = module.HostCommitMetadata.__new__(module.HostCommitMetadata)
    owner.context = object()
    calls = []
    owner.original = lambda ctx, *args, **kwargs: calls.append("original")
    owner.host_stage = lambda ctx, *args, **kwargs: calls.append("host")
    owner.calls = owner.avoided_uploads = 0
    owner.enabled = False
    owner.stage(owner.context)
    owner.enabled = True
    owner.stage(object())
    owner.stage(owner.context)
    assert calls == ["original", "original", "host"]
    assert owner.calls == 1 and owner.avoided_uploads == 4


def test_failed_packing_does_not_count_removed_uploads():
    owner = module.HostCommitMetadata.__new__(module.HostCommitMetadata)
    owner.context = object()
    owner.enabled = True
    owner.calls = owner.avoided_uploads = 0

    def fail(*args):
        raise ValueError("packing failed")

    owner.host_stage = fail
    with pytest.raises(ValueError, match="packing failed"):
        owner.stage(owner.context)
    assert owner.calls == owner.avoided_uploads == 0
