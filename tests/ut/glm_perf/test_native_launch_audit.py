# SPDX-License-Identifier: Apache-2.0
import hashlib
from types import SimpleNamespace

import pytest

from tools.glm_perf.native_launch_audit import NativeLaunchAudit


def resource(tmp_path):
    calls = []
    kernel = object()
    native = SimpleNamespace(pack_kernel=kernel)
    native.launch = lambda *args: calls.append(args) or "submitted"
    (tmp_path / "glm_fused_pack.bin").write_bytes(b"versioned kernel")
    return native, calls


def test_only_live_submissions_count_and_arguments_are_preserved(tmp_path):
    native, calls = resource(tmp_path)
    logs = []
    audit = NativeLaunchAudit(native, tmp_path, lambda *args: logs.append(args))
    tensors = object()  # No shape, item or CPU-transfer API is needed.
    assert native.launch(native.pack_kernel, tensors, 8) == "submitted"
    assert not audit.snapshot()["counts"]
    audit.begin_requests()
    native.launch(native.pack_kernel, tensors, 8)
    audit.begin_requests()
    native.launch(native.pack_kernel, tensors, 8)
    assert calls == [(native.pack_kernel, tensors, 8)] * 3
    assert audit.snapshot()["counts"] == {"pack_kernel": 2}
    assert len(logs) == 1
    assert audit.binaries["pack_kernel"]["sha256"] == hashlib.sha256(b"versioned kernel").hexdigest()


def test_reinstall_unwraps_previous_audit(tmp_path):
    native, calls = resource(tmp_path)
    first = NativeLaunchAudit(native, tmp_path)
    first.begin_requests()
    second = NativeLaunchAudit(native, tmp_path)
    second.begin_requests()
    native.launch(native.pack_kernel, [], 8)
    assert len(calls) == 1 and first.counts == {} and second.counts == {"pack_kernel": 1}


def test_unregistered_kernel_rejected_before_submission(tmp_path):
    native, calls = resource(tmp_path)
    NativeLaunchAudit(native, tmp_path)
    with pytest.raises(ValueError, match="audited native resource"):
        native.launch(object(), [], 8)
    assert not calls


def test_missing_binary_rejected_before_changing_dispatch(tmp_path):
    native, _ = resource(tmp_path)
    original = native.launch
    (tmp_path / "glm_fused_pack.bin").unlink()
    with pytest.raises(FileNotFoundError):
        NativeLaunchAudit(native, tmp_path)
    assert native.launch is original
