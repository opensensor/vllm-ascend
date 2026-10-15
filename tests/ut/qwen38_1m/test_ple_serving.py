# SPDX-License-Identifier: Apache-2.0
"""Admission tests for the serving boundary and shared-table lifetime."""

import copy
import fcntl
import os
import sys
from types import ModuleType, SimpleNamespace

import pytest

from tools.qwen4exp.apply_ple_serving import apply, engine_source
from tools.qwen4exp.ple_serving import load_library, open_table, request_count
from tools.qwen4exp.ple_sources import IMMUTABLE_SEALS


@pytest.mark.parametrize("tokens", [1, 2, 3, 6, 9, 12])
def test_decode_layout_including_empty_padding_groups(tokens):
    history = SimpleNamespace(shape=(5, 2), ndim=2)
    starts = SimpleNamespace(numel=lambda: 6)
    assert request_count(tokens, starts, history, [0, tokens, tokens, tokens, tokens, tokens]) == 4


@pytest.mark.parametrize("tokens,end", [(13, 13), (2048, 2048), (0, 0), (6, 3)])
def test_prefill_and_fifth_padding_group_fall_back(tokens, end):
    history = SimpleNamespace(shape=(5, 2), ndim=2)
    starts = SimpleNamespace(numel=lambda: 6)
    assert request_count(tokens, starts, history, [0, end, end, end, end, tokens]) is None


def test_missing_host_layout_falls_back():
    assert request_count(3, None, None, None) is None


def test_shared_mapping_survives_owner_descriptor_close():
    original = os.memfd_create("ple-test", os.MFD_ALLOW_SEALING)
    try:
        os.write(original, b"x" * 320)
        fcntl.fcntl(original, fcntl.F_ADD_SEALS, IMMUTABLE_SEALS)
        info = os.fstat(original)
        metadata = dict(fd_path=f"/proc/self/fd/{original}", device=info.st_dev, inode=info.st_ino, bytes=320)
        owned = open_table(metadata)
        with pytest.raises(ValueError, match="identity changed"):
            open_table(dict(metadata, inode=info.st_ino + 1))
    finally:
        os.close(original)
    try:
        assert os.pread(owned, 320, 0) == b"x" * 320
        assert fcntl.fcntl(owned, fcntl.F_GET_SEALS) & IMMUTABLE_SEALS == IMMUTABLE_SEALS
    finally:
        os.close(owned)


def test_reject_library_change_before_loading(tmp_path):
    library = tmp_path / "library.so"
    library.write_bytes(b"not the admitted library")
    with pytest.raises(ValueError, match="digest differs"):
        load_library(library, "0" * 64)


@pytest.mark.parametrize("changed", [None, "digest", "graphs_dirty"])
def test_attachment_preserves_resident_stack_and_never_recaptures(tmp_path, changed):
    state = dict(
        rank=0,
        pid=123,
        weight_storage_digest="weights",
        digest="optimized",
        candidate="qualified_stack",
        generation="old",
        graphs_dirty=False,
        native_failed=False,
    )

    class Client:
        def __init__(self):
            self.paths = []
            self.reads = 0
            self.resumed = False

        def rpc(self, method):
            assert method == "resident_status"
            self.reads += 1
            result = copy.deepcopy(state)
            if self.reads > 1 and changed:
                result[changed] = True if changed == "graphs_dirty" else "lost_stack"
            return [result]

        def request(self, path, payload=None):
            self.paths.append(path)
            assert path in ("/pause?mode=wait&clear_cache=false", "/qfn_engine_control")
            return dict(passed=True)

        def resume(self):
            self.resumed = True
            return dict(status="resumed")

    client = Client()
    if changed:
        with pytest.raises(RuntimeError, match="keep paused"):
            apply(client, engine_source("off"), tmp_path / "receipt")
        assert not client.resumed
    else:
        apply(client, engine_source("off"), tmp_path / "receipt")
        assert client.resumed
    assert (tmp_path / "receipt/receipt.json").exists()


def test_engine_attachment_retains_owners_and_existing_rpc(monkeypatch):
    target = ModuleType("ple_attachment_test")
    original = lambda: "original"
    target.forward = original
    monkeypatch.setitem(sys.modules, target.__name__, target)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(npu=SimpleNamespace(synchronize=lambda: None)))
    adapter = """
class ServingPLE:
    def __init__(self, config):
        self.lookups = {}
        self.failed = self.busy = False
        self.enabled = self.verify = True
        self.checked = 1
        self.elapsed, self.native_elapsed = [], []
    def status(self):
        return dict(enabled=self.enabled, verify=self.verify)
def Lookup(*args):
    return object()
def make_replacements(config, *, manager):
    return {"ple_attachment_test:forward": lambda: "native"}
"""
    worker = SimpleNamespace(
        rank=0,
        model_runner=SimpleNamespace(execute_model_state=None, device=SimpleNamespace(index=0)),
        qfn_cpu_profile_stop=original,
    )
    namespace = {}
    exec(engine_source("install", adapter, {}), namespace)
    assert namespace["operation"](worker)["passed"]
    state = worker._qfn_native_ple_serving
    assert len(state["manager"].lookups) == 4
    assert state["originals"][0][2] is original
    assert worker.qfn_cpu_profile_stop is original
    assert target.forward() == "native"
    for action in ("off", "on", "verify"):
        exec(engine_source(action), namespace)
        status = namespace["operation"](worker)["native_ple"]
        assert status["enabled"] == (action != "off")
        assert status["verify"] == (action == "verify")
        assert worker._qfn_native_ple_serving is state
        assert worker.qfn_cpu_profile_stop is original
    exec(engine_source("install", adapter, {}), namespace)
    with pytest.raises(RuntimeError, match="already exists"):
        namespace["operation"](worker)
