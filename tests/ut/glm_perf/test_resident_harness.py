"""Check pause ordering, rank agreement, and failed-switch behavior."""

import json
import uuid

import pytest

from tools.glm_perf.resident_control import Control
from tools.glm_perf.resident_harness import ResidentClient
from tools.glm_perf.resident_native import NativeManifest


class Server(ResidentClient):
    def __init__(self, fail=None, paused=False):
        super().__init__("http://localhost", expected_workers=2)
        self.paused = paused
        self.calls = []
        self.fail = fail
        self.control = None

    def request(self, path, payload=None, method="POST"):
        operation = payload["method"] if path == "/collective_rpc" else path
        self.calls.append(operation)
        if operation == self.fail:
            raise OSError("injected RPC failure")
        if path == "/is_paused":
            return {"is_paused": self.paused}
        if path.startswith("/pause"):
            self.paused = True
            return {"status": "paused"}
        if path == "/resume":
            self.paused = False
            return {"status": "resumed"}
        if operation == "resident_prepare":
            assert self.paused
            self.control = Control.from_dict(json.loads(payload["args"][0]))
            results = [{"generation": self.control.generation, "digest": self.control.digest, "targets": []}] * 2
            if self.fail == "disagree":
                results[1] = {**results[1], "digest": "incorrect"}
            return {"results": results}
        if operation != "resident_status":
            assert self.paused
        setting = self.control
        return {
            "results": [
                {
                    "rank": rank,
                    "pid": 100 + rank,
                    "generation": setting.generation if setting else None,
                    "mode": setting.mode if setting else "graph",
                    "candidate": "baseline",
                    "digest": setting.digest if setting else None,
                    "graphs_dirty": False,
                    "weight_storage_digest": f"resident-storage-{rank}",
                }
                for rank in range(2)
            ]
        }


def test_switch_drains_applies_captures_and_resets_before_resuming():
    server = Server()
    results = server.switch(Control(uuid.uuid4().hex, "direct-draft"))
    assert len(results) == 2
    assert server.calls == [
        "resident_status",
        "/is_paused",
        "/pause?mode=wait&clear_cache=true",
        "resident_prepare",
        "resident_reset",
        "resident_apply",
        "resident_capture",
        "resident_reset",
        "/resume",
    ]
    assert not server.paused


@pytest.mark.parametrize("failure", ["resident_prepare", "disagree"])
def test_preparation_failure_resumes_without_applying(failure):
    server = Server(fail=failure)
    with pytest.raises((OSError, RuntimeError)):
        server.switch(Control(uuid.uuid4().hex))
    assert not server.paused
    assert "resident_apply" not in server.calls


@pytest.mark.parametrize("failure", ["resident_apply", "resident_capture", "resident_reset"])
def test_mutation_or_capture_failure_keeps_scheduler_paused(failure):
    server = Server(fail=failure)
    with pytest.raises(RuntimeError, match="remains paused"):
        server.switch(Control(uuid.uuid4().hex))
    assert server.paused
    assert "/resume" not in server.calls


def test_existing_pause_is_preserved():
    server = Server(paused=True)
    server.switch(Control(uuid.uuid4().hex))
    assert server.paused
    assert "/resume" not in server.calls
    assert "/pause?mode=wait&clear_cache=true" in server.calls


def test_missing_worker_acknowledgment_prevents_apply():
    server = Server()
    server.expected_workers = 3
    with pytest.raises(RuntimeError, match="acknowledgments"):
        server.switch(Control(uuid.uuid4().hex))
    assert "resident_apply" not in server.calls


def test_resume_rejects_disagreeing_workers():
    server = Server(paused=True)
    original = server.rpc

    def disagree(method, *args):
        results = original(method, *args)
        results[1]["mode"] = "direct-both"
        return results

    server.rpc = disagree
    with pytest.raises(RuntimeError, match="disagree"):
        server.resume()
    assert server.paused


def test_capture_error_receipts_are_collected_before_reporting_failure():
    server = Server()
    original = server.request

    def fail_capture(path, payload=None, method="POST"):
        result = original(path, payload, method)
        if payload and payload["method"] == "resident_capture":
            result["results"][1]["error"] = "capture failed"
        return result

    server.request = fail_capture
    with pytest.raises(RuntimeError, match="remains paused") as error:
        server.switch(Control(uuid.uuid4().hex))
    assert "all acknowledgments" in str(error.value.__cause__)
    assert server.paused
    assert server.calls[-1] == "resident_capture"


@pytest.mark.parametrize("failure", [None, "prepare", "load", "identity", "validation"])
def test_native_transaction_pause_identity_and_failures(failure):
    server = Server()
    manifest = NativeManifest(
        {
            "name": "native_v1",
            "libraries": [{"path": "/test.so", "sha256": "0" * 64}],
            "assets": [],
            "operators": ["native_v1::launch"],
            "validation_source": "def validate(): return {'passed': True}",
        }
    )
    original = server.request
    loaded = False

    def request(path, payload=None, method="POST"):
        nonlocal loaded
        op = payload.get("method") if payload else None
        if op in ("resident_native_prepare", "resident_native_load"):
            assert server.paused
            server.calls.append(op)
            if failure == "prepare" and op.endswith("prepare"):
                raise RuntimeError("bad hash")
            if failure == "load" and op.endswith("load"):
                raise RuntimeError("device error")
            if op.endswith("load"):
                loaded = True
            return {
                "results": [
                    {"rank": i, "native_digest": manifest.digest, "validation": {"passed": failure != "validation"}}
                    for i in range(2)
                ]
            }
        result = original(path, payload, method)
        if op == "resident_status" and loaded:
            for worker in result["results"]:
                worker["native_loaded"] = {"native_v1": {"native_digest": manifest.digest}}
                worker["native_failed"] = False
            if failure == "identity":
                result["results"][0]["weight_storage_digest"] = "changed"
        return result

    server.request = request
    if failure:
        with pytest.raises(RuntimeError):
            server.load_native(manifest)
        assert server.paused == (failure != "prepare")
    else:
        server.load_native(manifest)
        assert not server.paused
    assert "resident_apply" not in server.calls
    assert "resident_capture" not in server.calls


def test_native_failure_prevents_resume_and_dispatch_switch():
    server = Server(paused=True)
    original = server.rpc

    def failed(method, *args):
        results = original(method, *args)
        if method == "resident_status":
            results[0]["native_failed"] = True
        return results

    server.rpc = failed
    with pytest.raises(RuntimeError, match="restart"):
        server.resume()
    with pytest.raises(RuntimeError, match="restart"):
        server.switch(Control(uuid.uuid4().hex))
    assert server.paused
