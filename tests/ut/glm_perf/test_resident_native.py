# SPDX-License-Identifier: Apache-2.0
import asyncio
import copy
import json

import pytest

from tools.glm_perf.resident_middleware import ResidentControlMiddleware
from tools.glm_perf.resident_native import NativeManifest, NativeSession, file_digest


@pytest.fixture
def manifest(tmp_path):
    library = tmp_path / "v1.so"
    library.write_bytes(b"test library")
    binary = tmp_path / "v1.bin"
    binary.write_bytes(b"test binary")
    return {
        "name": "test_v1",
        "libraries": [{"path": str(library), "sha256": file_digest(library)}],
        "assets": [{"path": str(binary), "sha256": file_digest(binary)}],
        "operators": ["test_v1::launch"],
        "validation_source": "def validate():\n    return {'passed': True}\n",
    }


def test_native_load_is_idempotent_and_keeps_resource(manifest):
    manifest["validation_source"] = (
        "def prepare():\n    return object()\ndef validate(x):\n    return {'passed': x is not None}\n"
    )
    session = NativeSession()
    operations = set()
    events = []
    info = session.prepare(manifest, operations.__contains__)
    assert not events and not operations

    def load(path):
        events.append(path)
        operations.add("test_v1::launch")

    receipt = session.load(info["native_digest"], load, operations.__contains__, lambda: events.append("sync"))
    assert receipt["validation"]["passed"] and session.resources["test_v1"] is not None
    assert not session.failed
    count = len(events)
    info = session.prepare(manifest, operations.__contains__)
    assert info["already_loaded"]
    assert session.load(info["native_digest"], load, operations.__contains__, lambda: None) == receipt
    assert len(events) == count


@pytest.mark.parametrize("problem", ["digest", "collision", "relative", "duplicate", "source", "unknown"])
def test_preflight_rejects_without_loading(manifest, problem):
    value = copy.deepcopy(manifest)
    if problem == "digest":
        value["libraries"][0]["sha256"] = "0" * 64
    elif problem == "relative":
        value["libraries"][0]["path"] = "v1.so"
    elif problem == "duplicate":
        value["operators"] *= 2
    elif problem == "source":
        value["validation_source"] = ""
    elif problem == "unknown":
        value["unexpected"] = True
    session = NativeSession()
    with pytest.raises(ValueError):
        session.prepare(value, lambda op: problem == "collision")
    assert not session.failed and session.pending is None


@pytest.mark.parametrize("failure", ["load", "registration", "validation", "sync"])
def test_partial_native_mutation_is_not_rolled_back_or_retried(manifest, failure):
    if failure == "validation":
        manifest["validation_source"] = "def validate():\n    return {'passed': False}\n"
    session = NativeSession()
    info = session.prepare(manifest, lambda op: False)

    def load(path):
        if failure == "load":
            raise RuntimeError("dlopen failed")

    def sync():
        if failure == "sync":
            raise RuntimeError("device failed")

    with pytest.raises(RuntimeError):
        session.load(info["native_digest"], load, lambda op: failure != "registration", sync)
    assert session.failed and not session.loaded
    with pytest.raises(RuntimeError, match="restart"):
        session.prepare(manifest, lambda op: False)


def test_artifacts_rechecked_between_prepare_and_load(manifest):
    from pathlib import Path

    session = NativeSession()
    info = session.prepare(manifest, lambda op: False)
    Path(manifest["assets"][0]["path"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="digest"):
        session.load(info["native_digest"], lambda _: pytest.fail("must not load"), lambda op: True, lambda: None)


@pytest.mark.parametrize(
    "client,path,allowed",
    [
        ("192.168.50.211", "/v1/chat/completions", True),
        ("192.168.50.211", "/collective_rpc", False),
        ("192.168.50.211", "/resume", False),
        ("192.168.50.211", "/future_debug_route", False),
        ("127.0.0.1", "/collective_rpc", True),
        ("::1", "/collective_rpc", True),
    ],
)
def test_public_inference_local_control_only(client, path, allowed):
    messages = []

    async def app(scope, receive, send):
        await send({"status": 200})

    async def send(message):
        messages.append(message)

    scope = {"type": "http", "path": path, "client": (client, 5000), "headers": [(b"x-forwarded-for", b"127.0.0.1")]}
    asyncio.run(ResidentControlMiddleware(app)(scope, None, send))
    assert messages[0]["status"] == (200 if allowed else 403)


def test_manifest_digest_covers_validation_and_assets(manifest):
    first = NativeManifest(manifest).digest
    manifest["validation_source"] += "\n# changed\n"
    assert NativeManifest(manifest).digest != first
    assert json.loads(NativeManifest(manifest).payload) == manifest
