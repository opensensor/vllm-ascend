# SPDX-License-Identifier: Apache-2.0
"""A cancelled stream must release the generator's serialized request lock."""

import threading
from types import SimpleNamespace

import anyio

from tools.sidecard.layer_server import engine_lifespan, managed_deltas


def test_engine_lifespan_releases_ipc_owners_on_shutdown():
    calls = []

    async def run():
        async with engine_lifespan(SimpleNamespace(close=lambda: calls.append("release"))):
            calls.append("serving")
        assert calls == ["serving", "release"]

    anyio.run(run)


def test_engine_lifespan_releases_ipc_owners_on_error():
    calls = []

    async def run():
        try:
            async with engine_lifespan(SimpleNamespace(close=lambda: calls.append("release"))):
                raise RuntimeError("server stopped")
        except RuntimeError:
            pass
        assert calls == ["release"]

    anyio.run(run)


def test_closing_suspended_stream_releases_request_lock():
    lock = threading.Lock()

    def source():
        with lock:
            yield "first"
            yield "second"

    async def exercise():
        stream = managed_deltas(source())
        assert await anext(stream) == "first"
        assert lock.locked()
        await stream.aclose()
        assert not lock.locked()

    anyio.run(exercise)


def test_complete_stream_releases_request_lock():
    lock = threading.Lock()

    def source():
        with lock:
            yield "first"
            yield "second"

    async def exercise():
        assert [delta async for delta in managed_deltas(source())] == ["first", "second"]
        assert not lock.locked()

    anyio.run(exercise)
