# SPDX-License-Identifier: Apache-2.0
"""A cancelled stream must release the generator's serialized request lock."""

import threading

import anyio

from tools.sidecard.layer_server import managed_deltas


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
