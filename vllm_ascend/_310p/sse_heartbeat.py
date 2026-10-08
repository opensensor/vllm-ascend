# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Keep long 310P chat prefills from leaving SSE connections idle."""

import asyncio
from contextlib import suppress

from starlette.types import ASGIApp, Message, Receive, Scope, Send


class SSEHeartbeatMiddleware:
    """Send SSE comments while a chat completion has no output to stream.

    The first model token can take several minutes on a long 310P prefill.
    SSE comments are ignored by clients but keep the HTTP response active.
    The middleware is opt-in through vLLM's ``--middleware`` argument.
    """

    def __init__(self, app: ASGIApp, interval_seconds: float = 15.0) -> None:
        if interval_seconds <= 0:
            raise ValueError("interval_seconds must be positive")
        self.app = app
        self.interval_seconds = interval_seconds

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope["path"] != "/v1/chat/completions":
            await self.app(scope, receive, send)
            return

        finished = asyncio.Event()
        send_lock = asyncio.Lock()
        heartbeat_task: asyncio.Task[None] | None = None

        async def heartbeat() -> None:
            try:
                while not finished.is_set():
                    with suppress(TimeoutError):
                        await asyncio.wait_for(finished.wait(), self.interval_seconds)
                    if finished.is_set():
                        break
                    async with send_lock:
                        if not finished.is_set():
                            await send(
                                {
                                    "type": "http.response.body",
                                    "body": b": keepalive\n\n",
                                    "more_body": True,
                                }
                            )
            except OSError:
                finished.set()

        async def send_with_heartbeat(message: Message) -> None:
            nonlocal heartbeat_task
            if message["type"] == "http.response.start":
                async with send_lock:
                    await send(message)
                content_type = next(
                    (value for key, value in message.get("headers", []) if key.lower() == b"content-type"),
                    b"",
                )
                if message["status"] == 200 and content_type.startswith(b"text/event-stream"):
                    heartbeat_task = asyncio.create_task(heartbeat())
                return
            if message["type"] == "http.response.body":
                async with send_lock:
                    if not message.get("more_body", False):
                        finished.set()
                    await send(message)
                return
            await send(message)

        try:
            await self.app(scope, receive, send_with_heartbeat)
        finally:
            finished.set()
            if heartbeat_task is not None:
                heartbeat_task.cancel()
                with suppress(asyncio.CancelledError):
                    await heartbeat_task
