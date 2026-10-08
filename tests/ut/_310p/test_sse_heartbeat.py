# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio

from vllm_ascend._310p.sse_heartbeat import SSEHeartbeatMiddleware


def test_chat_stream_emits_comments_before_first_model_event() -> None:
    async def run() -> list[dict]:
        messages: list[dict] = []

        async def send(message: dict) -> None:
            messages.append(message)

        async def app(scope: dict, receive, send_response) -> None:
            await send_response(
                {
                    "type": "http.response.start",
                    "status": 200,
                    "headers": [(b"content-type", b"text/event-stream; charset=utf-8")],
                }
            )
            await asyncio.sleep(0.04)
            await send_response(
                {
                    "type": "http.response.body",
                    "body": b'data: {"choices":[]}\n\n',
                    "more_body": True,
                }
            )
            await send_response({"type": "http.response.body", "body": b"data: [DONE]\n\n"})

        middleware = SSEHeartbeatMiddleware(app, interval_seconds=0.01)
        await middleware({"type": "http", "path": "/v1/chat/completions"}, None, send)
        return messages

    messages = asyncio.run(run())
    bodies = [message for message in messages if message["type"] == "http.response.body"]
    assert bodies[0]["body"] == b": keepalive\n\n"
    assert sum(message["body"] == b": keepalive\n\n" for message in bodies) >= 1
    assert bodies[-2]["body"] == b'data: {"choices":[]}\n\n'
    assert bodies[-1]["body"] == b"data: [DONE]\n\n"
    assert bodies[-1].get("more_body", False) is False


def test_non_streaming_response_is_unchanged() -> None:
    async def run() -> list[dict]:
        messages: list[dict] = []

        async def send(message: dict) -> None:
            messages.append(message)

        async def app(scope: dict, receive, send_response) -> None:
            await send_response(
                {
                    "type": "http.response.start",
                    "status": 200,
                    "headers": [(b"content-type", b"application/json")],
                }
            )
            await asyncio.sleep(0.03)
            await send_response({"type": "http.response.body", "body": b"{}"})

        middleware = SSEHeartbeatMiddleware(app, interval_seconds=0.01)
        await middleware({"type": "http", "path": "/v1/chat/completions"}, None, send)
        return messages

    messages = asyncio.run(run())
    assert [message["type"] for message in messages] == ["http.response.start", "http.response.body"]
    assert messages[-1]["body"] == b"{}"
