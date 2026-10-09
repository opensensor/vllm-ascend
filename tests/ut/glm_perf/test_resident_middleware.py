# SPDX-License-Identifier: Apache-2.0
"""Keep arriving public requests out of a drain/reset/capture transaction."""

import asyncio
import json

import pytest

from tools.glm_perf.resident_middleware import ResidentControlMiddleware


def request(middleware, path, *, host="127.0.0.1", method="GET", headers=()):
    messages = []

    async def send(message):
        messages.append(message)

    async def receive():
        return {"type": "http.request", "body": b""}

    asyncio.run(
        middleware(
            {"type": "http", "path": path, "client": (host, 1000), "method": method, "headers": headers},
            receive,
            send,
        )
    )
    return messages


@pytest.fixture
def middleware():
    async def app(scope, receive, send):
        await send({"type": "http.response.start", "status": 204, "headers": []})

    return ResidentControlMiddleware(app)


@pytest.mark.parametrize("host", ["127.0.0.1", "::1"])
def test_local_maintenance_holds_public_inference_but_allows_benchmark_and_health(middleware, host):
    enabled = request(middleware, "/resident/maintenance", host=host, method="POST")
    assert json.loads(enabled[1]["body"]) == {"maintenance": True}
    for path in ("/v1/chat/completions", "/v1/completions"):
        blocked = request(middleware, path, host="192.168.50.211", method="POST")
        assert blocked[0]["status"] == 503
        assert (b"retry-after", b"30") in blocked[0]["headers"]
        assert request(middleware, path, host=host, method="POST")[0]["status"] == 204
    assert request(middleware, "/health", host="192.168.50.211")[0]["status"] == 204
    assert json.loads(request(middleware, "/resident/maintenance", host=host)[1]["body"])["maintenance"]
    request(middleware, "/resident/maintenance", host=host, method="DELETE")
    assert request(middleware, "/v1/completions", host="192.168.50.211", method="POST")[0]["status"] == 204


def test_external_client_cannot_enable_disable_or_spoof_maintenance(middleware):
    for method in ("POST", "DELETE", "GET"):
        assert (
            request(
                middleware,
                "/resident/maintenance",
                host="192.168.50.211",
                method=method,
                headers=((b"x-forwarded-for", b"127.0.0.1"),),
            )[0]["status"]
            == 403
        )
    assert not middleware.maintenance


def test_unsupported_method_does_not_change_gate(middleware):
    assert request(middleware, "/resident/maintenance", method="PATCH")[0]["status"] == 405
    assert not middleware.maintenance
