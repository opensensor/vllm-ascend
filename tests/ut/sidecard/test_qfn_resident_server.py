# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Private admin authentication and close-before-engine-shutdown ordering."""

import asyncio
from contextlib import asynccontextmanager, contextmanager
from types import ModuleType, SimpleNamespace

import pytest

from tools.sidecard import qfn_resident_server as server

TOKEN = "ab" * server.TOKEN_BYTES


def closed_workers():
    return [{"rank": rank, "enabled": True, "state": "closed"} for rank in range(4)]


class Engine:
    def __init__(self, *, fail=False):
        self.events = []
        self.fail = fail

    async def pause_generation(self, **kwargs):
        self.events.append(("pause", kwargs))

    async def collective_rpc(self, method, **kwargs):
        self.events.append((method, kwargs))
        if self.fail:
            raise RuntimeError("peer retirement failed")
        return closed_workers()

    def shutdown(self, timeout=None):
        self.events.append(("shutdown", timeout))


@pytest.mark.parametrize("path", ["/health", "/v1/models", "/v1/chat/completions", "/tokenize"])
def test_public_inference_does_not_need_admin_token(path):
    assert asyncio.run(request(path, "192.168.53.10", []))[0]["status"] == 204


async def request(path, peer, headers, *, method="POST", middleware=None):
    async def app(scope, receive, send):
        await send({"type": "http.response.start", "status": 204})

    messages = []

    async def send(message):
        messages.append(message)

    async def receive():
        return {"type": "http.request", "body": b""}

    middleware = middleware or server.AuthenticatedResidentMiddleware(app, token=TOKEN)
    await middleware(
        {"type": "http", "path": path, "client": (peer, 1234), "headers": headers, "method": method}, receive, send
    )
    return messages


@pytest.mark.parametrize(
    "path", ["/collective_rpc", "/pause", "/resume", "/resident/maintenance", "/future_admin", "/qfn_engine_control"]
)
@pytest.mark.parametrize("peer", ["127.0.0.1", "192.168.53.10"])
def test_admin_requires_token_even_from_loopback(path, peer):
    messages = asyncio.run(request(path, peer, [(b"x-forwarded-for", b"127.0.0.1")]))
    assert messages[0]["status"] == 403


@pytest.mark.parametrize("peer", ["127.0.0.1", "::1"])
def test_local_authenticated_rpc_is_admitted(peer):
    messages = asyncio.run(request("/collective_rpc", peer, [(server.TOKEN_HEADER, TOKEN.encode())]))
    assert messages[0]["status"] == 204


def test_remote_peer_cannot_use_valid_token_or_forwarded_loopback():
    headers = [(server.TOKEN_HEADER, TOKEN.encode()), (b"x-forwarded-for", b"127.0.0.1")]
    assert asyncio.run(request("/collective_rpc", "192.168.53.10", headers))[0]["status"] == 403


def test_duplicate_authentication_header_is_rejected():
    headers = [(server.TOKEN_HEADER, TOKEN.encode())] * 2
    assert asyncio.run(request("/collective_rpc", "127.0.0.1", headers))[0]["status"] == 403


def test_maintenance_needs_auth_and_still_blocks_new_public_inference():
    async def run():
        async def app(scope, receive, send):
            await send({"type": "http.response.start", "status": 204})

        middleware = server.AuthenticatedResidentMiddleware(app, token=TOKEN)
        authorized = [(server.TOKEN_HEADER, TOKEN.encode())]
        await request("/resident/maintenance", "127.0.0.1", authorized, middleware=middleware)
        assert middleware.maintenance
        messages = await request("/v1/chat/completions", "192.168.53.10", [], middleware=middleware)
        assert messages[0]["status"] == 503
        await request("/resident/maintenance", "127.0.0.1", authorized, method="DELETE", middleware=middleware)
        assert not middleware.maintenance

    asyncio.run(run())


def test_token_creation_permissions_reuse_and_symlink_rejection(tmp_path):
    tmp_path.chmod(0o700)
    token_file = tmp_path / "token"
    token = server.create_control_token(token_file)
    assert len(token) == 64
    assert server.create_control_token(token_file) == token
    assert token_file.stat().st_mode & 0o777 == 0o600
    alias = tmp_path / "alias"
    alias.symlink_to(token_file)
    with pytest.raises(OSError):
        server.read_control_token(alias)
    token_file.chmod(0o644)
    with pytest.raises(ValueError, match="0600"):
        server.read_control_token(token_file)


def test_token_parent_must_be_private(tmp_path):
    tmp_path.chmod(0o755)
    with pytest.raises(ValueError, match="0700"):
        server.create_control_token(tmp_path / "token")


def test_thread_shutdown_drains_all_ranks_first_and_closes_once(tmp_path):
    async def run():
        engine = Engine()
        guard = server.CollectiveShutdownGuard(engine, tmp_path / "closed.json")
        await asyncio.to_thread(guard.shutdown, timeout=60)
        await guard.close_collectives()
        assert [name for name, _ in engine.events] == ["pause", "npu_collective_close", "shutdown"]
        assert engine.events[0][1] == {"mode": "wait", "clear_cache": False}
        assert engine.events[-1] == ("shutdown", 60)
        assert guard.closed
        assert (tmp_path / "closed.json").exists()

    asyncio.run(run())


def test_retirement_error_prevents_ordinary_shutdown(tmp_path):
    async def run():
        engine = Engine(fail=True)
        guard = server.CollectiveShutdownGuard(engine, tmp_path / "closed.json")
        with pytest.raises(RuntimeError, match="peer retirement"):
            await asyncio.to_thread(guard.shutdown)
        assert not guard.closed
        assert all(name != "shutdown" for name, _ in engine.events)
        assert not (tmp_path / "closed.json").exists()

    asyncio.run(run())


def test_loop_thread_cannot_block_waiting_for_itself(tmp_path):
    async def run():
        engine = Engine()
        guard = server.CollectiveShutdownGuard(engine, tmp_path / "closed.json")
        with pytest.raises(RuntimeError, match="await collective"):
            guard.shutdown()
        await guard.close_collectives()
        guard.shutdown()
        assert engine.events[-1][0] == "shutdown"

    asyncio.run(run())


@pytest.mark.parametrize("workers", [[], closed_workers()[:3], [closed_workers()[0]] * 4])
def test_requires_four_distinct_close_receipts(workers):
    with pytest.raises(RuntimeError, match="4 distinct"):
        server.validate_closed_workers(workers)


def test_six_rank_close_receipts_require_every_rank():
    workers = [dict(rank=rank, enabled=True, state="closed") for rank in range(6)]
    server.validate_closed_workers(workers, expected_ranks=6)
    for invalid in (workers[:4], workers[:5] + [workers[0]]):
        with pytest.raises(RuntimeError, match="6 distinct"):
            server.validate_closed_workers(invalid, expected_ranks=6)


def test_factory_retirement_precedes_inner_shutdown_and_proxy_headers_disabled(tmp_path):
    async def run():
        engine = Engine()

        @asynccontextmanager
        async def factory(*args, **kwargs):
            try:
                yield engine
            finally:
                engine.shutdown()

        async def serve_http(*args, **kwargs):
            return kwargs

        added = []
        entry = SimpleNamespace(
            build_async_engine_client_from_engine_args=factory,
            build_app=lambda: SimpleNamespace(add_middleware=lambda cls, **kwargs: added.append((cls, kwargs))),
            serve_http=serve_http,
        )
        server.install_server_hooks(entry, token=TOKEN, receipt_path=tmp_path / "closed.json")
        async with entry.build_async_engine_client_from_engine_args() as result:
            assert result is engine
        assert [name for name, _ in engine.events] == ["pause", "npu_collective_close", "shutdown"]
        entry.build_app()
        assert added == [(server.AuthenticatedResidentMiddleware, {"token": TOKEN})]
        assert await entry.serve_http(proxy_headers=True) == {"proxy_headers": False}

    asyncio.run(run())


@pytest.mark.parametrize("url", ["http://192.168.53.187:8001", "https://127.0.0.1:8001", "http://user@127.0.0.1:8001"])
def test_client_refuses_sending_control_token_to_non_loopback_origin(url):
    with pytest.raises(ValueError, match="loopback HTTP"):
        server.QfnResidentClient(url, "/unused-token-file")


def test_main_runs_hooked_api_in_current_process(monkeypatch):
    package = ModuleType("vllm.entrypoints.launchers.api_server")
    calls = []
    package.entry = SimpleNamespace(main=lambda: calls.append(tuple(server.sys.argv)))
    monkeypatch.setitem(server.sys.modules, package.__name__, package)
    monkeypatch.setattr(server, "create_control_token", lambda path: TOKEN)
    monkeypatch.setattr(server, "install_server_hooks", lambda entry, **kwargs: calls.append((entry, kwargs)))
    monkeypatch.setattr(
        server.sys,
        "argv",
        [
            "qfn_resident_server",
            "--control-token-file",
            "/private/token",
            "--close-receipt",
            "/private/closed.json",
            "--",
            "serve",
            "/model",
            "--port",
            "8001",
        ],
    )
    server.main()
    assert calls[0][0] is package.entry
    assert calls[1] == ("qfn-resident-api", "--model", "/model", "--port", "8001")


def test_hooks_wrap_already_built_starlette_stack_without_readding_middleware(tmp_path):
    async def run():
        async def original_stack(scope, receive, send):
            await send({"type": "http.response.start", "status": 204})

        def reject_late_add(*args, **kwargs):
            raise RuntimeError("Cannot add middleware after an application has started")

        app = SimpleNamespace(middleware_stack=original_stack, add_middleware=reject_late_add, state=object())
        entry = SimpleNamespace(build_app=lambda: app, serve_http=lambda: None)
        entry.build_async_engine_client_from_engine_args = lambda: None
        server.install_server_hooks(entry, token=TOKEN, receipt_path=tmp_path / "closed.json")
        assert entry.build_app() is app
        assert isinstance(app.middleware_stack, server.AuthenticatedResidentMiddleware)
        assert app.middleware_stack.app is original_stack
        blocked = await request("/collective_rpc", "192.168.53.10", [], middleware=app.middleware_stack)
        assert blocked[0]["status"] == 403
        allowed = await request("/v1/models", "192.168.53.10", [], middleware=app.middleware_stack)
        assert allowed[0]["status"] == 204

    asyncio.run(run())


@pytest.mark.parametrize("fail", [False, True])
def test_main_holds_leases_during_initialization_and_serving(monkeypatch, fail):
    events = []

    @contextmanager
    def reserve(ids):
        assert ids == [0, 3]
        events.append("acquired")
        try:
            yield ()
        finally:
            events.append("closed")

    def main():
        events.append("serving")
        if fail:
            raise RuntimeError("engine failed")

    package = ModuleType("vllm.entrypoints.launchers.api_server")
    package.entry = SimpleNamespace(main=main)
    monkeypatch.setitem(server.sys.modules, package.__name__, package)
    monkeypatch.setattr(server, "reserve_devices", reserve)
    monkeypatch.setattr(server, "create_control_token", lambda path: events.append("token") or TOKEN)
    monkeypatch.setattr(server, "install_server_hooks", lambda *args, **kwargs: events.append("hooks"))
    monkeypatch.setattr(
        server.sys,
        "argv",
        [
            "server",
            "--control-token-file",
            "/private/token",
            "--close-receipt",
            "/private/close.json",
            "--dma-reserve-device",
            "0",
            "--dma-reserve-device",
            "3",
            "--",
            "serve",
            "/model",
        ],
    )
    if fail:
        with pytest.raises(RuntimeError, match="engine failed"):
            server.main()
    else:
        server.main()
    assert events == ["acquired", "token", "hooks", "serving", "closed"]
