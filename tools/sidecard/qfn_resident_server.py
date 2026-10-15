# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Keep QFN loaded across authenticated resident changes and safe shutdown.

Cold initialization only: wrap the qualified vLLM API factory and serve_http.
Generation and graph replay acquire no new observers or Python callbacks.
Public OpenAI routes retain their existing identity. Administrative routes need
both a real loopback peer and a per-service token; proxy headers are disabled.
"""

import argparse
import asyncio
import functools
import hmac
import ipaddress
import json
import os
import secrets
import stat
import sys
import threading
import urllib.parse
import urllib.request
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path

from tools.glm_perf.resident_harness import ResidentClient
from tools.glm_perf.resident_middleware import ResidentControlMiddleware
from tools.qwen4exp.ple_driver import reserve_devices

PUBLIC_PATHS = frozenset(
    {"/health", "/v1/models", "/v1/chat/completions", "/v1/completions", "/tokenize", "/detokenize"}
)
TOKEN_HEADER = b"x-qfn-resident-token"
TOKEN_BYTES = 32
EXPECTED_RANKS = 4
CLOSE_TIMEOUT_SECONDS = 300


def read_control_token(path):
    """Read a caller-owned 0600 regular file without following symlinks."""
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(descriptor) as stream:
        metadata = os.fstat(stream.fileno())
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != os.getuid()
            or stat.S_IMODE(metadata.st_mode) != 0o600
        ):
            raise ValueError("resident token must be a caller-owned regular file with mode 0600")
        token = stream.read(TOKEN_BYTES * 2 + 2).strip()
    if len(token) != TOKEN_BYTES * 2 or any(character not in "0123456789abcdef" for character in token):
        raise ValueError("resident token file must contain one 256-bit lowercase hexadecimal token")
    return token


def create_control_token(path):
    path = Path(path)
    # The launcher creates a private run directory. Refuse an exposed parent,
    # rather than changing permissions on any existing operator directory.
    parent = path.parent.stat()
    if parent.st_uid != os.getuid() or stat.S_IMODE(parent.st_mode) != 0o700:
        raise ValueError("resident token parent must be caller-owned with mode 0700")
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    except FileExistsError:
        return read_control_token(path)
    with os.fdopen(descriptor, "w") as stream:
        stream.write(secrets.token_hex(TOKEN_BYTES) + "\n")
    return read_control_token(path)


class AuthenticatedResidentMiddleware(ResidentControlMiddleware):
    """Reuse maintenance behavior; authenticate every non-inference route."""

    def __init__(self, app, *, token):
        super().__init__(app)
        self.token = token.encode("ascii")

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http" and scope.get("path", "") not in PUBLIC_PATHS:
            try:
                local = ipaddress.ip_address(scope.get("client", ("", 0))[0]).is_loopback
            except (ValueError, TypeError):
                local = False
            headers = [value for key, value in scope.get("headers", ()) if key.lower() == TOKEN_HEADER]
            if not local or len(headers) != 1 or not hmac.compare_digest(headers[0], self.token):
                await self.respond(send, 403, {"error": "authenticated local control only"})
                return
        await super().__call__(scope, receive, send)


class QfnResidentClient(ResidentClient):
    """Existing resident transactions, with credentials read from a local file."""

    def __init__(self, base_url, token_file, expected_workers=EXPECTED_RANKS, timeout=900):
        parsed = urllib.parse.urlsplit(base_url)
        try:
            local = ipaddress.ip_address(parsed.hostname or "").is_loopback
        except ValueError:
            local = False
        if (
            not local
            or parsed.scheme != "http"
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
            or parsed.path not in ("", "/")
        ):
            raise ValueError("resident client requires an explicit loopback HTTP origin")
        super().__init__(base_url, expected_workers=expected_workers, timeout=timeout)
        self.token = read_control_token(token_file)

    def request(self, path, payload=None, method="POST"):
        request = urllib.request.Request(
            self.base_url + path,
            data=json.dumps(payload).encode() if payload is not None else None,
            headers={"Content-Type": "application/json", TOKEN_HEADER.decode(): self.token},
            method=method,
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            data = response.read()
        return json.loads(data) if data else None


def validate_closed_workers(workers):
    if (
        not isinstance(workers, list)
        or len(workers) != EXPECTED_RANKS
        or any(not isinstance(worker, dict) for worker in workers)
        or {worker.get("rank") for worker in workers} != set(range(EXPECTED_RANKS))
        or any(worker.get("enabled") is not True or worker.get("state") != "closed" for worker in workers)
    ):
        raise RuntimeError("custom collective shutdown requires four distinct closed worker receipts")


class CollectiveShutdownGuard:
    """Drain/retire peers before any upstream synchronous engine shutdown."""

    def __init__(self, engine, receipt_path):
        self.engine = engine
        self.receipt_path = Path(receipt_path)
        self.loop = asyncio.get_running_loop()
        self.loop_thread = threading.get_ident()
        self.original_shutdown = engine.shutdown
        self.lock = asyncio.Lock()
        self.closed = False
        self.close_error = None

    async def close_collectives(self):
        async with self.lock:
            if self.closed:
                return
            try:
                # Actual service shutdown only. Graph-changing resident reload
                # instead uses the existing clear-cache/reset/capture workflow.
                await self.engine.pause_generation(mode="wait", clear_cache=False)
                workers = await self.engine.collective_rpc("npu_collective_close", timeout=CLOSE_TIMEOUT_SECONDS)
                validate_closed_workers(workers)
                self.receipt_path.write_text(json.dumps({"workers": workers, "closed": True}, indent=2) + "\n")
                self.closed = True
            except Exception as error:
                self.close_error = error
                # Never call ordinary shutdown/free through a failed custom
                # drain. The still-running process retains worker ownership.
                raise

    def shutdown(self, timeout=None):
        if not self.closed:
            if threading.get_ident() == self.loop_thread:
                raise RuntimeError("await collective retirement before loop-thread engine shutdown")
            future = asyncio.run_coroutine_threadsafe(self.close_collectives(), self.loop)
            future.result()
        return self.original_shutdown(timeout=timeout)


def install_server_hooks(entry, *, token, receipt_path, engine_control=False):
    """Patch one API process before construction; preserve public HTTP routing."""
    original_factory = entry.build_async_engine_client_from_engine_args
    original_build_app = entry.build_app
    original_serve_http = entry.serve_http
    if engine_control:
        from tools.sidecard.qfn_engine_control import install_engine_launch_hook

        install_engine_launch_hook()

    @asynccontextmanager
    async def factory(*args, **kwargs):
        async with original_factory(*args, **kwargs) as engine:
            guard = CollectiveShutdownGuard(engine, receipt_path)
            engine.shutdown = guard.shutdown
            try:
                yield engine
            finally:
                # Upstream's SIGTERM path invokes shutdown from a thread before
                # returning here. Other successful exit paths retire here first.
                await guard.close_collectives()

    @functools.wraps(original_build_app)
    def build_app(*args, **kwargs):
        app = original_build_app(*args, **kwargs)
        if engine_control:
            from tools.sidecard.qfn_engine_control import install_engine_route

            install_engine_route(app)
        # The qualified builder eagerly materializes Starlette's stack before
        # returning. add_middleware rejects that state even though no HTTP
        # request has run. Wrap the already-built stack during cold setup;
        # retain the FastAPI object and app.state expected by vLLM.
        if getattr(app, "middleware_stack", None) is None:
            app.add_middleware(AuthenticatedResidentMiddleware, token=token)
        else:
            app.middleware_stack = AuthenticatedResidentMiddleware(app.middleware_stack, token=token)
        return app

    @functools.wraps(original_serve_http)
    async def serve_http(*args, **kwargs):
        # Middleware must see the TCP peer, even if a caller sets forwarded
        # allowlists in its environment. X-Forwarded-For cannot grant control.
        kwargs["proxy_headers"] = False
        return await original_serve_http(*args, **kwargs)

    entry.build_async_engine_client_from_engine_args = factory
    entry.build_app = build_app
    entry.serve_http = serve_http


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-token-file", type=Path, required=True)
    parser.add_argument("--close-receipt", type=Path, required=True)
    parser.add_argument("--engine-control", action="store_true", help="Enable authenticated, drained engine updates")
    parser.add_argument(
        "--dma-reserve-device",
        type=int,
        action="append",
        default=[],
        metavar="ID",
        help="Reserve one PCIe DMA queue on this driver device at startup (repeat per device)",
    )
    parser.add_argument("server_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    server_args = args.server_args
    if server_args and server_args[0] == "--":
        server_args = server_args[1:]
    if not server_args or server_args[0] != "serve":
        parser.error("pass -- serve MODEL and the qualified server arguments")
    if len(server_args) < 2 or server_args[1].startswith("-"):
        parser.error("serve requires a positional model path")
    with reserve_devices(args.dma_reserve_device) as leases:
        if leases:
            print(
                json.dumps(
                    {"device_dma_reservations": [asdict(lease.info) for lease in leases], "device_submit": False}
                ),
                flush=True,
            )
        token = create_control_token(args.control_token_file)
        # API process isolation: this module's tests and resident client do not
        # import vLLM/Torch or initialize the NPU runtime.
        from vllm.entrypoints.launchers.api_server import entry

        install_server_hooks(entry, token=token, receipt_path=args.close_receipt, engine_control=args.engine_control)
        # Invoke the in-process API entry directly. CLI ServeSubcommand may spawn
        # frontends (e.g. Rust/multi-API mode), which would lose these cold hooks.
        # EngineCore and TP workers still use the qualified subprocess executor.
        sys.argv = ["qfn-resident-api", "--model", server_args[1], *server_args[2:]]
        entry.main()


if __name__ == "__main__":
    main()
