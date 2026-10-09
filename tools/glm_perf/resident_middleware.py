# SPDX-License-Identifier: Apache-2.0
"""Expose inference publicly while restricting resident control to loopback."""

import ipaddress
import json

INFERENCE_PATHS = frozenset({"/v1/chat/completions", "/v1/completions"})
MAINTENANCE_PATH = "/resident/maintenance"


class ResidentControlMiddleware:
    def __init__(self, app):
        self.app = app
        self.maintenance = False

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            # Public inference and health routes only. Every administrative or
            # future debug route stays local; forwarded headers are not trusted.
            path = scope.get("path", "")
            public = path in {
                "/health",
                "/v1/models",
                "/v1/chat/completions",
                "/v1/completions",
                "/tokenize",
                "/detokenize",
            }
            try:
                local = ipaddress.ip_address(scope.get("client", ("", 0))[0]).is_loopback
            except (ValueError, TypeError):
                local = False
            if path == MAINTENANCE_PATH and local:
                method = scope.get("method", "GET")
                if method == "POST":
                    self.maintenance = True
                elif method == "DELETE":
                    self.maintenance = False
                elif method != "GET":
                    await self.respond(send, 405, {"error": "use GET, POST or DELETE"})
                    return
                await self.respond(send, 200, {"maintenance": self.maintenance})
                return
            if self.maintenance and path in INFERENCE_PATHS and not local:
                # Hold new public requests out before draining the scheduler.
                # Loopback remains available for the controlled benchmark;
                # existing requests must still finish before worker reset.
                await self.respond(send, 503, {"error": "resident benchmark maintenance"}, retry_after=True)
                return
            if not public and not local:
                await send(
                    {"type": "http.response.start", "status": 403, "headers": [(b"content-type", b"application/json")]}
                )
                await send({"type": "http.response.body", "body": b'{"error":"local control only"}'})
                return
        await self.app(scope, receive, send)

    @staticmethod
    async def respond(send, status, payload, *, retry_after=False):
        headers = [(b"content-type", b"application/json")]
        if retry_after:
            headers.append((b"retry-after", b"30"))
        await send({"type": "http.response.start", "status": status, "headers": headers})
        await send({"type": "http.response.body", "body": json.dumps(payload).encode()})
