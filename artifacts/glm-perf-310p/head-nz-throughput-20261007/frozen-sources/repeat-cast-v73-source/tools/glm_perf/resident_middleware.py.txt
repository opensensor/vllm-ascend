# SPDX-License-Identifier: Apache-2.0
"""Expose inference publicly while restricting resident control to loopback."""

import ipaddress


class ResidentControlMiddleware:
    def __init__(self, app):
        self.app = app

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
            if not public and not local:
                await send(
                    {"type": "http.response.start", "status": 403, "headers": [(b"content-type", b"application/json")]}
                )
                await send({"type": "http.response.body", "body": b'{"error":"local control only"}'})
                return
        await self.app(scope, receive, send)
