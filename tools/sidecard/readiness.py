# SPDX-License-Identifier: Apache-2.0
"""Bounded startup acknowledgement after resident expert coverage validation."""

import json
import time

from .wire import METADATA_LIMIT, PREFIX, _read_exact, unique_metadata

READY_MAGIC = b"READY001"
MAX_ERROR_CHARACTERS = 128


class WorkerStartupError(RuntimeError):
    pass


def send_ready(connection, owner, identity, timeout_seconds, error=None):
    metadata = json.dumps(
        {
            "status": "ready" if error is None else "failed",
            "owner": owner,
            "placement_sha256": identity.placement_sha256,
            "checkpoint_sha256": identity.checkpoint_sha256,
        },
        separators=(",", ":"),
    ).encode()
    if error is not None:
        value = json.loads(metadata)
        value["error"] = str(error)[:MAX_ERROR_CHARACTERS]
        metadata = json.dumps(value, separators=(",", ":")).encode()
    if len(metadata) > METADATA_LIMIT:
        raise ValueError("oversized readiness metadata")
    connection.settimeout(timeout_seconds)
    connection.sendall(PREFIX.pack(READY_MAGIC, len(metadata), 0) + metadata)


def receive_ready(connection, owner, identity, deadline):
    prefix = _read_exact(connection, PREFIX.size, deadline)
    magic, metadata_bytes, body_bytes = PREFIX.unpack(prefix)
    if magic != READY_MAGIC or not 1 <= metadata_bytes <= METADATA_LIMIT or body_bytes != 0:
        raise ValueError("invalid worker readiness frame")
    metadata = json.loads(_read_exact(connection, metadata_bytes, deadline), object_pairs_hook=unique_metadata)
    expected = {
        "status": "ready",
        "owner": owner,
        "placement_sha256": identity.placement_sha256,
        "checkpoint_sha256": identity.checkpoint_sha256,
    }
    if isinstance(metadata, dict) and metadata.get("status") == "failed":
        failed = dict(expected, status="failed", error=metadata.get("error"))
        if (
            metadata != failed
            or not isinstance(metadata.get("error"), str)
            or len(metadata["error"]) > MAX_ERROR_CHARACTERS
        ):
            raise ValueError("worker failure identity mismatch")
        raise WorkerStartupError(f"worker {owner} failed during preparation: {metadata['error']}")
    if metadata != expected or time.monotonic() > deadline:
        raise ValueError("worker readiness identity mismatch")
