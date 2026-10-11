# SPDX-License-Identifier: Apache-2.0
"""Bounded binary expert packets: FP16 inputs, FP32 outputs, no pickled objects."""

import hashlib
import json
import math
import socket
import struct
import time
from dataclasses import dataclass, replace

from .placement import ExpertPlacement
from .reference import ExpertReply, ExpertRequest, Route

MAGIC = b"SCARD001"
HALF_REPLY_MAGIC = b"SCARD002"
REDUCED_MAGIC = b"SCARD003"
PREFIX = struct.Struct("!8sII")
METADATA_LIMIT = 2048
IDENTITY_LIMIT = 256
ROUTE_STRUCT = struct.Struct("<III")
TOKEN_STRUCT = struct.Struct("<I")


def unique_metadata(pairs):
    result = {}
    for name, value in pairs:
        if name in result:
            raise ValueError("duplicate control metadata field")
        result[name] = value
    return result


@dataclass(frozen=True)
class WireIdentity:
    placement_sha256: str
    checkpoint_sha256: str

    def __post_init__(self):
        for value in (self.placement_sha256, self.checkpoint_sha256):
            if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                raise ValueError("wire identity requires lowercase SHA256 digests")

    @classmethod
    def for_placement(cls, placement: ExpertPlacement, checkpoint_sha256: str):
        canonical = json.dumps(placement.to_dict(), sort_keys=True, separators=(",", ":")).encode()
        return cls(hashlib.sha256(canonical).hexdigest(), checkpoint_sha256)


@dataclass(frozen=True)
class PacketLimits:
    hidden_width: int
    max_tokens: int = 640
    max_routes: int = 5120
    experts: int = 288
    top_k: int = 8

    def __post_init__(self):
        if any(type(value) is not int or value <= 0 for value in self.__dict__.values()):
            raise ValueError("packet limits require positive integers")

    @property
    def max_body_bytes(self):
        request = self.max_tokens * (TOKEN_STRUCT.size + self.hidden_width * 2)
        return max(request, self.max_routes * self.hidden_width * 4) + self.max_routes * ROUTE_STRUCT.size


def _integer(value, stop):
    return type(value) is int and 0 <= value < stop


def validate_control(packet, limits):
    """Validate identities and row/route mappings without expanding tensor bytes."""
    if not isinstance(packet, (ExpertRequest, ExpertReply)):
        raise ValueError("unsupported packet type")
    for value in (packet.execution_id, packet.owner):
        if not isinstance(value, str) or not 1 <= len(value.encode()) <= IDENTITY_LIMIT:
            raise ValueError("invalid packet identity length")
    if not _integer(packet.layer_id, 2**32):
        raise ValueError("invalid packet layer")
    if not 1 <= len(packet.routes) <= limits.max_routes:
        raise ValueError("route count exceeds limits")
    positions = set()
    for route in packet.routes:
        if not isinstance(route, Route) or not (
            _integer(route.token, limits.max_tokens)
            and _integer(route.slot, limits.top_k)
            and _integer(route.expert, limits.experts)
        ):
            raise ValueError("invalid route identity")
        key = (route.token, route.slot)
        if key in positions:
            raise ValueError("duplicate token/slot route")
        positions.add(key)
    if isinstance(packet, ExpertRequest):
        ids = packet.token_ids
        if not 1 <= len(ids) <= limits.max_tokens or any(not _integer(token, limits.max_tokens) for token in ids):
            raise ValueError("invalid token count or identity")
        if len(set(ids)) != len(ids) or set(ids) != {route.token for route in packet.routes}:
            raise ValueError("token rows do not match routes")
        expected_rows = len(ids)
    else:
        expected_rows = len(packet.routes)
    return expected_rows


def validate_packet(packet, limits):
    expected_rows = validate_control(packet, limits)
    rows = packet.hidden if isinstance(packet, ExpertRequest) else packet.outputs
    if len(rows) != expected_rows or any(
        len(row) != limits.hidden_width or any(not math.isfinite(value) for value in row) for row in rows
    ):
        raise ValueError("invalid packet tensor shape or nonfinite value")


def encode_packet(packet, identity, limits):
    validate_packet(packet, limits)
    request = isinstance(packet, ExpertRequest)
    rows = packet.hidden if request else packet.outputs
    body = b"".join(TOKEN_STRUCT.pack(token) for token in packet.token_ids) if request else b""
    body += b"".join(ROUTE_STRUCT.pack(route.token, route.slot, route.expert) for route in packet.routes)
    row_struct = struct.Struct("<" + ("e" if request else "f") * limits.hidden_width)
    try:
        tensor = b"".join(row_struct.pack(*row) for row in rows)
    except (OverflowError, struct.error) as error:
        raise ValueError("packet tensor exceeds wire dtype range") from error
    # A finite Python value can overflow the FP32 encoding to infinity.
    if any(not math.isfinite(value) for row in row_struct.iter_unpack(tensor) for value in row):
        raise ValueError("packet tensor exceeds wire dtype range")
    body += tensor
    return frame_packet(packet, identity, limits, len(rows), body)


def frame_packet(packet, identity, limits, rows, body, *, reply_fp16=False, partial_reduce=False):
    """Frame a prevalidated binary payload with the shared identity contract."""
    metadata = json.dumps(
        {
            "kind": "request" if isinstance(packet, ExpertRequest) else "reply",
            "placement_sha256": identity.placement_sha256,
            "checkpoint_sha256": identity.checkpoint_sha256,
            "execution_id": packet.execution_id,
            "layer_id": packet.layer_id,
            "owner": packet.owner,
            "width": limits.hidden_width,
            "rows": rows,
            "routes": len(packet.routes),
            "payload_sha256": hashlib.sha256(body).hexdigest(),
        },
        separators=(",", ":"),
    ).encode()
    if len(metadata) > METADATA_LIMIT or len(body) > limits.max_body_bytes:
        raise ValueError("packet exceeds byte limits")
    magic = REDUCED_MAGIC if partial_reduce else HALF_REPLY_MAGIC if reply_fp16 else MAGIC
    return PREFIX.pack(magic, len(metadata), len(body)) + metadata + body


def frame_lengths(prefix, limits, *, allow_reply_fp16=False, allow_partial_reduce=False):
    if len(prefix) != PREFIX.size:
        raise ValueError("truncated frame prefix")
    magic, metadata_bytes, body_bytes = PREFIX.unpack(prefix)
    supported = (MAGIC,) + (HALF_REPLY_MAGIC,) if allow_reply_fp16 else (MAGIC,)
    if allow_partial_reduce:
        supported += (REDUCED_MAGIC,)
    max_body = (
        limits.max_routes * (limits.hidden_width * 2 + ROUTE_STRUCT.size)
        if magic == HALF_REPLY_MAGIC
        else limits.max_body_bytes
    )
    if magic == REDUCED_MAGIC:
        max_body = max(
            limits.max_tokens * (limits.hidden_width * 2 + 4) + limits.max_routes * 16,
            limits.max_tokens * limits.hidden_width * 4 + limits.max_routes * 12,
        )
    if magic not in supported or not 1 <= metadata_bytes <= METADATA_LIMIT or body_bytes > max_body:
        raise ValueError("unsupported frame or oversized allocation")
    return metadata_bytes, body_bytes


def decode_control(frame, identity, limits, *, reply_fp16=False, partial_reduce=False):
    """Authenticate framing and mappings, retaining immutable tensor storage."""
    metadata_bytes, body_bytes = frame_lengths(
        frame[: PREFIX.size], limits, allow_reply_fp16=reply_fp16, allow_partial_reduce=partial_reduce
    )
    if len(frame) != PREFIX.size + metadata_bytes + body_bytes:
        raise ValueError("truncated frame or trailing bytes")
    try:
        metadata = json.loads(frame[PREFIX.size : PREFIX.size + metadata_bytes], object_pairs_hook=unique_metadata)
    except (ValueError, UnicodeError) as error:
        raise ValueError("invalid frame metadata") from error
    required = {
        "kind",
        "placement_sha256",
        "checkpoint_sha256",
        "execution_id",
        "layer_id",
        "owner",
        "width",
        "rows",
        "routes",
        "payload_sha256",
    }
    if not isinstance(metadata, dict) or set(metadata) != required:
        raise ValueError("unsupported metadata schema")
    if (
        metadata["placement_sha256"] != identity.placement_sha256
        or metadata["checkpoint_sha256"] != identity.checkpoint_sha256
    ):
        raise ValueError("worker placement/checkpoint identity mismatch")
    request = metadata["kind"] == "request"
    magic = PREFIX.unpack(frame[: PREFIX.size])[0]
    if magic != (REDUCED_MAGIC if partial_reduce else MAGIC if request or not reply_fp16 else HALF_REPLY_MAGIC):
        raise ValueError("packet reply precision differs from configured transport")
    if (
        metadata["kind"] not in ("request", "reply")
        or type(metadata["width"]) is not int
        or metadata["width"] != limits.hidden_width
    ):
        raise ValueError("unsupported packet kind or width")
    rows, routes = metadata["rows"], metadata["routes"]
    max_rows = limits.max_tokens if request or partial_reduce else limits.max_routes
    if not _integer(rows, max_rows + 1) or not rows or not _integer(routes, limits.max_routes + 1) or not routes:
        raise ValueError("packet dimensions exceed limits")
    token_bytes = rows * TOKEN_STRUCT.size if request else 0
    route_bytes = routes * ROUTE_STRUCT.size
    weight_bytes = routes * 4 if request and partial_reduce else 0
    tensor_bytes = rows * limits.hidden_width * (2 if request or reply_fp16 else 4)
    if body_bytes != token_bytes + route_bytes + weight_bytes + tensor_bytes or (
        not request and not partial_reduce and rows != routes
    ):
        raise ValueError("declared shape does not match payload")
    body = memoryview(frame)[PREFIX.size + metadata_bytes :]
    if hashlib.sha256(body).hexdigest() != metadata["payload_sha256"]:
        raise ValueError("packet payload checksum mismatch")
    ids = tuple(value[0] for value in TOKEN_STRUCT.iter_unpack(body[:token_bytes]))
    route_values = tuple(
        Route(*values) for values in ROUTE_STRUCT.iter_unpack(body[token_bytes : token_bytes + route_bytes])
    )
    args = (metadata["execution_id"], metadata["layer_id"], metadata["owner"], route_values)
    packet = ExpertRequest(*args, ids, ()) if request else ExpertReply(*args, ())
    validate_control(packet, limits)
    if partial_reduce:
        if not request and rows != len({route.token for route in route_values}):
            raise ValueError("partial reply token rows differ from routes")
        return (
            packet,
            body[token_bytes + route_bytes + weight_bytes :],
            body[token_bytes + route_bytes : token_bytes + route_bytes + weight_bytes],
        )
    return packet, body[token_bytes + route_bytes :]


def decode_packet(frame, identity, limits):
    packet, tensor = decode_control(frame, identity, limits)
    request = isinstance(packet, ExpertRequest)
    row_struct = struct.Struct("<" + ("e" if request else "f") * limits.hidden_width)
    values = tuple(row_struct.iter_unpack(tensor))
    packet = replace(packet, **({"hidden": values} if request else {"outputs": values}))
    validate_packet(packet, limits)
    return packet


def _read_exact(connection: socket.socket, count: int, deadline: float):
    chunks = bytearray()
    while len(chunks) < count:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("expert exchange deadline expired")
        connection.settimeout(remaining)
        chunk = connection.recv(count - len(chunks))
        if not chunk:
            raise EOFError("expert connection closed during frame")
        chunks.extend(chunk)
    return bytes(chunks)


def receive_frame(connection, limits, deadline, *, allow_reply_fp16=False, allow_partial_reduce=False):
    prefix = _read_exact(connection, PREFIX.size, deadline)
    metadata_bytes, body_bytes = frame_lengths(
        prefix, limits, allow_reply_fp16=allow_reply_fp16, allow_partial_reduce=allow_partial_reduce
    )
    frame = prefix + _read_exact(connection, metadata_bytes + body_bytes, deadline)
    return frame


def receive_packet(connection, identity, limits, deadline):
    return decode_packet(receive_frame(connection, limits, deadline), identity, limits)
