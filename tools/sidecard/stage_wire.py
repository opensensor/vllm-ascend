# SPDX-License-Identifier: Apache-2.0
"""Bounded complete deferred mHC state at a whole-layer architecture boundary."""

import json
import math
import struct

from .layer_block_design import GLM_HIDDEN, MAX_PACKET_ROWS, STREAMS
from .wire import unique_metadata

MAGIC = b"SSTAGE01"
PREFIX = struct.Struct("!8sII")
MAX_HEADER_BYTES = 4096
MAX_BODY_BYTES = MAX_PACKET_ROWS * (GLM_HIDDEN * 2 + STREAMS * GLM_HIDDEN * 4 + (STREAMS + STREAMS**2) * 4 + 8)
FIELDS = ("hidden_states", "residual", "post", "comb", "positions")


def descriptors(rows, residual_dtype="fp32"):
    if type(rows) is not int or not 1 <= rows <= MAX_PACKET_ROWS or residual_dtype not in ("fp16", "fp32"):
        raise ValueError("invalid complete mHC stage geometry")
    geometry = (
        ("fp16", [rows, GLM_HIDDEN], 2),
        (residual_dtype, [rows, STREAMS, GLM_HIDDEN], 2 if residual_dtype == "fp16" else 4),
        ("fp32", [rows, STREAMS, 1], 4),
        ("fp32", [rows, STREAMS, STREAMS], 4),
        ("int64", [rows], 8),
    )
    result, offset = [], 0
    for name, (dtype, shape, size) in zip(FIELDS, geometry, strict=True):
        count = math.prod(shape) * size
        result.append(dict(name=name, dtype=dtype, shape=shape, offset=offset, bytes=count))
        offset += count
    return result, offset


def metadata(rows, sequence, checkpoint_sha256, *, residual_dtype="fp32"):
    fields, size = descriptors(rows, residual_dtype)
    if (
        type(sequence) is not int
        or sequence < 0
        or not isinstance(checkpoint_sha256, str)
        or len(checkpoint_sha256) != 64
        or any(c not in "0123456789abcdef" for c in checkpoint_sha256)
    ):
        raise ValueError("invalid stage sequence or checkpoint identity")
    return dict(
        schema_version=1,
        sequence=sequence,
        checkpoint_sha256=checkpoint_sha256,
        rows=rows,
        residual_dtype=residual_dtype,
        fields=fields,
        payload_bytes=size,
    )


def validate_metadata(value):
    if not isinstance(value, dict):
        raise ValueError("stage metadata must be an object")
    try:
        expected = metadata(
            value["rows"], value["sequence"], value["checkpoint_sha256"], residual_dtype=value["residual_dtype"]
        )
    except KeyError as error:
        raise ValueError("incomplete stage metadata") from error
    if value != expected:
        raise ValueError("missing deferred mHC field, altered dtype, or noncanonical byte range")
    return expected


def send(connection, value, body):
    validate_metadata(value)
    header = json.dumps(value, allow_nan=False, separators=(",", ":")).encode()
    if len(header) > MAX_HEADER_BYTES or len(body) != value["payload_bytes"]:
        raise ValueError("stage frame violates header or body bounds")
    connection.sendall(PREFIX.pack(MAGIC, len(header), len(body)))
    connection.sendall(header)
    connection.sendall(body)


def receive_into(connection, target, *, checkpoint_sha256, sequence):
    def read(count):
        result = bytearray(count)
        view = memoryview(result)
        while view:
            n = connection.recv_into(view)
            if not n:
                raise EOFError("stage peer closed")
            view = view[n:]
        return bytes(result)

    magic, header_bytes, body_bytes = PREFIX.unpack(read(PREFIX.size))
    if (
        magic != MAGIC
        or not 0 < header_bytes <= MAX_HEADER_BYTES
        or not 0 < body_bytes <= min(MAX_BODY_BYTES, len(target))
    ):
        raise ValueError("oversized or invalid stage frame prefix")
    value = json.loads(read(header_bytes), object_pairs_hook=unique_metadata)
    validate_metadata(value)
    if (
        value["payload_bytes"] != body_bytes
        or value["checkpoint_sha256"] != checkpoint_sha256
        or value["sequence"] != sequence
    ):
        raise ValueError("stage frame identity, sequence, or byte count differs")
    view = target[:body_bytes]
    while view:
        n = connection.recv_into(view)
        if not n:
            raise EOFError("stage peer closed during state transfer")
        view = view[n:]
    return value
