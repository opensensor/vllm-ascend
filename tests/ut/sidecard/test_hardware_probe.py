# SPDX-License-Identifier: Apache-2.0
"""Offline safety and lifetime gates for the accelerator copy probe."""

import ctypes
import hashlib
import socket
import struct
import threading
import time

import pytest

from tools.sidecard.hardware_probe import (
    HIDDEN_WIDTH,
    MAGIC,
    MAX_BODY_BYTES,
    MAX_METADATA_BYTES,
    PREFIX,
    CudaCopies,
    fixture,
    receive_frame,
    send_frame,
    serve_worker,
    validate_shape,
)


@pytest.mark.parametrize("rows,fanout", [(0, 1), (641, 1), (True, 1), (1, 0), (1, 9), (1, True)])
def test_probe_rejects_unbounded_shapes(rows, fanout):
    with pytest.raises(ValueError):
        validate_shape(rows, fanout)


def test_fixture_matches_independent_fp32_conversion_and_route_order():
    hidden, expected = fixture(3, 8)
    rows = struct.iter_unpack("<e", hidden)
    converted = [struct.pack("<f", value[0]) for value in rows]
    reference = b"".join(b"".join(converted[row * HIDDEN_WIDTH : (row + 1) * HIDDEN_WIDTH]) * 8 for row in range(3))
    assert expected == reference
    assert hidden[2:4] == b"\x00\x80"
    assert expected[4:8] == b"\x00\x00\x00\x80"


@pytest.mark.parametrize(
    "prefix",
    [
        PREFIX.pack(b"BADMAGIC", 1, 0),
        PREFIX.pack(MAGIC, MAX_METADATA_BYTES + 1, 0),
        PREFIX.pack(MAGIC, 1, MAX_BODY_BYTES + 1),
    ],
)
def test_probe_rejects_prefix_before_reading_payload(prefix):
    left, right = socket.socketpair()
    try:
        left.sendall(prefix)
        with pytest.raises(ValueError, match="prefix"):
            receive_frame(right, time.monotonic() + 1)
    finally:
        left.close()
        right.close()


def test_probe_duplicate_metadata_rejected():
    left, right = socket.socketpair()
    header = b'{"rows":1,"rows":2}'
    try:
        left.sendall(PREFIX.pack(MAGIC, len(header), 0) + header)
        with pytest.raises(ValueError):
            receive_frame(right, time.monotonic() + 1)
    finally:
        left.close()
        right.close()


class FixtureBackend:
    info = {"device": "cpu-fixture"}

    def transform(self, body, rows, fanout):
        hidden, expected = fixture(rows, fanout)
        assert body == hidden
        return expected, {"h2d_ms": 0.0, "cast_repeat_ms": 0.0, "d2h_ms": 0.0}


def test_probe_worker_persistent_buffers_identity_and_graceful_shutdown():
    parent, child = socket.socketpair()
    errors = []

    def run():
        try:
            with child:
                serve_worker(child, FixtureBackend(), 2)
        except BaseException as error:
            errors.append(error)

    thread = threading.Thread(target=run)
    thread.start()
    try:
        deadline = time.monotonic() + 2
        ready, body = receive_frame(parent, deadline)
        assert ready["status"] == "ready" and not body
        retained = []
        for sequence, rows in enumerate((1, 4)):
            hidden, expected = fixture(rows, 8)
            request = {"sequence": sequence, "rows": rows, "fanout": 8, "sha256": hashlib.sha256(hidden).hexdigest()}
            send_frame(parent, request, hidden, deadline)
            reply, body = receive_frame(parent, deadline)
            assert reply["sequence"] == sequence
            assert reply["sha256"] == hashlib.sha256(expected).hexdigest()
            assert body == expected
            retained.append(body)
        assert retained[0] == fixture(1, 8)[1]
        send_frame(parent, {"status": "stop"}, b"", deadline)
        assert receive_frame(parent, deadline) == ({"status": "stopped"}, b"")
    finally:
        parent.close()
        thread.join(3)
    assert not thread.is_alive()
    assert not errors


@pytest.mark.parametrize("failure", ["checksum", "shape", "sequence", "extra_field"])
def test_probe_worker_rejects_bad_copy_before_backend(failure):
    parent, child = socket.socketpair()
    errors = []

    def run():
        try:
            with child:
                serve_worker(child, FixtureBackend(), 2)
        except ValueError as error:
            errors.append(error)

    thread = threading.Thread(target=run)
    thread.start()
    try:
        deadline = time.monotonic() + 2
        receive_frame(parent, deadline)
        hidden, _ = fixture(1, 1)
        request = {"sequence": 0, "rows": 1, "fanout": 1, "sha256": hashlib.sha256(hidden).hexdigest()}
        if failure == "checksum":
            request["sha256"] = "0" * 64
        elif failure == "shape":
            request["rows"] = 2
        elif failure == "sequence":
            request["sequence"] = 1
        else:
            request["extra"] = 1
        send_frame(parent, request, hidden, deadline)
        with pytest.raises(EOFError):
            receive_frame(parent, deadline)
    finally:
        parent.close()
        thread.join(3)
    assert not thread.is_alive()
    assert len(errors) == 1


def test_probe_fragmentation_and_deadline():
    left, right = socket.socketpair()
    try:
        left.sendall(PREFIX.pack(MAGIC, 2, 4) + b"{}" + b"1234")
        assert receive_frame(right, time.monotonic() + 1) == ({}, b"1234")
        with pytest.raises(TimeoutError):
            receive_frame(right, time.monotonic() - 1)
    finally:
        left.close()
        right.close()


def test_cuda_probe_copy_fences_and_allocations_are_scoped():
    class FakeRuntime:
        def __init__(self):
            self.events = []
            self.memory = {}

        def cudaMemcpy(self, destination, source, size, direction):
            self.events.append(("copy", direction))
            if direction == 1:
                self.memory[destination.value] = ctypes.string_at(source, size)
            else:
                ctypes.memmove(destination, self.memory[source.value], size)
            return 0

        def cudaDeviceSynchronize(self):
            self.events.append(("fence",))
            return 0

        def cudaFree(self, pointer):
            self.events.append(("free", pointer.value))
            return 0

    copies = CudaCopies.__new__(CudaCopies)
    copies.library = FakeRuntime()
    copies.input, copies.output = ctypes.c_void_p(1), ctypes.c_void_p(2)
    copies.max_input, copies.max_output = 4, 8
    first, _ = copies.stage(b"abcd")
    second, _ = copies.stage(b"efghijkl", output=True)
    assert first == b"abcd" and second == b"efghijkl"
    assert copies.library.events == [("copy", 1), ("fence",), ("copy", 2), ("fence",)] * 2
    with pytest.raises(ValueError, match="allocation"):
        copies.stage(b"oversized")
    copies.close()
    copies.close()
    assert copies.library.events[-2:] == [("free", 2), ("free", 1)]


def test_empty_control_body_does_not_send_after_receiver_can_close():
    class SendRecorder:
        def __init__(self):
            self.parts = []

        def settimeout(self, timeout):
            assert timeout > 0

        def sendall(self, data):
            assert data, "empty send can race the receiver closing its completed stream"
            self.parts.append(data)

    connection = SendRecorder()
    send_frame(connection, {"status": "stopped"}, b"", time.monotonic() + 1)
    assert len(connection.parts) == 2
