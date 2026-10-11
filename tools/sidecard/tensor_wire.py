# SPDX-License-Identifier: Apache-2.0
"""SCARD001 tensor storage without per-element Python float conversion.

The wire bytes and validation requirements match the reference protocol.
NumPy is loaded only in the worker/serving path; payloads remain immutable.
"""

import socket
import time
from dataclasses import dataclass

from .readiness import send_ready
from .reference import ExpertReply, ExpertRequest
from .runtime import EXCHANGE_TIMEOUT_SECONDS
from .wire import PREFIX, ROUTE_STRUCT, TOKEN_STRUCT, decode_control, frame_packet, receive_frame, validate_control


@dataclass(frozen=True)
class TensorPacket:
    control: ExpertRequest | ExpertReply
    tensor: bytes


def validate_tensor(packet, limits):
    import numpy as np

    rows = validate_control(packet.control, limits)
    request = isinstance(packet.control, ExpertRequest)
    if (packet.control.hidden if request else packet.control.outputs) != ():
        raise ValueError("raw tensor control must not carry a second tensor representation")
    item_bytes = 2 if request else 4
    if not isinstance(packet.tensor, bytes) or len(packet.tensor) != rows * limits.hidden_width * item_bytes:
        raise ValueError("raw tensor storage differs from declared geometry")
    if not np.isfinite(np.frombuffer(packet.tensor, dtype="<f2" if request else "<f4")).all():
        raise ValueError("nonfinite raw tensor storage")


def encode_tensor(packet, identity, limits):
    validate_tensor(packet, limits)
    control = packet.control
    request = isinstance(control, ExpertRequest)
    body = b"".join(TOKEN_STRUCT.pack(token) for token in control.token_ids) if request else b""
    body += b"".join(ROUTE_STRUCT.pack(route.token, route.slot, route.expert) for route in control.routes)
    body += packet.tensor
    return frame_packet(control, identity, limits, validate_control(control, limits), body)


def decode_tensor(frame, identity, limits):
    control, tensor = decode_control(frame, identity, limits)
    packet = TensorPacket(control, bytes(tensor))
    validate_tensor(packet, limits)
    return packet


class TensorSocketPeer:
    def __init__(self, connection, identity, limits, timeout_seconds):
        from .runtime import SocketPeer

        self.socket_peer = SocketPeer(connection, identity, limits, timeout_seconds)

    def exchange(self, packet):
        peer = self.socket_peer
        with peer.lock:
            try:
                if not isinstance(packet.control, ExpertRequest):
                    raise ValueError("tensor peer requires an expert request")
                deadline = time.monotonic() + peer.timeout_seconds
                peer.connection.settimeout(peer.timeout_seconds)
                peer.connection.sendall(encode_tensor(packet, peer.identity, peer.limits))
                reply = decode_tensor(receive_frame(peer.connection, peer.limits, deadline), peer.identity, peer.limits)
                request = packet.control
                if not isinstance(reply.control, ExpertReply) or (
                    reply.control.execution_id,
                    reply.control.layer_id,
                    reply.control.owner,
                    reply.control.routes,
                ) != (request.execution_id, request.layer_id, request.owner, request.routes):
                    raise ValueError("raw expert reply identity or route order differs")
                return reply
            except Exception:
                peer.close()
                raise

    def close(self):
        self.socket_peer.close()


def serve_tensor_worker(connection, worker, identity):
    """Serve already admitted CUDA banks; malformed tensors close the stream."""
    try:
        send_ready(connection, worker.owner, identity, EXCHANGE_TIMEOUT_SECONDS)
        while True:
            connection.settimeout(None)
            if not connection.recv(PREFIX.size, socket.MSG_PEEK):
                return True
            deadline = time.monotonic() + EXCHANGE_TIMEOUT_SECONDS
            packet = decode_tensor(receive_frame(connection, worker.limits, deadline), identity, worker.limits)
            control = packet.control
            if not isinstance(control, ExpertRequest) or control.owner != worker.owner:
                raise ValueError("tensor worker received foreign request")
            owners = worker.placement.layer(control.layer_id)
            if any(route.expert >= len(owners) or owners[route.expert] != worker.owner for route in control.routes):
                raise ValueError("tensor worker received foreign expert")
            reply = worker.backend.compute_tensor(packet)
            if reply.control != ExpertReply(control.execution_id, control.layer_id, control.owner, control.routes, ()):
                raise ValueError("tensor backend returned mismatched control")
            connection.settimeout(EXCHANGE_TIMEOUT_SECONDS)
            connection.sendall(encode_tensor(reply, identity, worker.limits))
    except (EOFError, OSError, ValueError, OverflowError):
        return False
    finally:
        connection.close()
