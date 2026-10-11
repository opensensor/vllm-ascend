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
    reply_fp16: bool = False
    partial_reduce: bool = False
    route_weights: bytes = b""


def validate_tensor(packet, limits):
    import numpy as np

    rows = validate_control(packet.control, limits)
    request = isinstance(packet.control, ExpertRequest)
    if not isinstance(packet.route_weights, bytes):
        raise ValueError("route weight storage must be immutable bytes")
    if type(packet.partial_reduce) is not bool or (packet.partial_reduce and packet.reply_fp16):
        raise ValueError("partial reply requires FP32 output")
    if type(packet.reply_fp16) is not bool or (request and packet.reply_fp16):
        raise ValueError("FP16 reply mode cannot label a request")
    if (packet.control.hidden if request else packet.control.outputs) != ():
        raise ValueError("raw tensor control must not carry a second tensor representation")
    item_bytes = 2 if request or packet.reply_fp16 else 4
    if packet.partial_reduce:
        if request:
            if len(packet.route_weights) != len(packet.control.routes) * 4:
                raise ValueError("partial request route weights differ from routes")
            weights = np.frombuffer(packet.route_weights, dtype="<f4")
            if not np.isfinite(weights).all() or (weights < 0).any():
                raise ValueError("invalid partial request route weights")
        else:
            rows = len({route.token for route in packet.control.routes})
            if packet.route_weights:
                raise ValueError("partial reply cannot carry weights")
    elif packet.route_weights:
        raise ValueError("raw expert packets cannot carry weights")
    if not isinstance(packet.tensor, bytes) or len(packet.tensor) != rows * limits.hidden_width * item_bytes:
        raise ValueError("raw tensor storage differs from declared geometry")
    if not np.isfinite(np.frombuffer(packet.tensor, dtype="<f2" if item_bytes == 2 else "<f4")).all():
        raise ValueError("nonfinite raw tensor storage")


def encode_tensor(packet, identity, limits):
    validate_tensor(packet, limits)
    control = packet.control
    request = isinstance(control, ExpertRequest)
    body = b"".join(TOKEN_STRUCT.pack(token) for token in control.token_ids) if request else b""
    body += b"".join(ROUTE_STRUCT.pack(route.token, route.slot, route.expert) for route in control.routes)
    body += packet.route_weights
    body += packet.tensor
    rows = (
        len({r.token for r in control.routes})
        if packet.partial_reduce and not request
        else validate_control(control, limits)
    )
    return frame_packet(
        control, identity, limits, rows, body, reply_fp16=packet.reply_fp16, partial_reduce=packet.partial_reduce
    )


def decode_tensor(frame, identity, limits, *, reply_fp16=False, partial_reduce=False):
    decoded = decode_control(frame, identity, limits, reply_fp16=reply_fp16, partial_reduce=partial_reduce)
    control, tensor = decoded[:2]
    weights = bytes(decoded[2]) if partial_reduce else b""
    packet = TensorPacket(
        control,
        bytes(tensor),
        reply_fp16=reply_fp16 and isinstance(control, ExpertReply),
        partial_reduce=partial_reduce,
        route_weights=weights,
    )
    validate_tensor(packet, limits)
    return packet


class TensorSocketPeer:
    def __init__(self, connection, identity, limits, timeout_seconds, *, reply_fp16=False, partial_reduce=False):
        from .runtime import SocketPeer

        self.socket_peer = SocketPeer(connection, identity, limits, timeout_seconds)
        self.reply_fp16 = reply_fp16
        self.partial_reduce = partial_reduce

    def exchange(self, packet):
        peer = self.socket_peer
        with peer.lock:
            try:
                if not isinstance(packet.control, ExpertRequest):
                    raise ValueError("tensor peer requires an expert request")
                if packet.partial_reduce != self.partial_reduce:
                    raise ValueError("tensor peer partial combination mode differs")
                deadline = time.monotonic() + peer.timeout_seconds
                peer.connection.settimeout(peer.timeout_seconds)
                peer.connection.sendall(encode_tensor(packet, peer.identity, peer.limits))
                reply = decode_tensor(
                    receive_frame(
                        peer.connection,
                        peer.limits,
                        deadline,
                        allow_reply_fp16=self.reply_fp16,
                        allow_partial_reduce=self.partial_reduce,
                    ),
                    peer.identity,
                    peer.limits,
                    reply_fp16=self.reply_fp16,
                    partial_reduce=self.partial_reduce,
                )
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
        partial = getattr(worker.backend, "specification", {}).get("partial_reduce", False)
        send_ready(connection, worker.owner, identity, EXCHANGE_TIMEOUT_SECONDS)
        while True:
            connection.settimeout(None)
            if not connection.recv(PREFIX.size, socket.MSG_PEEK):
                return True
            deadline = time.monotonic() + EXCHANGE_TIMEOUT_SECONDS
            packet = decode_tensor(
                receive_frame(connection, worker.limits, deadline, allow_partial_reduce=partial),
                identity,
                worker.limits,
                partial_reduce=partial,
            )
            control = packet.control
            if not isinstance(control, ExpertRequest) or control.owner != worker.owner:
                raise ValueError("tensor worker received foreign request")
            owners = worker.placement.layer(control.layer_id)
            if any(route.expert >= len(owners) or owners[route.expert] != worker.owner for route in control.routes):
                raise ValueError("tensor worker received foreign expert")
            reply = worker.backend.compute_tensor(packet)
            if reply.control != ExpertReply(control.execution_id, control.layer_id, control.owner, control.routes, ()):
                raise ValueError("tensor backend returned mismatched control")
            if reply.partial_reduce != partial:
                raise ValueError("tensor backend returned mismatched reduction mode")
            connection.settimeout(EXCHANGE_TIMEOUT_SECONDS)
            connection.sendall(encode_tensor(reply, identity, worker.limits))
    except (EOFError, OSError, ValueError, OverflowError):
        return False
    finally:
        connection.close()
