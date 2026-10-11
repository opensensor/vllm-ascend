# SPDX-License-Identifier: Apache-2.0
"""Offline persistent process/socket execution with a fail-closed coordinator."""

import math
import multiprocessing
import socket
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress

from .readiness import receive_ready, send_ready
from .reference import ExpertReply, combine, dispatch, validate_shared_output
from .wire import PREFIX, WireIdentity, encode_packet, receive_packet
from .worker import ExpertWorker

EXCHANGE_TIMEOUT_SECONDS = 5.0
PROCESS_JOIN_SECONDS = 1.0


def serve_worker(connection, worker, identity, *, prepared=False):
    """Own a socket and resident backend; exit on malformed input or disconnect."""
    try:
        try:
            if not prepared:
                worker.prepare()
        except Exception as error:
            send_ready(connection, worker.owner, identity, EXCHANGE_TIMEOUT_SECONDS, error)
            return
        send_ready(connection, worker.owner, identity, EXCHANGE_TIMEOUT_SECONDS)
        while True:
            connection.settimeout(None)
            if not connection.recv(PREFIX.size, socket.MSG_PEEK):
                return True
            deadline = time.monotonic() + EXCHANGE_TIMEOUT_SECONDS
            request = receive_packet(connection, identity, worker.limits, deadline)
            reply = worker.execute(request)
            connection.settimeout(EXCHANGE_TIMEOUT_SECONDS)
            connection.sendall(encode_packet(reply, identity, worker.limits))
    except (EOFError, OSError, ValueError, OverflowError):
        # The coordinator treats a closed stream as failure, never a zero output.
        return False
    finally:
        connection.close()


class SocketPeer:
    def __init__(self, connection, identity, limits, timeout_seconds):
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("exchange timeout must be positive")
        self.connection = connection
        self.identity = identity
        self.limits = limits
        self.timeout_seconds = timeout_seconds
        self.lock = threading.Lock()

    def exchange(self, request):
        with self.lock:
            try:
                frame = encode_packet(request, self.identity, self.limits)
                deadline = time.monotonic() + self.timeout_seconds
                self.connection.settimeout(self.timeout_seconds)
                self.connection.sendall(frame)
                reply = receive_packet(self.connection, self.identity, self.limits, deadline)
                if not isinstance(reply, ExpertReply) or (reply.execution_id, reply.layer_id, reply.owner) != (
                    request.execution_id,
                    request.layer_id,
                    request.owner,
                ):
                    raise ValueError("unexpected expert reply identity")
                return reply
            except Exception:
                self.close()
                raise

    def close(self):
        with suppress(OSError):
            self.connection.shutdown(socket.SHUT_RDWR)
        self.connection.close()


class LocalWorkerPool:
    """Spawn one persistent CPU process per owner, with an independent backend.

    This exercises IPC and lifetime contracts. It initializes no accelerators,
    claims no overlap on device, and holds no KV or recurrent state.
    """

    def __init__(
        self,
        placement,
        limits,
        identity,
        backends,
        timeout_seconds=EXCHANGE_TIMEOUT_SECONDS,
        startup_timeout_seconds=EXCHANGE_TIMEOUT_SECONDS,
    ):
        if set(backends) != set(placement.owners):
            raise ValueError("one backend is required for each placement owner")
        if WireIdentity.for_placement(placement, identity.checkpoint_sha256) != identity:
            raise ValueError("pool identity differs from placement")
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("exchange timeout must be positive")
        if not math.isfinite(startup_timeout_seconds) or startup_timeout_seconds <= 0:
            raise ValueError("startup timeout must be positive")
        self.placement, self.limits, self.identity = placement, limits, identity
        self.peers, self.processes = {}, {}
        self.execution_prefix = uuid.uuid4().hex
        self.sequence = 0
        self.closed = False
        self.combiner = combine
        self.lock = threading.Lock()
        context = multiprocessing.get_context("spawn")
        try:
            for owner in placement.owners:
                parent, child = socket.socketpair()
                self.peers[owner] = SocketPeer(parent, identity, limits, timeout_seconds)
                worker = ExpertWorker(owner, placement, limits, backends[owner])
                process = context.Process(target=serve_worker, args=(child, worker, identity), name=f"sidecard-{owner}")
                try:
                    process.start()
                    self.processes[owner] = process
                finally:
                    child.close()
            deadline = time.monotonic() + startup_timeout_seconds
            for owner, peer in self.peers.items():
                receive_ready(peer.connection, owner, identity, deadline)
        except Exception:
            self.close()
            raise

    def execute(self, layer_id, hidden, expert_ids, weights, shared, routed_scale):
        with self.lock:
            if self.closed:
                raise RuntimeError("worker pool is closed")
            self.sequence += 1
            execution_id = f"{self.execution_prefix}:{self.sequence}"
            hidden, expert_ids, weights, shared = (
                tuple(tuple(row) for row in values) for values in (hidden, expert_ids, weights, shared)
            )
            requests = dispatch(self.placement, execution_id, layer_id, hidden, expert_ids, weights)
            validate_shared_output(hidden, shared, routed_scale)
            # Encode before any sends, so an invalid input cannot partly dispatch.
            for request in requests:
                encode_packet(request, self.identity, self.limits)
            try:
                with ThreadPoolExecutor(max_workers=len(requests)) as executor:
                    futures = [executor.submit(self.peers[request.owner].exchange, request) for request in requests]
                    replies = [future.result() for future in futures]
                return self.combiner(
                    self.placement, execution_id, layer_id, hidden, expert_ids, weights, replies, shared, routed_scale
                )
            except Exception:
                self.close()
                raise

    def close(self):
        self.closed = True
        for peer in self.peers.values():
            peer.close()
        for process in self.processes.values():
            process.join(PROCESS_JOIN_SECONDS)
            if process.is_alive():
                process.terminate()
                process.join(PROCESS_JOIN_SECONDS)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
