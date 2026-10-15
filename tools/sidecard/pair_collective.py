# SPDX-License-Identifier: Apache-2.0
"""Experimental fixed-shape, two-rank FP32 SUM; no production admission.

The backend must enqueue every operation on one owning stream. Input producers
and output consumers must use that same stream. Imports have process-local
addresses; descriptors contain IPC keys, never transferable raw pointers.
There is no host handshake or device synchronization in submit(). Hardware
support, numerical parity and graph capture are NOT established by this module.
"""

import hashlib
import json
import os
import threading
from dataclasses import dataclass

PROTOCOL = "sidecard-pair-sum-v2"
WIDTH = 4096
MAX_ROWS = 32
MAX_PENDING = 64
BOUNDARIES = tuple(f"{layer}:{branch}" for layer in range(18, 33) for branch in ("attention", "mlp"))


class PairQuarantined(RuntimeError):
    """Retain allocations and mappings: completion or ownership is uncertain."""


@dataclass(frozen=True)
class PairConfig:
    session: str
    rows: int = 1
    link_mode: str = "hccs"
    exchange_mode: str = "push"
    reducer_identity: str = "aclnn"
    # CANN C/Python documentation disagrees on Notify timeout units. This is
    # deliberately a raw SDK value, not a claimed wall-clock bound. A separate
    # millisecond stream-drain timeout is required by the backend.
    notify_timeout_raw: int = 30
    handshake: str = "consumed"

    def __post_init__(self):
        if (
            type(self.session) is not str
            or len(self.session) != 64
            or any(c not in "0123456789abcdef" for c in self.session)
        ):
            raise ValueError("session must be a fresh 256-bit lowercase hexadecimal nonce")
        if type(self.rows) is not int or not 1 <= self.rows <= MAX_ROWS:
            raise ValueError("rows must be an integer from 1 through 32")
        if self.link_mode not in ("hccs", "pcie"):
            raise ValueError("link_mode must be hccs or pcie")
        if self.exchange_mode not in ("push", "remote-read"):
            raise ValueError("exchange_mode must be push or remote-read")
        if self.reducer_identity != "aclnn" and (
            type(self.reducer_identity) is not str
            or len(self.reducer_identity) != 64
            or any(c not in "0123456789abcdef" for c in self.reducer_identity)
        ):
            raise ValueError("reducer identity must be aclnn or an authenticated local-kernel SHA256")
        if type(self.notify_timeout_raw) is not int or not 1 <= self.notify_timeout_raw <= 30000:
            raise ValueError("notify_timeout_raw must be a nonzero bounded SDK value")
        if self.handshake not in ("consumed", "ping-pong"):
            raise ValueError("handshake must be consumed or ping-pong")

    @property
    def elements(self):
        return self.rows * WIDTH

    @property
    def nbytes(self):
        return self.elements * 4

    @property
    def identity(self):
        value = dict(
            protocol=PROTOCOL,
            session=self.session,
            rows=self.rows,
            width=WIDTH,
            dtype="float32",
            operation="sum",
            link_mode=self.link_mode,
            exchange_mode=self.exchange_mode,
            reducer_identity=self.reducer_identity,
            boundaries=BOUNDARIES,
            notify_timeout_raw=self.notify_timeout_raw,
            handshake=self.handshake,
        )
        return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


class PairEndpoint:
    """One rank's persistent lane, with explicit two-party shutdown receipts.

    backend.prepare(config, peer_pid) exports a P2P inbox and two Notify keys
    with an explicit bare-device-PID allowlist. connect imports the opposite
    keys. Backend resources stay attached after *any* partial native failure.
    The transport does not silently retry, reset a device, or fall back midway.
    """

    def __init__(self, config, rank, backend):
        if not isinstance(config, PairConfig) or type(rank) is not int or rank not in (0, 1):
            raise ValueError("a PairConfig and rank 0 or 1 are required")
        self.config, self.rank, self.backend = config, rank, backend
        self.pid, self.thread = os.getpid(), threading.get_ident()
        self.state, self.sequence, self.pending = "new", 0, 0

    def _require(self, *states):
        if self.pid != os.getpid() or self.thread != threading.get_ident():
            raise RuntimeError("pair endpoint belongs to another process or thread")
        if self.state == "quarantined":
            raise PairQuarantined("pair resources are quarantined; keep their owners alive")
        if self.state not in states:
            raise RuntimeError(f"invalid pair state: {self.state}")

    def _native(self, operation, *args):
        try:
            return getattr(self.backend, operation)(*args)
        except Exception as error:
            self.state = "quarantined"
            raise PairQuarantined(f"{operation} failed; retain pair resources") from error

    def _packet(self, phase, **values):
        return dict(
            protocol=PROTOCOL,
            config_sha256=self.config.identity,
            session=self.config.session,
            rank=self.rank,
            phase=phase,
            **values,
        )

    def _peer(self, packet, phase, fields):
        expected = self._packet(phase)
        expected["rank"] = 1 - self.rank
        if type(packet) is not dict or set(packet) != set(expected) | set(fields):
            raise ValueError("invalid peer receipt fields")
        if any(type(packet[k]) is not type(v) or packet[k] != v for k, v in expected.items()):
            raise ValueError("peer session, rank, configuration or phase differs")

    def prepare(self, peer_bare_pid):
        self._require("new")
        if type(peer_bare_pid) is not int or peer_bare_pid <= 0:
            raise ValueError("peer bare-device PID must be a positive integer")
        keys = self._native("prepare", self.config, peer_bare_pid)
        self.state = "exported"
        return self._packet("exported", keys=keys)

    def connect(self, peer_export):
        self._require("exported")
        self._peer(peer_export, "exported", ("keys",))
        self._native("connect", peer_export["keys"])
        self.state = "connected"
        return self._packet("connected")

    def activate(self, peer_connected):
        self._require("connected")
        self._peer(peer_connected, "connected", ())
        self.state = "active"

    def submit(self, sequence, boundary):
        """Queue one exchange/SUM with the configured inbox-reuse handshake.

        Each rank follows the same immutable 30-boundary schedule. Sequence
        numbers never reset within a session, including across requests. One
        fixed-shape lane is used; varying row counts need separately connected
        lanes and an agreed schedule. Concurrent host submitters are forbidden.
        """
        self._require("active")
        if type(sequence) is not int or sequence != self.sequence:
            raise ValueError("collective sequence is stale or out of order")
        if boundary != BOUNDARIES[sequence % len(BOUNDARIES)]:
            raise ValueError("collective boundary differs from the agreed schedule")
        if self.pending >= MAX_PENDING:
            raise RuntimeError("drain the bounded submission window before enqueueing more")
        # A single inbox needs an explicit CONSUMED ACK. Two alternating inbox
        # and READY slots get that proof implicitly from peer READY(s+1), which
        # precedes reuse at s+2. The backend must implement the agreed protocol.
        self._native("enqueue", self.config.notify_timeout_raw)
        self.sequence += 1
        self.pending += 1

    def synchronize(self):
        """Bounded local completion; does not prove the peer is drained."""
        self._require("active")
        self._native("synchronize")
        self.pending = 0

    def drain(self):
        self._require("active")
        self.synchronize()
        self.state = "drained"
        return self._packet("drained", next_sequence=self.sequence)

    def release_imports(self, peer_drained):
        self._require("drained")
        self._peer(peer_drained, "drained", ("next_sequence",))
        if type(peer_drained["next_sequence"]) is not int or peer_drained["next_sequence"] != self.sequence:
            raise ValueError("peer did not drain the same submission window")
        self._native("release_imports")
        self.state = "imports-closed"
        return self._packet("imports-closed", next_sequence=self.sequence)

    def close(self, peer_imports_closed):
        self._require("imports-closed")
        self._peer(peer_imports_closed, "imports-closed", ("next_sequence",))
        if (
            type(peer_imports_closed["next_sequence"]) is not int
            or peer_imports_closed["next_sequence"] != self.sequence
        ):
            raise ValueError("peer import-release receipt differs")
        self._native("close")
        self.state = "closed"
