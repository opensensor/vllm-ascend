# SPDX-License-Identifier: Apache-2.0
"""Persistent expert workers launched in explicitly chosen Python environments."""

import json
import math
import socket
import subprocess
import threading
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path

from .readiness import receive_ready
from .reference import combine
from .runtime import LocalWorkerPool, SocketPeer
from .wire import WireIdentity

GRACEFUL_EXIT_SECONDS = 30.0
MAX_CONFIG_BYTES = 1024 * 1024


@dataclass(frozen=True)
class WorkerLaunch:
    python: Path
    backend: dict


class ExternalWorkerPool(LocalWorkerPool):
    """Reuse dispatch/combine with independent interpreters and graceful exits.

    Closing sockets makes healthy workers release their own device contexts.
    A stuck device worker is reported with its PID; this pool never resets a
    device or hard-kills a process. Configuration is trusted local JSON, and
    request/reply traffic uses the existing bounded expert protocol.
    """

    def __init__(
        self,
        placement,
        limits,
        identity,
        launches,
        log_dir,
        timeout_seconds=30.0,
        startup_timeout_seconds=120.0,
        exit_timeout_seconds=GRACEFUL_EXIT_SECONDS,
        combiner=None,
    ):
        if set(launches) != set(placement.owners):
            raise ValueError("one external launch is required for each owner")
        if WireIdentity.for_placement(placement, identity.checkpoint_sha256) != identity:
            raise ValueError("external pool identity differs from placement")
        if any(
            not math.isfinite(value) or value <= 0
            for value in (timeout_seconds, startup_timeout_seconds, exit_timeout_seconds)
        ):
            raise ValueError("external worker timeouts must be positive")
        configs = {}
        for owner, launch in launches.items():
            config = {
                "owner": owner,
                "placement": placement.to_dict(),
                "limits": asdict(limits),
                "identity": asdict(identity),
                "backend": launch.backend,
            }
            raw = json.dumps(config, allow_nan=False).encode()
            if len(raw) > MAX_CONFIG_BYTES:
                raise ValueError("external worker configuration exceeds its limit")
            configs[owner] = raw
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=False)
        self.placement, self.limits, self.identity = placement, limits, identity
        self.peers, self.processes = {}, {}
        self.execution_prefix = uuid.uuid4().hex
        self.sequence, self.closed = 0, False
        self.lock = threading.Lock()
        self.combiner = combine if combiner is None else combiner
        self.exit_timeout_seconds = exit_timeout_seconds
        self.log_dir = log_dir
        try:
            for index, owner in enumerate(placement.owners):
                config_path = log_dir / f"worker-{index:04d}.json"
                config_path.write_bytes(configs[owner])
                parent, child = socket.socketpair()
                self.peers[owner] = SocketPeer(parent, identity, limits, timeout_seconds)
                try:
                    command = [
                        str(launches[owner].python),
                        "-m",
                        "tools.sidecard.external_worker",
                        "--socket-fd",
                        str(child.fileno()),
                        "--config",
                        str(config_path.resolve()),
                    ]
                    with (log_dir / f"worker-{index:04d}.log").open("xb") as log:
                        self.processes[owner] = subprocess.Popen(
                            command, pass_fds=(child.fileno(),), stdout=log, stderr=log
                        )
                finally:
                    child.close()
            deadline = time.monotonic() + startup_timeout_seconds
            for owner, peer in self.peers.items():
                receive_ready(peer.connection, owner, identity, deadline)
        except BaseException:
            self.close()
            raise

    def close(self):
        self.closed = True
        for peer in self.peers.values():
            peer.close()
        errors = []
        deadline = time.monotonic() + self.exit_timeout_seconds
        for owner, process in self.processes.items():
            try:
                code = process.wait(timeout=max(0.001, deadline - time.monotonic()))
                if code:
                    errors.append(f"{owner} PID {process.pid} exited {code}")
            except subprocess.TimeoutExpired:
                errors.append(f"{owner} PID {process.pid} has not exited; inspect {self.log_dir}")
        if errors:
            raise RuntimeError("; ".join(errors))
