# SPDX-License-Identifier: Apache-2.0
"""Trusted local JSON startup for separate-environment expert workers."""

import argparse
import json
import socket
from pathlib import Path

from .canonical_cpu import CanonicalCPUBackend
from .external_runtime import MAX_CONFIG_BYTES
from .placement import ExpertPlacement
from .runtime import serve_worker
from .wire import PacketLimits, WireIdentity, unique_metadata
from .worker import ExpertWorker


def worker_from_config(value):
    if not isinstance(value, dict) or set(value) != {"owner", "placement", "limits", "identity", "backend"}:
        raise ValueError("invalid external worker configuration")
    placement = ExpertPlacement.from_dict(value["placement"])
    identity = WireIdentity(**value["identity"])
    limits = PacketLimits(**value["limits"])
    owner = value["owner"]
    if owner not in placement.owners or identity != WireIdentity.for_placement(placement, identity.checkpoint_sha256):
        raise ValueError("external worker placement identity mismatch")
    backend = dict(value["backend"])
    kind = backend.pop("kind")
    if kind == "canonical_cpu":
        if set(backend) != {"root", "hidden_width"}:
            raise ValueError("invalid canonical CPU worker configuration")
        prepared = CanonicalCPUBackend(Path(backend["root"]), owner, placement, identity, backend["hidden_width"])
    elif kind in ("ascend_native_gate", "cuda_oracle_gate"):
        # Runtime imports occur only inside the selected worker process.
        from .device_backend import NativeGateBackend

        prepared = NativeGateBackend(kind, backend, owner, placement, identity, limits)
    elif kind == "cuda_resident":
        from .resident import CudaResidentBackend

        prepared = CudaResidentBackend(backend, owner, placement, identity, limits)
    else:
        raise ValueError("unknown external worker backend")
    return ExpertWorker(owner, placement, limits, prepared), identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--socket-fd", type=int, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    if args.config.stat().st_size > MAX_CONFIG_BYTES:
        raise ValueError("oversized external worker configuration")
    value = json.loads(args.config.read_bytes(), object_pairs_hook=unique_metadata)
    worker, identity = worker_from_config(value)
    with socket.socket(fileno=args.socket_fd) as connection:
        try:
            serve_worker(connection, worker, identity)
        finally:
            close = getattr(worker.backend, "close", None)
            if close is not None:
                close()
            metrics = getattr(worker.backend, "metrics", None)
            if metrics is not None:
                print(json.dumps({"kind": "backend_service_totals", **metrics}), flush=True)


if __name__ == "__main__":
    main()
