# SPDX-License-Identifier: Apache-2.0
"""Prepare CUDA experts, then serve a bounded number of sequential peers."""

import argparse
import json
import os
import socket
from pathlib import Path

from .external_runtime import MAX_CONFIG_BYTES
from .external_worker import worker_from_config
from .native_expert_probe import digest_file
from .runtime import serve_worker
from .tensor_wire import serve_tensor_worker
from .wire import unique_metadata


def run(config_path, socket_path, ready_path, *, max_sessions=1, tensor_transport=False):
    if type(max_sessions) is not int or max_sessions < 1:
        raise ValueError("max_sessions must be a positive integer")
    if config_path.stat().st_size > MAX_CONFIG_BYTES:
        raise ValueError("oversized resident daemon configuration")
    config_sha = digest_file(config_path)
    value = json.loads(config_path.read_bytes(), object_pairs_hook=unique_metadata)
    if value["backend"]["kind"] != "cuda_resident":
        raise ValueError("resident daemon requires the admitted CUDA backend")
    worker, identity = worker_from_config(value)
    if socket_path.exists() or ready_path.exists():
        raise ValueError("resident daemon requires fresh socket and readiness paths")
    try:
        worker.prepare()
        if digest_file(config_path) != config_sha:
            raise ValueError("resident daemon configuration changed during preparation")
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as listener:
            listener.bind(str(socket_path))
            os.chmod(socket_path, 0o600)
            listener.listen(1)
            listener.settimeout(3600)
            receipt = {
                "schema_version": 1,
                "pid": os.getpid(),
                "config_sha256": config_sha,
                "checkpoint_sha256": identity.checkpoint_sha256,
                "placement_sha256": identity.placement_sha256,
                "cuda_allocated_bytes": worker.backend.cuda.allocated_bytes,
                "resident_experts": len(worker.backend.keys),
                "resident_layers": len(worker.backend.plan.by_layer),
                "max_sessions": max_sessions,
                "tensor_transport": tensor_transport,
            }
            temporary = ready_path.with_suffix(".tmp")
            temporary.write_text(json.dumps(receipt, indent=2) + "\n")
            temporary.replace(ready_path)
            for session in range(max_sessions):
                connection, _ = listener.accept()
                with connection:
                    clean = (
                        serve_tensor_worker(connection, worker, identity)
                        if tensor_transport
                        else serve_worker(connection, worker, identity, prepared=True)
                    )
                print(json.dumps({"kind": "peer_closed", "session": session + 1, "clean": clean}), flush=True)
                if not clean:
                    break
    finally:
        worker.backend.close()
        socket_path.unlink(missing_ok=True)
        print(json.dumps({"kind": "backend_service_totals", **worker.backend.metrics}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--socket", type=Path, required=True)
    parser.add_argument("--ready", type=Path, required=True)
    parser.add_argument("--max-sessions", type=int, default=1)
    parser.add_argument("--tensor-transport", action="store_true")
    args = parser.parse_args()
    run(args.config, args.socket, args.ready, max_sessions=args.max_sessions, tensor_transport=args.tensor_transport)


if __name__ == "__main__":
    main()
