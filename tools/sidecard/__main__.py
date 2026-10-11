# SPDX-License-Identifier: Apache-2.0
"""Generate offline plans or collect a read-only inventory on the joined host."""

import argparse
import hashlib
import json
import shlex
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from .bundle import DEFAULT_SHARD_BYTES, PartitionStore, bundle_identity, export_checkpoint
from .checkpoint import plan_checkpoint
from .partition_smoke import run_partition_smoke
from .placement import DEFAULT_CUDA_EXPERTS, ExpertPlacement, glm_placement, storage_lower_bound, validate_glm_config
from .smoke import run_smoke

INVENTORY_COMMANDS = (
    ("host", ("hostname",)),
    ("kernel", ("uname", "-a")),
    ("ascend", ("npu-smi", "info")),
    ("cuda", ("nvidia-smi", "--query-gpu=name,memory.total,pci.bus_id,driver_version", "--format=csv")),
    ("pcie_tree", ("lspci", "-tv")),
    ("memory", ("free", "-b")),
)


def inventory(ssh_host: str | None = None):
    receipts = {}
    for name, command in INVENTORY_COMMANDS:
        if ssh_host:
            command = (
                "ssh",
                "-F",
                "/dev/null",
                "-o",
                "BatchMode=yes",
                "-o",
                "ConnectTimeout=5",
                "--",
                ssh_host,
                shlex.join(command),
            )
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=20, check=False)
            receipts[name] = {
                "command": command,
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
            }
        except (OSError, subprocess.TimeoutExpired) as error:
            receipts[name] = {"command": command, "error": str(error)}
    return {
        "recorded_at": datetime.now(UTC).isoformat(),
        "hardware_validated": False,
        "inventory_only": True,
        "receipts": receipts,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    plan_parser = subparsers.add_parser("plan")
    plan_parser.add_argument("--model-config", type=Path, required=True)
    plan_parser.add_argument("--cuda-experts", type=int, default=DEFAULT_CUDA_EXPERTS)
    plan_parser.add_argument("--expert-bits", type=int, choices=(2, 3, 4), default=4)
    inventory_parser = subparsers.add_parser("inventory")
    inventory_parser.add_argument("--ssh", help="Collect on this SSH host using batch authentication")
    subparsers.add_parser("smoke", help="Run isolated CPU workers with synthetic resident experts")
    subparsers.add_parser("partition-smoke", help="Export, verify and execute small packed CPU partitions")
    checkpoint_parser = subparsers.add_parser(
        "checkpoint-plan", help="Audit canonical tensor headers and plan ownership"
    )
    checkpoint_parser.add_argument("--source", type=Path, required=True)
    checkpoint_parser.add_argument("--placement", type=Path, required=True)
    checkpoint_parser.add_argument("--include-tensor-slices", action="store_true")
    export_parser = subparsers.add_parser("export-canonical", help="Copy unchanged owner tensors into a new bundle")
    export_parser.add_argument("--source", type=Path, required=True)
    export_parser.add_argument("--placement", type=Path, required=True)
    export_parser.add_argument("--output", type=Path, required=True)
    export_parser.add_argument("--target-shard-bytes", type=int, default=DEFAULT_SHARD_BYTES)
    verify_parser = subparsers.add_parser("verify-partition", help="Verify a published canonical owner partition")
    verify_parser.add_argument("--root", type=Path, required=True)
    verify_parser.add_argument("--placement", type=Path, required=True)
    verify_parser.add_argument("--owner", required=True)
    verify_parser.add_argument("--expected-checkpoint-sha256")
    args = parser.parse_args()
    if args.command == "inventory":
        result = inventory(args.ssh)
    elif args.command == "smoke":
        result = run_smoke()
    elif args.command == "partition-smoke":
        result = run_partition_smoke()
    elif args.command in ("export-canonical", "verify-partition"):
        value = json.loads(args.placement.read_text())
        placement = ExpertPlacement.from_dict(value.get("placement", value))
        if args.command == "export-canonical":
            manifest = export_checkpoint(args.source, placement, args.output, args.target_shard_bytes)
            identity = bundle_identity(args.output, placement)
            result = {
                "output": str(args.output),
                "checkpoint_sha256": identity.checkpoint_sha256,
                "native_layout_exported": False,
                "hardware_validated": False,
                "capacity_admitted": False,
                "owner_tensor_bytes": {
                    owner: info["source_tensor_bytes"] for owner, info in manifest["owners"].items()
                },
            }
        else:
            identity = bundle_identity(args.root, placement)
            if args.expected_checkpoint_sha256 and identity.checkpoint_sha256 != args.expected_checkpoint_sha256:
                raise ValueError("bundle manifest differs from pinned checkpoint identity")
            store = PartitionStore(args.root, args.owner, placement, identity)
            result = {
                "owner": args.owner,
                "tensor_count": len(store.records),
                "source_tensor_bytes": store.source_tensor_bytes,
                "checkpoint_sha256": identity.checkpoint_sha256,
                "manifest_pinned": args.expected_checkpoint_sha256 is not None,
                "shard_checksums_verified": True,
                "hardware_validated": False,
                "capacity_admitted": False,
            }
    elif args.command == "checkpoint-plan":
        value = json.loads(args.placement.read_text())
        placement = ExpertPlacement.from_dict(value.get("placement", value))
        result = plan_checkpoint(args.source, placement, args.include_tensor_slices)
    else:
        config_bytes = args.model_config.read_bytes()
        validate_glm_config(json.loads(config_bytes))
        placement = glm_placement(args.cuda_experts)
        result = {
            "status": "offline_candidate",
            "hardware_validated": False,
            "model_config_sha256": hashlib.sha256(config_bytes).hexdigest(),
            "model_config_path": str(args.model_config),
            "mtp_enabled": False,
            "transport": "host_staged_candidate",
            "placement": placement.to_dict(),
            "logical_expert_code_bytes_lower_bound": storage_lower_bound(placement, args.expert_bits),
            "expert_bits_for_estimate": args.expert_bits,
            "capacity_admitted": False,
            "excluded_from_estimate": [
                "scales",
                "padding",
                "dense_weights",
                "KV_and_KDA_state",
                "workspaces",
                "graphs",
                "driver_reserve",
            ],
        }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
