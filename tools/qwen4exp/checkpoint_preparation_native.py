# SPDX-License-Identifier: Apache-2.0
"""Owned fused pre-copy kernel with the same admitted bank geometry as commit."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from tools.qwen4exp.mamba_commit import PADDED_ROWS, group_geometry


def load(build):
    root = Path(build).resolve(strict=True)
    manifest = json.loads((root / "provenance.json").read_text())
    namespace = "qwen_checkpoint_preparation_v1"
    if manifest["namespace"] != namespace or manifest["flush_host_queue"] is not False:
        raise ValueError("checkpoint native bridge contract differs")
    for name, digest in manifest["sources"].items():
        if Path(name).name != name or hashlib.sha256((root / name).read_bytes()).hexdigest() != digest:
            raise ValueError("checkpoint source provenance differs")
    for item in [manifest["bridge"], *manifest["binaries"].values()]:
        path = Path(item["path"]).resolve(strict=True)
        if path.parent != root or hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
            raise ValueError("checkpoint binary provenance differs")
    torch.ops.load_library(manifest["bridge"]["path"])
    kernel = getattr(torch.classes, namespace).Kernel(
        manifest["binaries"]["native_checkpoint_preparation"]["path"], namespace
    )
    return kernel, getattr(torch.ops, namespace).launch


class CheckpointPrecopyGroup:
    def __init__(self, kernel, launch, banks, tables, convolution, *, block_size):
        configs = group_geometry(banks, tables, convolution, block_size=block_size)
        device = banks[0].device
        if device.type != "npu" or torch.npu.is_current_stream_capturing() or not 1 <= configs[0][0] <= 4:
            raise ValueError("construct admitted checkpoint group before capture")
        self.kernel, self.launch = kernel, launch
        self.banks, self.tables = tuple(banks), tuple(tables)
        self.rows = configs[0][0]
        self.descriptors = torch.tensor(
            [[bank.data_ptr(), table.data_ptr(), *cfg] for bank, table, cfg in zip(banks, tables, configs)],
            dtype=torch.int64,
            device=device,
        )
        self.config = torch.tensor([len(banks)], dtype=torch.int64, device=device)
        self.receipt = torch.empty((len(banks), 2, PADDED_ROWS), dtype=torch.int32, device=device)

    def __call__(self, previous, computed, accepted):
        if any(
            tensor.shape != (self.rows,)
            or tensor.dtype != torch.int32
            or tensor.device != self.receipt.device
            or not tensor.is_contiguous()
            for tensor in (previous, computed, accepted)
        ):
            raise ValueError("pre-copy inputs must be admitted contiguous INT32 request vectors")
        self.launch(self.kernel, [previous, computed, accepted, self.descriptors, self.receipt, self.config], 8)
        return self.receipt


@torch.inference_mode()
def install_fused_graphs(owner, kernel, launch):
    """Cold-only additional captures; keep the qualified tensor graphs for A/B."""
    if hasattr(owner, "fused_records") or owner.failed:
        raise RuntimeError("fused checkpoint graph ownership already exists or failed")
    owner.materialize_counts()
    torch.npu.synchronize()
    owner.failed = True
    owner.tensor_records = owner.records
    owner.fused_records = {}
    owner.backend = "tensor"
    owner.backend_checks = {"tensor": (owner.checked, owner.checked_blocks), "fused": (0, 0)}
    commit = owner.commit
    stream = torch.npu.Stream(device=owner.runner.device)
    stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(stream):
        saved_normalized = owner.normalized.clone()
        for n in range(1, 5):
            values = owner.tensor_records[n]["values"]
            saved = tuple(value.clone() for value in values)
            group = CheckpointPrecopyGroup(
                kernel,
                launch,
                commit.banks,
                [owner.tables[key][:n] for key in commit.bank_groups],
                commit.convolution,
                block_size=commit.block_size,
            )
            record = dict(
                graph=torch.npu.NPUGraph(),
                group=group,
                values=values,
                plan=SimpleNamespace(error=group.receipt[0, 1, :n]),
            )
            owner.fused_records[n] = record

            def prepare(group=group, values=values, n=n):
                group(*values)
                owner.normalized.fill_(1)
                owner.normalized[:n].copy_(group.receipt[0, 0, :n])

            values[0].zero_()
            values[1].fill_(10)
            values[2].fill_(1)
            for _ in range(3):
                prepare()
            torch.npu.synchronize()
            with torch.npu.graph(record["graph"], stream=stream):
                prepare()
            torch.npu.synchronize()
            receipt = group.receipt.cpu()
            if receipt[:, 1].any() or not torch.all(receipt[:, 0] == 1):
                raise RuntimeError("native checkpoint no-copy warmup differs")
            for value, old in zip(values, saved):
                value.copy_(old)
        owner.normalized.copy_(saved_normalized)
    torch.npu.current_stream().wait_stream(stream)
    torch.npu.synchronize()
    owner.failed = False


def select_backend(owner, backend, *, verify):
    """Only drained controls may change graph owners or verification mode."""
    if backend not in ("tensor", "fused") or owner.failed:
        raise ValueError("unknown or failed checkpoint backend")
    checked, blocks = (
        owner.backend_checks[backend] if backend != owner.backend else (owner.checked, owner.checked_blocks)
    )
    if not verify and (not checked or not blocks):
        raise RuntimeError("live decision and state-bank comparisons required")
    owner.materialize_counts()
    owner.backend_checks[owner.backend] = (owner.checked, owner.checked_blocks)
    owner.checked, owner.checked_blocks = checked, blocks
    owner.records = owner.tensor_records if backend == "tensor" else owner.fused_records
    owner.backend = backend
    owner.enabled = True
    owner.verify = verify
