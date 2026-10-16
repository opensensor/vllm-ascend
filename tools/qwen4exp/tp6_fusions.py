# SPDX-License-Identifier: Apache-2.0
"""Private eager TP6 operators. Importing never loads a library or patches a model."""

import json
from pathlib import Path

import torch

from tools.qwen4exp.native_prefill import NativeResource, _capturing, _tensors_on_one_device
from tools.qwen4exp.tp6_fusion_sources import digest

MAX_ROUTES = 20480
TOP_K = 10
WIDTH = 2560
MODES = ("ordered", "sequential", "pairwise")
NAMESPACE = "qwen_hc_residual_producer_v1"


def verify_build(directory):
    directory = Path(directory).resolve(strict=True)
    receipt = json.loads((directory / "provenance.json").read_text())
    if receipt.get("schema_version") != 1:
        raise ValueError("unsupported TP6 fusion manifest")
    required = {"native_tp6_gate_pack.bin", "native_tp6_finalize.bin"}
    if set(receipt["binaries"]) != required:
        raise ValueError("both compiled fusion binaries are required")
    for section in ("sources", "binaries"):
        for name, checksum in receipt[section].items():
            path = directory / name
            if Path(name).name != name or path.is_symlink() or digest(path.read_bytes()) != checksum:
                raise ValueError("fusion artifact identity changed")
    return receipt


class EagerResource(NativeResource):
    """No graph admission; failed submissions retain arguments and poison reuse."""

    def __init__(self, kernel, launch):
        super().__init__(kernel, launch)
        self.failed = False
        self.retained = ()
        self.failure = None

    def preflight(self, device):
        if self.failed:
            raise RuntimeError("failed fusion resource cannot be reused")
        if torch.is_grad_enabled() or _capturing(device):
            raise RuntimeError("TP6 fusion candidates require eager inference")

    def submit(self, args):
        self.retained = tuple(args)
        try:
            self.launch(self.kernel, args, 8)
        except BaseException as error:
            self.failed, self.failure = True, error
            raise
        self.retained = ()


class GatePack(EagerResource):
    def __call__(self, bank, prepared, ends):
        if len(prepared) != 4 or bank.weight.ndim != 3:
            raise ValueError("require four packed operands and a TP6 gate bank")
        low, high, scale, total = prepared
        device = low.device
        self.preflight(device)
        experts = bank.weight.shape[0]
        routed = ends.dtype == torch.int32
        rows = ends.numel() if routed else low.shape[0]
        source_rows = low.shape[0]
        broadcast = TOP_K if routed and rows == source_rows * TOP_K else 1
        metadata = (experts, 1280, 20)
        if (
            experts not in (85, 86)
            or bank.weight.shape != (experts, 1280, 1280)
            or not 0 < rows <= MAX_ROUTES
            or (routed and rows > 128)
            or rows != source_rows * broadcast
            or low.shape != (source_rows, 1280)
            or high.shape != low.shape
            or any(t.dtype != torch.int8 for t in (low, high, bank.weight))
            or scale.shape != (source_rows, 20, 8)
            or total.shape != scale.shape
            or any(t.dtype != torch.float32 for t in (scale, total))
            or any(
                t.shape != metadata or t.dtype != torch.float16
                for t in (bank.weight_scale, bank.weight_offset, bank.weight_sum)
            )
            or ends.shape != (rows if routed else experts,)
            or ends.dtype != (torch.int32 if routed else torch.int64)
        ):
            raise ValueError("unsupported TP6 gate/pack geometry or dtype")
        operands = [*prepared, bank.weight, bank.weight_scale, bank.weight_offset, bank.weight_sum, ends]
        _tensors_on_one_device(operands, device)
        output = (
            low.new_empty((rows, 320)),
            high.new_empty((rows, 320)),
            scale.new_empty((rows, 5, 8)),
            total.new_empty((rows, 5, 8)),
        )
        config = self.config((rows, experts, 1280, 2560, 8, int(routed), broadcast), device)
        self.submit([*operands, *output, config])
        return output


class Finalize(EagerResource):
    def __init__(self, kernel, launch, mode="ordered"):
        if mode not in MODES:
            raise ValueError("unknown reduction order")
        super().__init__(kernel, launch)
        self.mode = mode

    def __call__(self, routed, inverse, weights, ends):
        self.preflight(routed.device)
        rows = routed.shape[0]
        if (
            not 0 < rows <= MAX_ROUTES
            or rows % TOP_K
            or routed.shape != (rows, WIDTH)
            or routed.dtype != torch.float16
            or inverse.shape != (rows,)
            or inverse.dtype != torch.int64
            or weights.shape != (rows, 1)
            or weights.dtype != torch.float32
            or ends.shape not in ((85,), (86,))
            or ends.dtype != torch.int64
        ):
            raise ValueError("unsupported TP6 finalization geometry or dtype")
        operands = [routed, inverse, weights, ends]
        _tensors_on_one_device(operands, routed.device)
        count = rows if self.mode == "ordered" else rows // TOP_K
        result = torch.empty((count, WIDTH), dtype=torch.float32, device=routed.device)
        config = self.config((rows, ends.numel()), routed.device)
        self.submit([*operands, result, config])
        # Preserve the retained backend's reduction unless explicitly testing
        # a different tree. Never infer bitwise equivalence from algebra.
        return result.reshape(-1, TOP_K, WIDTH).sum(1) if self.mode == "ordered" else result


def load_resources(directory, bridge, bridge_sha256, *, owner):
    """Explicit NPU operation. Caller must retain owner, including on failure."""
    if owner:
        raise ValueError("provide an empty persistent owner")
    directory, bridge = Path(directory).resolve(), Path(bridge).resolve(strict=True)
    receipt = verify_build(directory)
    if digest(bridge.read_bytes()) != bridge_sha256:
        raise ValueError("producer bridge identity mismatch")
    owner["manifest"] = receipt
    owner["kernels"] = []
    owner["resources"] = {}
    torch.ops.load_library(str(bridge))
    launch = getattr(torch.ops, NAMESPACE).launch
    for name, entry in receipt["entries"].items():
        if name not in ("gate_pack", *MODES):
            raise ValueError("unknown fusion entry")
        binary = "native_tp6_gate_pack.bin" if name == "gate_pack" else "native_tp6_finalize.bin"
        kernel = getattr(torch.classes, NAMESPACE).Kernel(str(directory / binary), entry)
        owner["kernels"].append(kernel)
        owner["resources"][name] = GatePack(kernel, launch) if name == "gate_pack" else Finalize(kernel, launch, name)
    return owner["resources"]
