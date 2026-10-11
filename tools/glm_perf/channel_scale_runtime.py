# SPDX-License-Identifier: Apache-2.0
"""Complete opt-in channel-scale MoE pipeline; no serving default is changed."""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import regex as re
import torch

from .channel_scale_layout import native_weight_parameters, product_offsets, quantization_offsets
from .channel_scale_math import LAYOUT, SUPPORTED_GROUPS

CORES = 8
LANES = 8
DECODE_ROWS = 16
DEFAULT_SCRATCH_BYTES = 128 * 1024**2
KERNEL_NAMES = ("glm_channel_quantize", "glm_channel_projection", "glm_channel_combine")


def route_metadata(weights, ids, experts, expert_offset, boundaries):
    """Keep the sorted expert keys already produced by Sort for sparse decode."""
    local_ids = ids.reshape(-1) - expert_offset
    local = (local_ids >= 0) & (local_ids < experts) & (weights.reshape(-1) != 0)
    keys = torch.where(local, local_ids, experts).to(torch.int32).float()
    sorted_keys, order = torch.sort(keys, stable=True)
    ends = (keys.unsqueeze(0) < boundaries.unsqueeze(1)).sum(dim=1, dtype=torch.int32)
    return order.contiguous(), ends.contiguous(), sorted_keys.to(torch.int32).contiguous()


@dataclass(frozen=True)
class Geometry:
    experts: int
    hidden: int
    intermediate: int
    gate_group: int
    down_group: int
    fused_gate_up: bool = False
    fused_decode: bool = False
    fused_prefill: bool = False
    token_prefill: bool = False

    def __post_init__(self):
        if (
            any(
                type(v) is not int
                for k, v in self.__dict__.items()
                if k not in ("fused_gate_up", "fused_decode", "fused_prefill", "token_prefill")
            )
            or type(self.fused_gate_up) is not bool
            or type(self.fused_decode) is not bool
            or type(self.fused_prefill) is not bool
            or type(self.token_prefill) is not bool
            or self.token_prefill
            and not self.fused_prefill
        ):
            raise ValueError("channel-scale geometry requires integers")
        if not 1 <= self.experts <= 288 or any(
            k < 256 or k > 4096 or k % 256 for k in (self.hidden, self.intermediate)
        ):
            raise ValueError("unsupported channel-scale bank geometry")
        if any(
            g not in SUPPORTED_GROUPS or k % g
            for k, g in ((self.hidden, self.gate_group), (self.intermediate, self.down_group))
        ):
            raise ValueError("unsupported channel-scale accumulation groups")


def scratch_shapes(tokens, top_k, geometry, bits):
    if type(tokens) is not int or tokens < 1 or type(top_k) is not int or not 1 <= top_k <= LANES or bits not in (4, 8):
        raise ValueError("invalid channel-scale scratch geometry")
    if geometry.token_prefill and top_k != LANES:
        raise ValueError("token prefill requires eight route slots")
    rows, h, i = tokens * top_k, geometry.hidden, geometry.intermediate
    shapes = {
        "input_low": ((tokens, h // 2), torch.int8),
        "input_high": ((tokens, h // 2) if bits == 8 else (1,), torch.int8),
        "input_scales": ((tokens, h // geometry.gate_group, LANES), torch.float32),
        "gate_up": ((rows, 2 * i), torch.float16),
        "hidden_low": ((rows, i // 2), torch.int8),
        "hidden_high": ((rows, i // 2) if bits == 8 else (1,), torch.int8),
        "hidden_scales": ((rows, i // geometry.down_group, LANES), torch.float32),
        "routed": ((rows, h), torch.float16),
        "weight_lanes": ((tokens, LANES), torch.float32),
    }
    if top_k == LANES:
        del shapes["weight_lanes"]
    if geometry.fused_gate_up:
        del shapes["gate_up"]
    if geometry.fused_prefill and tokens > DECODE_ROWS:
        del shapes["routed"]
        if geometry.token_prefill:
            windows = (tokens + DECODE_ROWS - 1) // DECODE_ROWS
            shapes["token_rows"] = ((windows, DECODE_ROWS * LANES), torch.float32)
            shapes["token_sorted_rows"] = ((windows, DECODE_ROWS * LANES), torch.int32)
            shapes["token_positions"] = ((windows, DECODE_ROWS * LANES), torch.int64)
            shapes["route_tiles"] = ((1,), torch.float16)
        else:
            shapes["route_tiles"] = ((min(CORES, h // 128), rows, 128), torch.float16)
    if geometry.fused_decode and tokens <= DECODE_ROWS:
        del shapes["routed"]
    return shapes


def shape_bytes(shapes):
    total = 0
    for shape, dtype in shapes.values():
        count = 1
        for value in shape:
            count *= value
        total += count * torch.empty((), dtype=dtype).element_size()
    return total


class ScratchArena:
    """Explicit per-stage owner, separate captured decode slots and prefill growth.

    Callers share this object across sequential MoE layers. Returned FP32 model
    outputs have independent storage. Scratch has a declared byte budget, never
    reads device utilization, and never allocates the maximum context window.
    """

    def __init__(self, budget_bytes=DEFAULT_SCRATCH_BYTES):
        if type(budget_bytes) is not int or budget_bytes <= 0:
            raise ValueError("scratch budget must be positive")
        self.budget_bytes = budget_bytes
        self.slots = {}

    @property
    def allocated_bytes(self):
        return sum(
            shape_bytes(scratch_shapes(capacity, top_k, geometry, bits))
            for (_, top_k, geometry, bits, _), (capacity, _) in self.slots.items()
        )

    def get(self, tokens, top_k, geometry, bits, device):
        key = (tokens if tokens <= DECODE_ROWS else "prefill", top_k, geometry, bits, str(device))
        old_capacity, old = self.slots.get(key, (0, None))
        if old_capacity < tokens:
            shapes = scratch_shapes(tokens, top_k, geometry, bits)
            old_bytes = 0 if old is None else shape_bytes(scratch_shapes(old_capacity, top_k, geometry, bits))
            if self.allocated_bytes - old_bytes + shape_bytes(shapes) > self.budget_bytes:
                raise ValueError("channel-scale scratch exceeds the admitted per-stage budget")
            buffers = {name: torch.empty(shape, dtype=dtype, device=device) for name, (shape, dtype) in shapes.items()}
            self.slots[key] = tokens, buffers
            old = buffers
        shapes = scratch_shapes(tokens, top_k, geometry, bits)
        return {
            name: old[name].reshape(-1)[: torch.Size(shape).numel()].view(shape) for name, (shape, _) in shapes.items()
        }


def verify_bundle(root):
    root = Path(root)
    manifest = json.loads((root / "provenance.json").read_text())
    if manifest.get("abi_version") not in (3, 4):
        raise ValueError("requires channel-scale ABI 3 or token-window ABI 4")
    if manifest.get("complete") is not True or manifest.get("layout") != LAYOUT:
        raise ValueError("requires a complete new channel-scale kernel bundle")
    namespace = manifest.get("namespace")
    if not isinstance(namespace, str) or re.fullmatch(r"glm_channel_scale_v[1-9][0-9]*", namespace) is None:
        raise ValueError("channel-scale bundle requires its own operator namespace")
    if "cube_rows" in manifest:
        rows = manifest["cube_rows"]
        if (
            type(rows) is not int
            or rows not in (32, 64)
            or f"-DGLM_CHANNEL_ROWS={rows}" not in manifest.get("compiler_options", [])
        ):
            raise ValueError("declared Cube row geometry differs from compiler options")
    identity = manifest.get("identity_smoothing", False)
    if type(identity) is not bool or identity != (
        "-DGLM_CHANNEL_IDENTITY_SMOOTHING" in manifest.get("compiler_options", [])
    ):
        raise ValueError("identity smoothing marker differs from compiled contract")
    reciprocal = manifest.get("reciprocal_quant", False)
    if (
        type(reciprocal) is not bool
        or reciprocal != ("-DGLM_CHANNEL_RECIPROCAL_QUANT" in manifest.get("compiler_options", []))
        or (reciprocal and not identity)
    ):
        raise ValueError("reciprocal quantization marker differs from qualified compiled contract")
    integer = manifest.get("integer_reconstruct", False)
    if type(integer) is not bool or integer != (
        "-DGLM_CHANNEL_INTEGER_RECONSTRUCT" in manifest.get("compiler_options", [])
    ):
        raise ValueError("integer reconstruction marker differs from compiled contract")
    cached_rows = manifest.get("cache_source_rows", False)
    if type(cached_rows) is not bool or cached_rows != (
        "-DGLM_CHANNEL_CACHE_SOURCE_ROWS" in manifest.get("compiler_options", [])
    ):
        raise ValueError("source row cache marker differs from compiled contract")
    stream = manifest.get("stream_weights", False)
    if type(stream) is not bool or stream != ("-DGLM_CHANNEL_STREAM_WEIGHTS" in manifest.get("compiler_options", [])):
        raise ValueError("streamed weight marker differs from compiled contract")
    cached = manifest.get("cache_prefill_routes", False)
    if (
        type(cached) is not bool
        or cached != ("-DGLM_CHANNEL_CACHE_PREFILL_ROUTES" in manifest.get("compiler_options", []))
        or cached
        and (not manifest.get("fused_prefill") or manifest.get("token_prefill"))
    ):
        raise ValueError("prefill route cache marker differs from compiled contract")
    token = manifest.get("token_prefill", False)
    if token != (manifest.get("abi_version") == 4):
        raise ValueError("token prefill requires its distinct routing-metadata ABI 4")
    mapped = manifest.get("kernel_sources", {}).get("glm_channel_down_prefill")
    if not token and mapped not in (None, "glm_channel_down_prefill.cpp"):
        raise ValueError("bulk prefill cannot use the token-window kernel source")
    if (
        type(token) is not bool
        or token
        and (
            not manifest.get("fused_prefill")
            or manifest.get("kernel_sources", {}).get("glm_channel_down_prefill") != "glm_channel_down_token.cpp"
        )
    ):
        raise ValueError("token prefill marker lacks its compiled source")
    kernels = (*KERNEL_NAMES, "glm_channel_gate_up") if manifest.get("fused_gate_up") else KERNEL_NAMES
    if manifest.get("fused_decode"):
        kernels = (*kernels, "glm_channel_down_decode")
    if manifest.get("fused_prefill"):
        kernels = (*kernels, "glm_channel_down_prefill")
    names = (*[name + ".bin" for name in kernels], "bridge.so")
    for name in names:
        value = root / name
        if hashlib.sha256(value.read_bytes()).hexdigest() != manifest.get("files", {}).get(name):
            raise ValueError("channel-scale binary checksum differs: " + name)
    return manifest


class NativeChannelScaleMoE:
    """Input quant -> gate/up -> fused SwiGLU/quant -> down -> stable combine.

    This changes the numerical contract and therefore remains opt-in. CPU bank
    admission and all device metadata preparation happen during construction.
    Forward accepts already typed contiguous tensors, reads no device scalar,
    uploads no CPU tensor, and allocates only the incoming scratch geometry.
    """

    def __init__(
        self,
        root,
        gate_up,
        down,
        *,
        device,
        arena,
        expert_offset=0,
        top_k=8,
        max_tokens=640,
        decode_bits=8,
        prefill_bits=8,
        kernel_factory=None,
        launch=None,
    ):
        gate_up.validate()
        down.validate()
        experts, twice_i, hidden = gate_up.shape
        intermediate = twice_i // 2
        if down.shape != (experts, hidden, intermediate) or expert_offset < 0:
            raise ValueError("channel-scale gate/up and down partitions disagree")
        if not torch.equal(gate_up.smoothing, gate_up.smoothing[:1].expand_as(gate_up.smoothing)):
            raise ValueError("gate/up input smoothing must be shared across local experts")
        if (
            type(top_k) is not int
            or not 1 <= top_k <= LANES
            or type(max_tokens) is not int
            or not 1 <= max_tokens <= 8192
        ):
            raise ValueError("invalid channel-scale prepared request capacity")
        if decode_bits not in (4, 8) or prefill_bits not in (4, 8) or not isinstance(arena, ScratchArena):
            raise ValueError("explicit scratch arena and A4/A8 modes are required")
        manifest = verify_bundle(root)
        self.identity_smoothing = manifest.get("identity_smoothing", False)
        if self.identity_smoothing and any(
            not torch.equal(b.smoothing, torch.ones_like(b.smoothing)) for b in (gate_up, down)
        ):
            raise ValueError("identity kernel requires exactly unit stored diagonals")
        self.device = torch.device(device)
        self.fused_prefill = manifest.get("fused_prefill", False)
        self.token_prefill = manifest.get("token_prefill", False)
        if self.token_prefill and top_k != LANES:
            raise ValueError("token prefill currently requires eight route slots")
        self.fused_decode = manifest.get("fused_decode", False)
        self.fused_gate_up = manifest.get("fused_gate_up", False)
        if self.fused_gate_up and down.group_size != 256:
            raise ValueError("UB gate/up fusion requires G256 hidden quantization")
        self.geometry = Geometry(
            experts,
            hidden,
            intermediate,
            gate_up.group_size,
            down.group_size,
            self.fused_gate_up,
            self.fused_decode,
            self.fused_prefill,
            self.token_prefill,
        )
        self.offset, self.top_k, self.max_tokens = expert_offset, top_k, max_tokens
        self.decode_bits, self.prefill_bits, self.arena = decode_bits, prefill_bits, arena
        if kernel_factory is None:
            torch.ops.load_library(str(Path(root) / "bridge.so"))
        factory = kernel_factory or getattr(torch.classes, manifest["namespace"]).Kernel
        self.launch = launch or getattr(torch.ops, manifest["namespace"]).launch
        names = (*KERNEL_NAMES, "glm_channel_gate_up") if self.fused_gate_up else KERNEL_NAMES
        if self.fused_decode:
            names = (*names, "glm_channel_down_decode")
        if self.fused_prefill:
            names = (*names, "glm_channel_down_prefill")
        self.kernels = {name: factory(str(Path(root) / (name + ".bin")), name + "_v1") for name in names}
        # Private storage also prevents a future DMA upload from touching mmap pages.
        dummy = torch.empty(1, dtype=torch.float32, device=self.device) if self.identity_smoothing else None

        def upload(bank):
            parameters = native_weight_parameters(bank)
            values = tuple(value.clone().to(self.device) for value in parameters[:3])
            return (*values, dummy if self.identity_smoothing else parameters[3].clone().to(self.device))

        self.gate, self.down = upload(gate_up), upload(down)
        self.input_smoothing = self.gate[3] if self.identity_smoothing else self.gate[3][0]
        self.product_offsets = product_offsets(manifest.get("cube_rows", 32)).to(self.device)
        self.quantization_offsets = quantization_offsets().to(self.device)
        headers = []
        for tokens in range(1, max_tokens + 1):
            bits = decode_bits if tokens <= DECODE_ROWS else prefill_bits
            rows = tokens * top_k
            headers.append(
                [
                    (tokens, hidden, gate_up.group_size, bits, 0, top_k, experts, expert_offset),
                    (tokens, top_k, experts, 2 * intermediate, hidden, gate_up.group_size, bits, 1),
                    (rows, intermediate, down.group_size, bits, 1, top_k, experts, expert_offset),
                    (tokens, top_k, experts, hidden, intermediate, down.group_size, bits, 0),
                ]
            )
        self.headers = torch.tensor(headers, dtype=torch.int64).to(self.device)
        self.boundaries = torch.arange(1, experts + 1, dtype=torch.float32).to(self.device)
        self.positions = torch.arange(max_tokens * top_k, dtype=torch.float32).to(self.device)

    def __call__(self, x, weights, ids):
        if x.ndim != 2:
            raise ValueError("channel-scale input must be a token matrix")
        tokens = x.shape[0]
        if not 1 <= tokens <= self.max_tokens or x.shape != (tokens, self.geometry.hidden):
            raise ValueError("request exceeds prepared channel-scale geometry")
        if weights.shape != (tokens, self.top_k) or ids.shape != weights.shape:
            raise ValueError("channel-scale route geometry differs")
        if x.dtype != torch.float16 or weights.dtype != torch.float32 or ids.dtype != torch.int64:
            raise ValueError("channel-scale forward requires FP16 inputs, FP32 weights, INT64 ids")
        if any(t.device != self.device or not t.is_contiguous() for t in (x, weights, ids)):
            raise ValueError("channel-scale inputs must be contiguous on the prepared device")
        order, ends, sorted_keys = route_metadata(weights, ids, self.geometry.experts, self.offset, self.boundaries)
        if not self.fused_decode or tokens > DECODE_ROWS:
            inverse = torch.empty_like(self.positions[: ids.numel()]).scatter_(0, order, self.positions[: ids.numel()])
            # Preserve original top-k order. The old sorted-rank helper reorders slots.
            if not (self.token_prefill and tokens > DECODE_ROWS):
                inverse = inverse.to(torch.int32).contiguous()
        bits = self.decode_bits if tokens <= DECODE_ROWS else self.prefill_bits
        scratch = self.arena.get(tokens, self.top_k, self.geometry, bits, self.device)
        token_order, token_positions = order, None
        if self.token_prefill and tokens > DECODE_ROWS:
            padded = torch.nn.functional.pad(
                inverse, (0, scratch["token_rows"].numel() - inverse.numel()), value=tokens * self.top_k
            )
            torch.sort(
                padded.reshape_as(scratch["token_rows"]),
                dim=1,
                stable=True,
                out=(scratch["token_rows"], scratch["token_positions"]),
            )
            scratch["token_sorted_rows"].copy_(scratch["token_rows"])
            token_order, token_positions = scratch["token_sorted_rows"], scratch["token_positions"]
        headers = self.headers[tokens - 1]
        quant = self.kernels["glm_channel_quantize"]
        projection = self.kernels["glm_channel_projection"]
        self.launch(
            quant,
            [
                x,
                self.input_smoothing,
                ends,
                order,
                ids,
                scratch["input_low"],
                scratch["input_high"],
                scratch["input_scales"],
                self.quantization_offsets,
                headers[0],
            ],
            CORES,
        )
        if self.fused_gate_up:
            self.launch(
                self.kernels["glm_channel_gate_up"],
                [
                    scratch["input_low"],
                    scratch["input_high"],
                    scratch["input_scales"],
                    *self.gate[:3],
                    ends,
                    order,
                    sorted_keys,
                    self.product_offsets,
                    self.down[3],
                    scratch["hidden_low"],
                    scratch["hidden_high"],
                    scratch["hidden_scales"],
                    self.quantization_offsets,
                    headers[1],
                ],
                CORES,
            )
        else:
            self.launch(
                projection,
                [
                    scratch["input_low"],
                    scratch["input_high"],
                    scratch["input_scales"],
                    *self.gate[:3],
                    ends,
                    order,
                    sorted_keys,
                    self.product_offsets,
                    scratch["gate_up"],
                    headers[1],
                ],
                CORES,
            )
            self.launch(
                quant,
                [
                    scratch["gate_up"],
                    self.down[3],
                    ends,
                    order,
                    ids,
                    scratch["hidden_low"],
                    scratch["hidden_high"],
                    scratch["hidden_scales"],
                    self.quantization_offsets,
                    headers[2],
                ],
                CORES,
            )
        if self.top_k == LANES:
            weight_lanes = weights
        else:
            weight_lanes = scratch["weight_lanes"]
            weight_lanes.zero_()
            weight_lanes[:, : self.top_k].copy_(weights)
        output = torch.empty((tokens, self.geometry.hidden), dtype=torch.float32, device=self.device)
        if self.fused_decode and tokens <= DECODE_ROWS:
            self.launch(
                self.kernels["glm_channel_down_decode"],
                [
                    scratch["hidden_low"],
                    scratch["hidden_high"],
                    scratch["hidden_scales"],
                    *self.down[:3],
                    ends,
                    order,
                    sorted_keys,
                    self.product_offsets,
                    weight_lanes,
                    output,
                    headers[3],
                ],
                CORES,
            )
        elif self.fused_prefill and tokens > DECODE_ROWS:
            self.launch(
                self.kernels["glm_channel_down_prefill"],
                [
                    scratch["hidden_low"],
                    scratch["hidden_high"],
                    scratch["hidden_scales"],
                    *self.down[:3],
                    ends,
                    token_order,
                    sorted_keys,
                    self.product_offsets,
                    scratch["route_tiles"],
                    token_positions if self.token_prefill else inverse,
                    weight_lanes,
                    output,
                    headers[3],
                ],
                CORES,
            )
        else:
            self.launch(
                projection,
                [
                    scratch["hidden_low"],
                    scratch["hidden_high"],
                    scratch["hidden_scales"],
                    *self.down[:3],
                    ends,
                    order,
                    sorted_keys,
                    self.product_offsets,
                    scratch["routed"],
                    headers[3],
                ],
                CORES,
            )
            self.launch(
                self.kernels["glm_channel_combine"],
                [scratch["routed"], inverse, ends, weight_lanes, output, headers[3]],
                CORES,
            )
        return output


class ChannelScaleMoEMethod:
    """Use the existing GLM composed-MoE seam without changing model files."""

    def __init__(self, native):
        self.native = native

    def apply(self, layer, x, topk_weights, topk_ids, shared_experts, shared_experts_input):
        if layer.w2_experts.local_expert_offset != self.native.offset:
            raise ValueError("prepared channel-scale partition differs from live MoE")
        return self.native(x, topk_weights, topk_ids)
