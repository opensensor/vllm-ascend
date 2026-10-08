# SPDX-License-Identifier: Apache-2.0
"""Two native MoE stages with UB SwiGLU/quantization and weighted down epilogue.

Routing metadata remains device-resident. No FP16 gate/up, hidden activation,
tensor is materialized between the native stages.
Bulk prefill uses weighted FP32 routed rows to keep complete expert batches.
The optional half route workspace stores already-rounded down outputs and
defers the FP32 route multiply to the reducer; it does not reduce precision.
"""

import json
from dataclasses import dataclass

import torch

from .fused_weight_layout import PREROUNDED_SCALE_LAYOUT, pack_cube
from .glm_int4 import MAX_GROUPED_ROUTES, pack_nz_codes, packed_weight_bits

FUSED_REDUCTION_TOKENS = 16
SHARED_PREFILL_SCRATCH_TOKENS = 640
ROUTE_INPUT_ROWS = 32
ROUTE_INPUT_MAX_GROUPS = 128
ROUTE_INPUT_PAIR_BYTES = 2 * ROUTE_INPUT_ROWS * 32


def sorted_token_route_ranks(order, tokens, top_k):
    """Stable expert positions per token, including the unread peer suffix.

    Route positions are bounded by MAX_GROUPED_ROUTES and exactly represented
    in FP32. Float sorting and INT32 narrowing avoid 310P AI-CPU INT64 casts.
    """
    positions = torch.arange(order.numel(), dtype=torch.float32, device=order.device)
    inverse = torch.empty_like(positions).scatter_(0, order, positions)
    return inverse.reshape(tokens, top_k).sort(dim=1).values.to(torch.int32).contiguous()


def fused_route_metadata(weights, ids, experts, expert_offset=0):
    """Only the stable permutation and cumulative ends consumed by our kernels."""
    if ids.ndim != 2 or weights.shape != ids.shape or ids.dtype not in (torch.int32, torch.int64):
        raise ValueError("fused routing requires matching [tokens, top_k] integer ids and weights")
    if weights.device != ids.device or not 1 <= experts <= 288 or expert_offset < 0:
        raise ValueError("invalid fused routing device or expert partition")
    if not 1 <= ids.numel() <= MAX_GROUPED_ROUTES:
        raise ValueError("invalid fused route count")
    local_ids = ids.reshape(-1) - expert_offset
    local = (local_ids >= 0) & (local_ids < experts) & (weights.reshape(-1) != 0)
    # Clamp peer ids before narrowing. Bounded keys use AI-Core conversions;
    # the generic descriptor also computes an unused inverse sort and cumsum.
    keys = torch.where(local, local_ids, experts).to(torch.int32).float()
    order = torch.argsort(keys, stable=True)
    boundaries = torch.arange(1, experts + 1, dtype=torch.float32, device=ids.device)
    # Counts are bounded by MAX_GROUPED_ROUTES. Boolean -> INT64 reduction
    # dispatches an AI-CPU Cast on 310P; INT32 reduction and widening stay on
    # AI-Core while preserving the native kernel's INT64 metadata ABI.
    ends = (keys.unsqueeze(0) < boundaries.unsqueeze(1)).sum(dim=1, dtype=torch.int32).to(torch.int64)
    return order.contiguous(), ends.contiguous()


@dataclass(frozen=True)
class FusedGeometry:
    tokens: int
    top_k: int
    experts: int
    hidden: int
    intermediate: int
    gate_bits: int
    down_bits: int
    activation_bits: int

    def __post_init__(self):
        if any(type(value) is not int for value in self.__dict__.values()):
            raise ValueError("fused geometry requires integers")
        if not 1 <= self.tokens * self.top_k <= MAX_GROUPED_ROUTES or self.tokens <= 0 or self.top_k <= 0:
            raise ValueError("invalid fused route count")
        if not 1 <= self.experts <= 288 or self.hidden not in (256, 512, 2048, 4096):
            raise ValueError("unsupported fused bank geometry")
        if not 256 <= self.intermediate <= 4096 or self.intermediate % 256:
            raise ValueError("fused intermediate requires complete NZ tiles")
        if self.gate_bits not in (2, 3, 4) or self.down_bits not in (2, 3, 4) or self.activation_bits not in (4, 8):
            raise ValueError("unsupported fused precision")


def route_input_shapes(geometry):
    """Capacity from shapes only; valid slots are determined entirely on device."""
    if geometry.tokens <= FUSED_REDUCTION_TOKENS or geometry.activation_bits != 4:
        raise ValueError("routed input packing requires bulk A4 geometry")
    routes = geometry.tokens * geometry.top_k
    active_rows = ROUTE_INPUT_ROWS - 1
    slots = (routes + active_rows - 1) // active_rows + geometry.experts
    return (slots, geometry.hidden // 64, ROUTE_INPUT_PAIR_BYTES), (slots, ROUTE_INPUT_ROWS, ROUTE_INPUT_MAX_GROUPS)


def route_down_shape(geometry):
    """Share routed batch slots; gate/up owns disjoint intermediate groups."""
    packed, _ = route_input_shapes(geometry)
    return packed[0], geometry.intermediate // 64, ROUTE_INPUT_PAIR_BYTES


def route_down_scale_shape(geometry, row_lanes=8):
    """Four independent FP32 scales plus DMA padding per N128 tile row."""
    if row_lanes not in (4, 8):
        raise ValueError("down scale row requires four scalar or eight padded lanes")
    packed, _ = route_input_shapes(geometry)
    return packed[0], geometry.intermediate // 128, ROUTE_INPUT_ROWS, row_lanes


def weight_decode_table():
    """Pair lookup tables: signed fields, low fragments and upper fragments."""
    tables = []
    for table, width in enumerate((2, 3, 2, 1, 1, 2)):

        def nibble(q, table=table, width=width):
            if table in (0, 1):
                return (q - (1 << width) if q >= (1 << (width - 1)) else q) & 15
            if table in (2, 3):
                return q
            if table == 4:
                return 12 * q
            return (2 * (q - 4 if q >= 2 else q)) & 15

        mask = (1 << width) - 1
        tables.extend(nibble((i - 2048) & mask) + 16 * nibble(((i - 2048) >> 8) & mask) for i in range(4096))
    return torch.tensor(tables, dtype=torch.float16)


class NativeFusedMoE:
    input_dtype = torch.float16

    def __init__(
        self,
        root,
        *,
        namespace,
        activation_bits,
        kernel_factory=None,
        launch=None,
        prepared_weight_layout=False,
        weight_decode_lut=False,
        fp16_route_workspace=False,
    ):
        if activation_bits not in (4, 8):
            raise ValueError("activation bits must be 4 or 8")
        if type(fp16_route_workspace) is not bool:
            raise ValueError("FP16 route workspace flag must be boolean")
        options = {}
        provenance = root / "provenance.json"
        if provenance.exists():
            options = json.loads(provenance.read_text())["_build"]
            if options.get("fp16_route_workspace", False) != fp16_route_workspace:
                raise ValueError("route workspace dtype differs from compiled bundle")
        elif fp16_route_workspace:
            raise ValueError("FP16 route workspace requires build provenance")
        if type(options.get("route_packed_input", False)) is not bool:
            raise ValueError("routed input packing flag must be boolean")
        if type(options.get("route_packed_down", False)) is not bool:
            raise ValueError("packed down input flag must be boolean")
        if type(options.get("route_compact_down_scales", False)) is not bool:
            raise ValueError("compact down scales flag must be boolean")
        if type(options.get("raw_hidden_scales", False)) is not bool:
            raise ValueError("raw hidden scales flag must be boolean")
        if type(options.get("raw_input_scales", False)) is not bool:
            raise ValueError("raw input scales flag must be boolean")
        if type(options.get("prerounded_weight_scales", False)) is not bool:
            raise ValueError("prerounded weight scales flag must be boolean")
        if options.get("prerounded_weight_scales") and not prepared_weight_layout:
            raise ValueError("prerounded weight scales require the permanent prepared layout")
        if options.get("raw_input_scales") and not options.get("route_packed_input"):
            raise ValueError("raw input scales require routed input packing")
        if options.get("raw_hidden_scales") and not (
            options.get("route_compact_down_scales") and options.get("quad_hidden_quant")
        ):
            raise ValueError("raw hidden scales require compact down scales and four-row quantization")
        if options.get("route_compact_down_scales") and not options.get("route_packed_down"):
            raise ValueError("compact down scales require packed down input")
        if options.get("route_packed_down") and not options.get("route_packed_input"):
            raise ValueError("packed down input requires routed input packing")
        if options.get("route_packed_input") and not (root / "glm_fused_route_input.bin").exists():
            raise ValueError("routed input packing requires its compiled producer")
        compact = options.get("compact_w4_scratch", False)
        if type(compact) is not bool:
            raise ValueError("compact W4 scratch flag must be boolean")
        if compact and (not prepared_weight_layout or weight_decode_lut or options.get("weight_decode_lut")):
            raise ValueError("compact W4 scratch requires prepared weights without lookup tables")
        gate_w4, down_w4 = root / "glm_fused_gate_up_w4.bin", root / "glm_fused_down_w4.bin"
        if gate_w4.exists() != compact or down_w4.exists() != compact:
            raise ValueError("compact W4 scratch requires declared paired W4 stage binaries")
        self.compact_w4_scratch = compact
        self.activation_bits = activation_bits
        self.prepared_weight_layout = prepared_weight_layout
        self.prerounded_weight_scales = options.get("prerounded_weight_scales", False)
        self.fp16_route_workspace = fp16_route_workspace
        self.launch = launch or getattr(torch.ops, namespace).launch
        factory = kernel_factory or getattr(torch.classes, namespace).Kernel
        gate_w3, down_w3 = root / "glm_fused_gate_up_w3.bin", root / "glm_fused_down_w3.bin"
        if gate_w3.exists() != down_w3.exists():
            raise ValueError("W3 specialization requires both fused stage binaries")
        self.gate_w3_kernel = factory(str(gate_w3), "glm_fused_gate_up_w3_v1") if gate_w3.exists() else None
        self.down_w3_kernel = factory(str(down_w3), "glm_fused_down_w3_v1") if down_w3.exists() else None
        self.gate_w4_kernel = factory(str(gate_w4), "glm_fused_gate_up_w4_v1") if compact else None
        self.down_w4_kernel = factory(str(down_w4), "glm_fused_down_w4_v1") if compact else None
        self.gate_kernel = factory(str(root / "glm_fused_gate_up.bin"), "glm_fused_gate_up_v1")
        self.down_kernel = factory(str(root / "glm_fused_down.bin"), "glm_fused_down_v1")
        self.pack_kernel = factory(str(root / "glm_fused_pack.bin"), "glm_fused_pack_v1")
        self.raw_input_scales = options.get("raw_input_scales", False)
        self.raw_hidden_scales = options.get("raw_hidden_scales", False)
        self.route_compact_down_scales = options.get("route_compact_down_scales", False)
        self.route_packed_down = options.get("route_packed_down", False)
        self.route_packed_input = options.get("route_packed_input", False)
        self.route_input_kernel = (
            factory(str(root / "glm_fused_route_input.bin"), "glm_fused_route_input_v1")
            if self.route_packed_input
            else None
        )
        reduce_entry = "glm_fused_reduce_half_v1" if fp16_route_workspace else "glm_fused_reduce_v1"
        self.reduce_kernel = factory(str(root / "glm_fused_reduce.bin"), reduce_entry)
        self.device = torch.device("npu", torch.npu.current_device())
        self.weight_lookup = weight_decode_table().to(self.device) if weight_decode_lut else None
        self.configs = {}
        self.scratch = {}

    def stage_kernel(self, stage, bits):
        specialized = getattr(self, f"{stage}_w{bits}_kernel", None) if bits in (3, 4) else None
        return specialized if specialized is not None else getattr(self, stage + "_kernel")

    @property
    def required_weight_scale_layout(self):
        # Model dispatch checks this host marker before submitting any kernel.
        # Standalone arithmetic gates prepare their small scale fixtures on CPU.
        return PREROUNDED_SCALE_LAYOUT if getattr(self, "prerounded_weight_scales", False) else None

    @property
    def route_workspace_dtype(self):
        return torch.float16 if getattr(self, "fp16_route_workspace", False) else torch.float32

    def pack_weight_codes(self, signed, bits):
        return pack_cube(signed, bits) if self.prepared_weight_layout else pack_nz_codes(signed, bits)

    def geometry(self, x, gate_codes, gate_scales, down_codes, down_scales, weights, ids):
        tensors = (x, gate_codes, gate_scales, down_codes, down_scales, weights, ids)
        if any(t.device != self.device or not t.is_contiguous() for t in tensors):
            raise ValueError("fused tensors must be contiguous on the prepared NPU")
        if (
            x.ndim != 2
            or gate_codes.ndim != 3
            or down_codes.ndim != 3
            or weights.ndim != 2
            or ids.shape != weights.shape
        ):
            raise ValueError("invalid fused tensor ranks")
        if x.dtype != torch.float16 or weights.dtype != torch.float32 or ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("fused input, routes require FP16/FP32/integer")
        if gate_codes.dtype not in (torch.int8, torch.uint8) or down_codes.dtype not in (torch.int8, torch.uint8):
            raise ValueError("fused banks require byte storage")
        if gate_scales.dtype != torch.float32 or down_scales.dtype != torch.float32:
            raise ValueError("fused bank scales require FP32")
        tokens, hidden = x.shape
        experts, twice_inter, packed_hidden = gate_codes.shape
        inter = twice_inter // 2
        geometry = FusedGeometry(
            tokens,
            weights.shape[1],
            experts,
            hidden,
            inter,
            packed_weight_bits(hidden, packed_hidden),
            packed_weight_bits(inter, down_codes.shape[-1]),
            self.activation_bits,
        )
        if weights.shape[0] != tokens or twice_inter != 2 * inter or down_codes.shape[:2] != (experts, hidden):
            raise ValueError("fused banks and routes disagree")
        if gate_scales.shape != (experts, twice_inter // 32, hidden // 32) or down_scales.shape != (
            experts,
            hidden // 32,
            inter // 32,
        ):
            raise ValueError("fused scale geometry differs")
        return geometry

    def __call__(self, x, gate_codes, gate_scales, down_codes, down_scales, weights, ids, expert_offset=0):
        geometry = self.geometry(x, gate_codes, gate_scales, down_codes, down_scales, weights, ids)
        order, ends = fused_route_metadata(weights, ids, geometry.experts, expert_offset)
        return self.grouped(
            x,
            gate_codes,
            gate_scales,
            down_codes,
            down_scales,
            weights,
            order,
            ends,
            geometry,
        )

    def grouped(self, x, gate_codes, gate_scales, down_codes, down_scales, weights, order, ends, geometry):
        # Metadata depends only on CPU-visible shapes and precision. Preparing
        # a first prefill geometry never reads routes or expert counts on host.
        if geometry not in self.configs:
            common = (geometry.tokens * geometry.top_k, geometry.experts)
            suffix = (geometry.activation_bits, geometry.tokens, geometry.top_k)
            gate = (*common, 2 * geometry.intermediate, geometry.hidden, geometry.gate_bits, *suffix)
            down = (*common, geometry.hidden, geometry.intermediate, geometry.down_bits, *suffix)
            projection_configs = tuple(
                torch.tensor(header, dtype=torch.int64, device=self.device) for header in (gate, down)
            )
            pack_header = (geometry.tokens * geometry.hidden // 32, geometry.activation_bits)
            if getattr(self, "raw_input_scales", False):
                pack_header += (geometry.tokens,)
            pack_config = torch.tensor(pack_header, dtype=torch.int64).to(self.device)
            self.configs[geometry] = (*projection_configs, pack_config)
        gate_config, down_config, pack_config = self.configs[geometry]
        bulk = geometry.tokens > FUSED_REDUCTION_TOKENS
        if geometry.tokens == SHARED_PREFILL_SCRATCH_TOKENS:
            # One serving stream and eager state boundaries separate the
            # qualified 640-token graph segments. Share their large scratch,
            # but retain independent returned outputs.
            scratch_key = (geometry.tokens, geometry.top_k, geometry.hidden, geometry.intermediate)
            if getattr(self, "route_packed_down", False) and geometry.activation_bits == 4:
                scratch_key += (geometry.experts,)
            if scratch_key not in self.scratch:
                self.scratch[scratch_key] = self.allocate_scratch(geometry)
            input_low, input_high, input_scales, low, high, scales, workspace = self.scratch[scratch_key]
            output = torch.empty((geometry.tokens, geometry.hidden), dtype=torch.float32, device=self.device)
            down_output = workspace
        else:
            # Preserve the qualified default's allocation order and local
            # tensor lifetimes. Refactoring these is a full-model experiment.
            input_shape = (geometry.tokens, geometry.hidden // 32, 32)
            input_low = torch.empty(input_shape, dtype=torch.int8, device=self.device)
            input_high = torch.empty(
                input_shape if self.activation_bits == 8 else (1,), dtype=torch.int8, device=self.device
            )
            input_scales = torch.empty(self.input_scale_shape(geometry), dtype=torch.float32, device=self.device)
            routes = geometry.tokens * geometry.top_k
            shape = (routes, geometry.intermediate // 32, 32)
            low = torch.empty(self.hidden_code_shape(geometry), dtype=torch.int8, device=self.device)
            high = torch.empty(shape if self.activation_bits == 8 else (1,), dtype=torch.int8, device=self.device)
            scales = torch.empty(self.hidden_scale_shape(geometry), dtype=torch.float32, device=self.device)
            output = torch.empty((geometry.tokens, geometry.hidden), dtype=torch.float32, device=self.device)
            bulk = geometry.tokens > FUSED_REDUCTION_TOKENS
            down_output = (
                torch.empty((routes, geometry.hidden), dtype=self.route_workspace_dtype, device=self.device)
                if bulk
                else output
            )
        ranks = sorted_token_route_ranks(order, geometry.tokens, geometry.top_k) if bulk else None
        lookup_args = [self.weight_lookup] if self.weight_lookup is not None else []
        self.launch(self.pack_kernel, [x, input_low, input_high, input_scales, pack_config], 8)
        gate_input_high, gate_input_scales = input_high, input_scales
        if getattr(self, "route_input_kernel", None) is not None and bulk and self.activation_bits == 4:
            shapes = route_input_shapes(geometry)
            key = ("route-input", geometry.tokens, geometry.top_k, geometry.hidden, geometry.experts)
            if geometry.tokens == SHARED_PREFILL_SCRATCH_TOKENS:
                if key not in self.scratch:
                    self.scratch[key] = (
                        torch.empty(shapes[0], dtype=torch.int8, device=self.device),
                        torch.empty(shapes[1], dtype=torch.float32, device=self.device),
                    )
                routed_input, routed_scales = self.scratch[key]
            else:
                routed_input = torch.empty(shapes[0], dtype=torch.int8, device=self.device)
                routed_scales = torch.empty(shapes[1], dtype=torch.float32, device=self.device)
            self.launch(
                self.route_input_kernel,
                [input_low, input_scales, order, ends, routed_input, routed_scales, gate_config],
                8,
            )
            gate_input_high, gate_input_scales = routed_input, routed_scales
        self.launch(
            self.stage_kernel("gate", geometry.gate_bits),
            [
                input_low,
                gate_input_high,
                gate_input_scales,
                order,
                gate_codes,
                gate_scales,
                ends,
                low,
                high,
                scales,
                gate_config,
            ]
            + lookup_args,
            8,
        )
        self.launch(
            self.stage_kernel("down", geometry.down_bits),
            [low, high, scales, down_codes, down_scales, ends, order, weights, down_output, down_config] + lookup_args,
            8,
        )
        if bulk:
            reduce_args = [down_output, ranks, ends, output, down_config]
            if getattr(self, "fp16_route_workspace", False):
                reduce_args += [order, weights]
            self.launch(self.reduce_kernel, reduce_args, 8)
        return output

    def input_scale_shape(self, geometry):
        if (
            getattr(self, "raw_input_scales", False)
            and geometry.tokens > FUSED_REDUCTION_TOKENS
            and geometry.activation_bits == 4
        ):
            return geometry.tokens, geometry.hidden // 32
        return geometry.tokens, geometry.hidden // 32, 8

    def hidden_code_shape(self, geometry):
        if (
            getattr(self, "route_packed_down", False)
            and geometry.tokens > FUSED_REDUCTION_TOKENS
            and geometry.activation_bits == 4
        ):
            return route_down_shape(geometry)
        return geometry.tokens * geometry.top_k, geometry.intermediate // 32, 32

    def hidden_scale_shape(self, geometry):
        if (
            getattr(self, "route_compact_down_scales", False)
            and geometry.tokens > FUSED_REDUCTION_TOKENS
            and geometry.activation_bits == 4
        ):
            return route_down_scale_shape(geometry, 4 if getattr(self, "raw_hidden_scales", False) else 8)
        return geometry.tokens * geometry.top_k, geometry.intermediate // 32, 8

    def allocate_scratch(self, geometry):
        input_shape = (geometry.tokens, geometry.hidden // 32, 32)
        input_low = torch.empty(input_shape, dtype=torch.int8, device=self.device)
        input_high = torch.empty(
            input_shape if self.activation_bits == 8 else (1,), dtype=torch.int8, device=self.device
        )
        input_scales = torch.empty(self.input_scale_shape(geometry), dtype=torch.float32, device=self.device)
        routes = geometry.tokens * geometry.top_k
        shape = (routes, geometry.intermediate // 32, 32)
        low = torch.empty(self.hidden_code_shape(geometry), dtype=torch.int8, device=self.device)
        high = torch.empty(shape if self.activation_bits == 8 else (1,), dtype=torch.int8, device=self.device)
        scales = torch.empty(self.hidden_scale_shape(geometry), dtype=torch.float32, device=self.device)
        workspace = (
            torch.empty((routes, geometry.hidden), dtype=self.route_workspace_dtype, device=self.device)
            if geometry.tokens > FUSED_REDUCTION_TOKENS
            else None
        )
        return input_low, input_high, input_scales, low, high, scales, workspace
