# SPDX-License-Identifier: Apache-2.0
"""Offline byte and ownership proof for the Qwen G128 streaming ABI.

This module models command ownership, not device timing. Independent storage
permits overlap; the CPU trace cannot prove that hardware actually overlaps it.
No device runtime is imported and no unknown whole-rank budget is accepted.
"""

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

ABI_VERSION = 1
M = 16
N = 128
GROUP = 128
K0 = 64
BLOCK = 16
LANES = 8
MAX_K = 2560
MAX_GROUPS = MAX_K // GROUP
SLOTS = 2
BLOCKS = 8
CONTROL_EVENT = 2
ALIGNMENT = 32
QUANTIZER_BYTES = 32768
EVENTS_PER_DIRECTION = 8
CAPACITIES = (("UB", 253952), ("L1", 1048576), ("L0A", 65536), ("L0B", 65536), ("L0C", 262144))
EVENT_DIRECTIONS = (
    "S_MTE2",
    "MTE2_S",
    "V_MTE2",
    "MTE2_V",
    "MTE2_MTE1",
    "MTE1_MTE2",
    "MTE2_MTE3",
    "MTE3_MTE1",
    "MTE1_MTE3",
    "MTE3_MTE2",
    "MTE1_M",
    "M_MTE1",
    "M_V",
    "V_M",
    "V_MTE3",
    "MTE3_V",
)
RANK_COMPONENTS = (
    "checkpoint_resident",
    "old_native_resources",
    "new_native_resources",
    "kv_cache",
    "mamba_primary",
    "mamba_archive",
    "activation_workspace",
    "route_workspace",
    "graphs",
    "hccl",
    "shadow_comparison",
    "verification_scratch",
    "runtime_reserve",
)
SLOTTED_REGIONS = (
    "packed_activation",
    "raw_product",
    "float_product",
    "metadata_stage",
    "activation_metadata",
    "activation_l1",
    "activation_l0",
    "weight_l0",
)


@dataclass(frozen=True)
class Region:
    name: str
    space: str
    offset: int
    nbytes: int
    dtype: str
    layout: str
    lifetime: str

    @property
    def end(self):
        return self.offset + self.nbytes


def regions():
    """Allocate all simultaneous scratch disjointly, with no phase aliases."""
    specs = (
        (
            "packed_activation",
            "UB",
            SLOTS * 2 * M * GROUP // 2,
            "int4_packed",
            "[slot,limb,GROUP/K0,M,K0/2]",
            "load through L0A staging",
        ),
        ("raw_product", "UB", SLOTS * 2 * M * N * 4, "int32", "[slot,N/16,2*M,16]", "CO1 readback through vector cast"),
        (
            "float_product",
            "UB",
            SLOTS * 2 * M * N * 4,
            "float32",
            "[slot,N/16,2*M,16]",
            "vector correction through accumulation",
        ),
        ("accumulator", "UB", M * N * 4, "float32", "[N/16,M,16]", "all G128 additions through final FP16 cast"),
        (
            "weight_metadata",
            "UB",
            3 * N * MAX_GROUPS * 2,
            "float16",
            "bank stride=N*MAX_GROUPS; active [N/16,groups,16] within each bank",
            "expert/output tile across every M tile",
        ),
        ("metadata_stage", "UB", SLOTS * 3 * N * 4, "float32", "[slot,bank,N]", "group staging through correction"),
        (
            "activation_metadata",
            "UB",
            SLOTS * 2 * M * LANES * 4,
            "float32",
            "[slot,scale_or_sum,M,8]",
            "load through correction",
        ),
        ("projected_output", "UB", M * N * 2, "float16", "[N/16,M,16]", "final cast through GM store completion"),
        ("persistent_gate", "UB", M * N * 2, "float16", "[N/16,M,16]", "gate complete through paired up epilogue"),
        (
            "quantizer_scratch",
            "UB",
            QUANTIZER_BYTES,
            "byte",
            "exclusive epilogue arena",
            "entire nonlinear/requantize call",
        ),
        ("ends", "UB", 32 * 8, "int64", "[32]", "route boundary iteration"),
        ("route_ids", "UB", 128 * 4, "int32", "[128]", "sparse route iteration"),
        (
            "activation_l1",
            "L1",
            SLOTS * 2 * M * GROUP // 2,
            "int4_packed",
            "[slot,limb,GROUP/K0,M,K0/2]",
            "MTE2 write through MTE1 read completion",
        ),
        (
            "resident_weight",
            "L1",
            N * MAX_K // 2,
            "int4_packed",
            "[N/16,MAX_GROUPS,G128/64,16,32]",
            "expert/output tile across every M tile",
        ),
        (
            "activation_l0",
            "L0A",
            SLOTS * 2 * M * GROUP // 2,
            "int4_packed",
            "[slot,limb,M/16,GROUP/K0,16,K0/2]",
            "MTE1 write through Cube consumption",
        ),
        (
            "weight_l0",
            "L0B",
            SLOTS * N * GROUP // 2,
            "int4_packed",
            "[slot,G128/64,N/16,16,32]",
            "MTE1 write through Cube consumption",
        ),
        ("cube_product", "L0C", 2 * M * N * 4, "int32", "[N/16,2*M,16]", "Cube issue through completed UB readback"),
    )
    allocated = []
    ends = {}
    for name, space, nbytes, dtype, layout, lifetime in specs:
        offset = (ends.get(space, 0) + ALIGNMENT - 1) // ALIGNMENT * ALIGNMENT
        allocated.append(Region(name, space, offset, nbytes, dtype, layout, lifetime))
        ends[space] = offset + nbytes
    return tuple(allocated)


def validate_regions(allocated, capacities=CAPACITIES):
    limits = dict(capacities)
    names = set()
    for index, region in enumerate(allocated):
        if region.name in names or region.space not in limits:
            raise ValueError("duplicate name or unknown memory space")
        names.add(region.name)
        if region.offset < 0 or region.nbytes <= 0 or region.offset % ALIGNMENT or region.nbytes % ALIGNMENT:
            raise ValueError("invalid byte size or alignment")
        if region.end > limits[region.space]:
            raise ValueError(f"{region.space} overcommit: {region.end} > {limits[region.space]}")
        for other in allocated[:index]:
            if region.space == other.space and region.offset < other.end and other.offset < region.end:
                raise ValueError(f"simultaneously live alias: {region.name}/{other.name}")
    return {space: max((r.end for r in allocated if r.space == space), default=0) for space in limits}


class Ownership:
    """Exclusive region ownership and counted, directional hardware events.

    Signals are outstanding until consumed once. Release can require an event
    token and never means merely that a command was submitted. Tokens carry a
    generation, so waiting on a previous group's completion is rejected.
    """

    def __init__(self, allocated=()):
        self.regions = {r.name: r for r in allocated}
        self.live = {}
        self.signals = {}
        self.completed = {}
        self.log = []

    def acquire(self, name, owner):
        if name not in self.regions:
            raise ValueError("unknown region")
        region = self.regions[name]
        if name in self.live:
            raise ValueError("premature buffer reuse")
        for live_name in self.live:
            other = self.regions[live_name]
            if region.space == other.space and region.offset < other.end and other.offset < region.end:
                raise ValueError("live alias")
        self.live[name] = (owner, len(self.log))
        self.log.append(("acquire", name, owner))

    def signal(self, direction, event_id, generation):
        key = (direction, event_id)
        if direction not in EVENT_DIRECTIONS or not 0 <= event_id < EVENTS_PER_DIRECTION:
            raise ValueError("invalid hardware event")
        if key in self.signals:
            raise ValueError("event reused before wait")
        if (direction, event_id, generation) in self.completed:
            raise ValueError("event generation reused")
        self.signals[key] = generation
        self.log.append(("signal", direction, event_id, generation))

    def wait(self, direction, event_id, generation):
        key = (direction, event_id)
        if key not in self.signals or self.signals[key] != generation:
            raise ValueError("unmatched or stale event")
        del self.signals[key]
        token = (direction, event_id, generation)
        self.completed[token] = len(self.log)
        self.log.append(("wait", *token))
        return token

    def release(self, name, owner, completion):
        if name not in self.live or self.live[name][0] != owner:
            raise ValueError("release by non-owner")
        if (
            completion not in self.completed
            or completion[2] != owner
            or self.completed[completion] <= self.live[name][1]
        ):
            raise ValueError("premature release without matching completion")
        del self.live[name]
        self.log.append(("release", name, owner))

    def drain(self):
        if self.live or self.signals:
            raise ValueError("unreleased storage or unmatched event at drain")
        return tuple(self.log)


def prove_schedule(groups, live_rows=M, mode="bulk"):
    """Simulate pipeline ownership for startup, alternating groups and drain.

    Cube j and consumption j-1 own disjoint regions at the same instant. One
    CO1 slot is used: V_M acknowledges its completed readback before Cube j+1.
    Metadata and activation remain group-owned until correction, not just MMAD.
    Sparse rows use the same paired layout with a separately bounded row loop.
    """
    if not 1 <= groups <= MAX_GROUPS or not 1 <= live_rows <= M or mode not in ("bulk", "sparse"):
        raise ValueError("unsupported projection geometry")
    if mode == "bulk" and live_rows != M:
        raise ValueError("partial rows require sparse correction")
    allocated = []
    for region in regions():
        if region.name in SLOTTED_REGIONS:
            for slot in range(SLOTS):
                allocated.append(
                    Region(
                        f"{region.name}_{slot}",
                        region.space,
                        region.offset + slot * region.nbytes // SLOTS,
                        region.nbytes // SLOTS,
                        region.dtype,
                        region.layout,
                        region.lifetime,
                    )
                )
        else:
            allocated.append(region)
    validate_regions(tuple(allocated))
    proof = Ownership(allocated)
    permanent = ("resident_weight", "weight_metadata", "accumulator")
    for name in permanent:
        proof.acquire(name, "tile")
    previous = None
    additions = []
    overlaps = []

    def complete(direction, slot, group):
        proof.signal(direction, slot, group)
        return proof.wait(direction, slot, group)

    complete("V_MTE2", CONTROL_EVENT, "tile")
    complete("MTE2_MTE1", CONTROL_EVENT, "tile")
    complete("MTE2_V", CONTROL_EVENT, "tile")

    def consume(group):
        slot = group % SLOTS
        proof.acquire(f"float_product_{slot}", group)
        # PIPE_V fences casts and each correction dependency; group order is
        # independent of producer slot order and is never reordered.
        additions.append(group)
        done = complete("V_MTE2", slot, group)
        for name in ("raw_product", "float_product", "metadata_stage", "activation_metadata"):
            proof.release(f"{name}_{slot}", group, done)

    for group in range(groups):
        slot = group % SLOTS
        for name in (
            "packed_activation",
            "activation_l1",
            "activation_l0",
            "weight_l0",
            "metadata_stage",
            "activation_metadata",
        ):
            proof.acquire(f"{name}_{slot}", group)
        # Existing UB -> L1 operand pack uses the MTE3 path. The directional
        # fences below prove the writer/reader lifetime, not just readiness.
        complete("MTE2_V", slot, group)
        complete("V_MTE3", slot, group)
        complete("MTE3_MTE1", slot, group)
        loaded = complete("MTE3_MTE2", slot, group)
        proof.release(f"packed_activation_{slot}", group, loaded)
        complete("MTE1_M", slot, group)
        staged = complete("MTE1_MTE3", slot, group)
        proof.release(f"activation_l1_{slot}", group, staged)
        proof.acquire("cube_product", group)
        if previous is not None:
            # Cube owns CO1/current L0 while vector owns prior UB, proving
            # permitted storage concurrency rather than claimed task timing.
            overlaps.append((group, previous))
            consume(previous)
        complete("M_V", slot, group)
        cube_done = complete("M_MTE1", slot, group)
        proof.release(f"activation_l0_{slot}", group, cube_done)
        proof.release(f"weight_l0_{slot}", group, cube_done)
        proof.acquire(f"raw_product_{slot}", group)
        readback_done = complete("V_M", slot, group)
        proof.release("cube_product", group, readback_done)
        previous = group
    consume(previous)
    output = "store"
    proof.acquire("projected_output", output)
    complete("V_MTE3", CONTROL_EVENT, output)
    stored = complete("MTE3_V", CONTROL_EVENT, output)
    proof.release("projected_output", output, stored)
    done = complete("MTE1_MTE2", CONTROL_EVENT, "tile")
    for name in permanent:
        proof.release(name, "tile", done)
    log = proof.drain()
    if additions != list(range(groups)):
        raise ValueError("G128 accumulation order changed")
    return {
        "groups": groups,
        "live_rows": live_rows,
        "mode": mode,
        "additions": additions,
        "permitted_cube_consumer_pairs": overlaps,
        "trace": log,
        "hardware_overlap_proven": False,
    }


def rank_envelope(available_bytes, components, guard_bytes=0):
    """Require measured/declared exclusive components before any allocation.

    Include simultaneously resident old/new resources and shadow scratch. Zero
    is allowed only as an explicit value, never as the default for an unknown.
    CPU spill bytes belong to a separate host budget and cannot increase NPU
    capacity. This is a per-rank bound; callers must validate all four ranks.
    """
    if set(components) != set(RANK_COMPONENTS):
        raise ValueError("whole-rank component set is incomplete or unknown")
    values = (available_bytes, guard_bytes, *components.values())
    if any(type(value) is not int or value < 0 for value in values) or available_bytes == 0:
        raise ValueError("unknown or invalid whole-rank byte budget")
    total = sum(components.values()) + guard_bytes
    if total > available_bytes:
        raise ValueError("whole-rank memory overcommit")
    return {
        "total_bytes": total,
        "available_bytes": available_bytes,
        "headroom_bytes": available_bytes - total,
        "guard_bytes": guard_bytes,
        "components": dict(components),
    }


def route_workspace(tokens, top_k, hidden=2560, intermediate=1280):
    """Conservative GM envelope retaining the FP16 gate/up boundary.

    Stable route IDs/weights/row map/ends remain on device. Quantized low/high
    limbs and scale/sum banks coexist across gate/up and down. This is only the
    routed MoE slice, not model weights, attention, caches, HCCL, or graphs.
    """
    if type(tokens) is not int or type(top_k) is not int or not 1 <= tokens <= 2560 or not 1 <= top_k <= 10:
        raise ValueError("invalid route workspace geometry")
    if (hidden, intermediate) != (2560, 1280):
        raise ValueError("unsupported production widths")
    rows = tokens * top_k
    parts = {
        "route_ids_and_weights": rows * 8,
        "stable_route_row_map": rows * 4,
        "routing_sort_scratch": rows * 16,
        "expert_ends": 128 * 8,
        "routed_input_fp16": rows * hidden * 2,
        "input_packed_limbs": rows * hidden,
        "input_scale_sum": rows * (hidden // GROUP) * LANES * 2 * 4,
        "projected_gate_up_fp16": rows * intermediate * 2,
        "swiglu_hidden_fp16": rows * 640 * 2,
        "hidden_packed_limbs": rows * 640,
        "hidden_scale_sum": rows * (640 // GROUP) * LANES * 2 * 4,
        "routed_output_fp16": rows * hidden * 2,
        "combined_output_fp32": tokens * hidden * 4,
    }
    return {
        "tokens": tokens,
        "top_k": top_k,
        "rows": rows,
        "components": parts,
        "total_bytes": sum(parts.values()),
        "measured_bus_bytes": False,
    }


def route_regions(tokens, top_k):
    """Disjoint GM arena bound; all intermediates conservatively coexist.

    It retains the gathered input and nonlinear output even if a future fusion
    removes them. Framework allocator overhead belongs in runtime_reserve.
    External inputs, shared expert and attention belong in activation_workspace.
    """
    allocated = []
    offset = 0
    for name, nbytes in route_workspace(tokens, top_k)["components"].items():
        size = (nbytes + ALIGNMENT - 1) // ALIGNMENT * ALIGNMENT
        allocated.append(
            Region(
                name, "GM", offset, size, "byte", "see route_workspace logical tensor sizes", "entire routed MoE call"
            )
        )
        offset += size
    validate_regions(tuple(allocated), (("GM", offset),))
    return tuple(allocated)


def contract():
    """Seal the complete byte, layout, event and admission contract together."""
    allocated = regions()
    usage = validate_regions(allocated)
    sdk_path = Path(__file__).resolve().parents[2] / "artifacts/qwen38-streaming-upgrade/T2/sdk-provenance.json"
    sdk = json.loads(sdk_path.read_text())
    for filename, expected_hash in sdk["files"].items():
        if hashlib.sha256((sdk_path.parent / filename).read_bytes()).hexdigest() != expected_hash:
            raise ValueError("installed SDK snapshot hash mismatch")
    sdk_limits = sdk["capacities"]
    if sdk["target"] != "dav-2002" or sdk["events_per_direction"] != EVENTS_PER_DIRECTION:
        raise ValueError("unsupported SDK provenance")
    for space, limit in CAPACITIES:
        if sdk_limits[space] - (sdk_limits["UB_reserved"] if space == "UB" else 0) != limit:
            raise ValueError("SDK capacity contract mismatch")
    result = {
        "abi_version": ABI_VERSION,
        "kind": "qwen_g128_streaming_ownership_contract",
        "target": "dav-2002",
        "sdk_provenance_sha256": hashlib.sha256(sdk_path.read_bytes()).hexdigest(),
        "constants": {
            "M": M,
            "N": N,
            "GROUP": GROUP,
            "K0": K0,
            "BLOCK": BLOCK,
            "LANES": LANES,
            "MAX_K": MAX_K,
            "MAX_GROUPS": MAX_GROUPS,
            "SLOTS": SLOTS,
            "BLOCKS": BLOCKS,
            "CONTROL_EVENT": CONTROL_EVENT,
            "ALIGNMENT": ALIGNMENT,
            "QUANTIZER_BYTES": QUANTIZER_BYTES,
        },
        "capacities_bytes": dict(CAPACITIES),
        "ub_sdk_reserve_bytes": 8192,
        "usage_bytes": usage,
        "regions": [asdict(region) for region in allocated],
        "event_ids": {
            direction: [0, 1, CONTROL_EVENT]
            if direction in ("V_MTE2", "MTE2_MTE1", "MTE1_MTE2", "MTE2_V", "V_MTE3", "MTE3_V")
            else [0, 1]
            for direction in EVENT_DIRECTIONS
        },
        "event_rules": {
            "generation": "group integer; distinct tile/store strings for non-group lifetime",
            "signal_wait": "one outstanding signal per direction/id, consumed exactly once",
            "co1_release": "V_M after complete UB readback, before the next Cube issue",
            "slot_release": "V_MTE2 after correction/addition, never merely after load submission",
            "operand_release": "M_V after Cube consumes both activation limbs and weight",
            "store_release": "MTE3_V after projected FP16 output store",
            "control_event": (
                "ID 2 for expert-cache and output-store lifetimes; all slot tokens are drained before reuse"
            ),
            "pipe_v": "cast and correction dependencies require explicit vector ordering",
        },
        "projection": {
            "gate_up": {"N": 1280, "K": 2560},
            "down": {"N": 2560, "K": 640},
            "blocks": 8,
            "output_tile_assignment": "tile = block_id + k*8; no duplicated writes",
            "column_windows": "optional first_tile/count<=8; full bank strides retained; output uses window stride",
            "product": "paired low/high INT4 MMAD: m=32,n=128,k=128; CO1/UB [N/16,2*M,16]",
            "correction": (
                "(((low+16*high)-offset*sum)+8*weight_sum)*weight_scale*activation_scale; add groups 0..K/128-1 in FP32"
            ),
            "projection_boundary": "FP16 CAST_NONE after all G128 groups",
            "bulk": "16 live rows; strip or row loop respecting paired layout",
            "sparse": "1..15 live rows; row loop only over live rows",
            "peer_rows": (
                "every peer FP16 row explicitly zeroed before completed launch is consumed; no stale graph output"
            ),
            "metadata": (
                "three FP16 banks cached once per expert/output tile; bank base=N*MAX_GROUPS; "
                "strip stride=actual groups*16; cast cached FP16 directly to per-slot FP32"
            ),
            "decode": "existing bounded decode remains separate; no automatic replacement",
        },
        "epilogue": {
            "paired_gate_owner": (
                "same core owns gate N128 columns and matching up N128 columns; reserve FP16 gate while computing up"
            ),
            "qualified_reference": (
                "cann_builtin_fp16 requires surviving projected GM boundary until nonlinear parity is proven"
            ),
            "fp32_candidate": (
                "separate explicit ABI; unqualified against builtin FP16; no implicit activation precision substitution"
            ),
            "handoff": (
                "packed low/high hidden plus FP32 scale/sum written to GM; "
                "down depends on multiple column owners; same-core shortcut unsupported"
            ),
            "persistent_scratch_bytes": QUANTIZER_BYTES + M * N * 2,
        },
        "whole_rank": {
            "required_components": list(RANK_COMPONENTS),
            "available_bytes": None,
            "values": {name: None for name in RANK_COMPONENTS},
            "qualification": "unknown; must reject before hardware allocation",
            "host_spill_capacity": "separate host budget; never credit against NPU bytes",
        },
        "rejected_alternatives": [
            "all G128 products simultaneously: 20*2*16*128*4=327680 bytes exceeds usable UB before scratch",
            "double resident gate/up K2560 weights with M32/N320 or all-rank inferred free memory",
            "reuse CO1 while UB readback is pending",
            "reuse result/activation/metadata slot before prior correction",
            "pair only K32 arithmetic borrowed from GLM: incompatible G128 correction contract",
            "same-core down consumes incomplete hidden columns",
            "silently replace builtin FP16 SwiGLU with FP32 fusion",
        ],
        "qualification": (
            "offline byte/ownership proof only; compiler, device correctness, overlap, "
            "image, service and thermal gates pending"
        ),
        "hardware_validated": False,
    }
    encoded = json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    result["contract_sha256"] = hashlib.sha256(encoded).hexdigest()
    return result


def header_text():
    value = contract()
    lines = [
        "// SPDX-License-Identifier: Apache-2.0",
        "// Generated from streaming_memory.py; byte contract only, not hardware qualification.",
        "#ifndef QWEN_STREAMING_CONTRACT_H",
        "#define QWEN_STREAMING_CONTRACT_H",
        "#include <stdint.h>",
        "#if defined(__NPU_ARCH__)",
        "  #define QWEN_DEVICE __aicore__",
        "#else",
        "  #define QWEN_DEVICE",
        "#endif",
        "namespace qwen_streaming {",
        f'constexpr const char* CONTRACT_SHA256 = "{value["contract_sha256"]}";',
        f"constexpr uint32_t ABI_VERSION = {ABI_VERSION};",
    ]
    for name, number in value["constants"].items():
        lines.append(f"constexpr uint32_t {name} = {number};")
    for region in regions():
        prefix = region.name.upper()
        lines.extend(
            (
                f"constexpr uint32_t {prefix}_OFFSET = {region.offset};",
                f"constexpr uint32_t {prefix}_BYTES = {region.nbytes};",
            )
        )
        if region.name in SLOTTED_REGIONS:
            lines.append(f"constexpr uint32_t {prefix}_SLOT_BYTES = {region.nbytes // SLOTS};")
    for space, used in value["usage_bytes"].items():
        lines.extend(
            (
                f"constexpr uint32_t {space}_USED = {used};",
                f"constexpr uint32_t {space}_LIMIT = {dict(CAPACITIES)[space]};",
                f'static_assert({space}_USED <= {space}_LIMIT, "{space} streaming overcommit");',
            )
        )
    lines.extend(
        (
            "// Each direction independently uses slot ID 0/1; never share a live generation.",
            "QWEN_DEVICE constexpr uint32_t EventId(uint32_t slot) { return slot; }",
            "QWEN_DEVICE constexpr uint32_t Slot(uint32_t group) { return group % SLOTS; }",
            "QWEN_DEVICE constexpr uint32_t ProductIndex(uint32_t strip, uint32_t limb, "
            "uint32_t row, uint32_t column) {",
            "  return strip * (2 * M * BLOCK) + limb * (M * BLOCK) + row * BLOCK + column;",
            "}",
            "QWEN_DEVICE constexpr uint32_t AccumulatorIndex(uint32_t strip, uint32_t row, uint32_t column) {",
            "  return strip * (M * BLOCK) + row * BLOCK + column;",
            "}",
            "// Kernel helper contract (member methods; no host/device launch ABI implied):",
            "// Produce(uint32_t slot, int64_t row, uint32_t group, uint32_t live_rows)",
            "// IssueCube(uint32_t slot): both packed limbs, m=2*M; CO1 must be free.",
            "// ReadBack(uint32_t slot): M_V -> DataCopy -> V_M, releases single CO1.",
            "// Consume(uint32_t slot, uint32_t live_rows): stable G128 correction/add.",
            "// Store(int64_t row, uint32_t live_rows): FP16 projection boundary + MTE3_V.",
            "}  // namespace qwen_streaming",
            "#undef QWEN_DEVICE",
            "#endif",
            "",
        )
    )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--header", type=Path)
    args = parser.parse_args()
    args.output.write_text(json.dumps(contract(), indent=2) + "\n")
    if args.header:
        args.header.write_text(header_text())


if __name__ == "__main__":
    main()
