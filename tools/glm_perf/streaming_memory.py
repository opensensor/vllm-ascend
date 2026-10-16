# SPDX-License-Identifier: Apache-2.0
"""CPU proof of the bounded two-slot projection schedule; never opens an NPU."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import regex as re

ROWS = 32
COLUMNS = 128
GROUP = 32
MAX_K = 4096
SLOTS = 2
SDK_RESERVE = 8192
ALIGNMENT = 32
CONTRACT_VERSION = 2
PRODUCT_SLOT_BYTES = 32768
CONVERTED_BYTES = 16384
SDK_CAPACITIES = (262144, 1048576, 65536, 65536, 262144)
PHASES = ("reconstruction", "projection", "nonlinear", "quantization", "handoff")
EVENTS = (
    ("MTE2_MTE1", "activation copy complete", "operand load"),
    ("MTE1_MTE2", "previous L1 operand reads complete", "L1 activation staging reuse"),
    ("MTE1_M", "L0 operands ready", "Cube issue"),
    ("M_MTE1", "Cube operand reads complete", "L0 operand reuse"),
    ("M_V", "Cube result ready", "CO1 readback"),
    ("V_M", "CO1 readback complete", "L0C reuse"),
)


def header_version(header: Path) -> int:
    """Validate fixed C++ constants without compiling or evaluating arbitrary code."""
    text = re.sub(r"/\*.*?\*/|//[^\n]*", "", header.read_text(), flags=re.DOTALL)
    constants = {}

    def integer(node):
        if isinstance(node, ast.Constant) and type(node.value) is int:
            return node.value
        if isinstance(node, ast.Name) and node.id in constants:
            return constants[node.id]
        if isinstance(node, ast.BinOp):
            left, right = integer(node.left), integer(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div) and right > 0:
                return left // right
        raise ValueError("unsupported fixed contract expression")

    for declaration in re.findall(r"constexpr\s+uint32_t\s+([^;]+);", text):
        for assignment in declaration.split(","):
            name, expression = assignment.split("=", 1)
            name = name.strip()
            if name in constants:
                raise ValueError("duplicate contract constant: " + name)
            expression = re.sub(
                r"sizeof\((int32_t|uint32_t|float|int64_t)\)",
                lambda match: "8" if match[1] == "int64_t" else "4",
                expression.strip(),
            )
            expression = re.sub(r"(?<=\d)[uUlL]+\b", "", expression)
            try:
                constants[name] = integer(ast.parse(expression, mode="eval").body)
            except (SyntaxError, ValueError) as error:
                raise ValueError("invalid contract constant: " + name) from error
    version = constants.get("VERSION")
    if version not in (1, 2):
        raise ValueError("unsupported streaming contract version in memory contract")
    expected = {
        "VERSION": version,
        "ROWS": ROWS,
        "MAX_ROWS": 31,
        "COLUMNS": COLUMNS,
        "GROUP": GROUP,
        "PAIR_K": 64,
        "WEIGHT_TILE_K": 256,
        "MAX_K": MAX_K,
        "SLOTS": SLOTS,
        "EVENT_FIRST": 0,
        "ELEMENTS": ROWS * COLUMNS,
        "OPERAND_A_SLOT_BYTES": 2048,
        "OPERAND_B_SLOT_BYTES": 4096,
        "PRODUCT_SLOT_BYTES": PRODUCT_SLOT_BYTES,
        "PRODUCT_STORAGE_BYTES": 65536,
        "CONVERTED_BYTES": CONVERTED_BYTES,
        "FACTORS_OFFSET": 0,
        "BROADCAST_OFFSET": 16384,
        "RETAINED_INDEX_OFFSET": 24576,
        "EPILOGUE_TILE_BYTES": 4096,
        "SDK_UB_RESERVE": SDK_RESERVE,
    }
    if version == 2:
        expected.update(
            MASK_BYTES=32768,
            RETAINED_EXPERT_END_OFFSET=24704,
            RETAINED_EXPERT_END_BYTES=2304,
            SCALE_GATHER_INDEX_OFFSET=27008,
            SCALE_GATHER_INDEX_BYTES=1024,
            GROUP_SCALE_SOURCE_OFFSET=28032,
            GROUP_SCALE_SOURCE_BYTES=128,
        )
    if constants != expected:
        raise ValueError("header constants do not match the frozen streaming memory contract")
    return version


def aligned(value: int) -> int:
    return (value + ALIGNMENT - 1) // ALIGNMENT * ALIGNMENT


def projection_allocations(*, gate: bool, compact_w4: bool) -> dict[str, int]:
    """Match every v1021 InitBuffer, plus one separately owned conversion plane."""
    buffers = {
        "raw": 2112 if compact_w4 else 16448,
        "decoded": 65536,
        "packed_b": 16384,
        "gathered": 8192 if compact_w4 else 32768,
        "mask": 32768,
        "weight_scales": 3584,
        "activation_scales": (ROWS + 8 + 1) * (MAX_K // GROUP) * 4,
        "output_offsets": COLUMNS * 4,
        "product_offsets": 2 * COLUMNS * 4,
        "output_row": COLUMNS * 2,
        "packed_a": 2048,
        "accumulator": ROWS * COLUMNS * 4,
        "output": ROWS * COLUMNS * 2,
        "stream_converted": CONVERTED_BYTES,
    }
    buffers["quant_offsets" if gate else "reduction"] = 4160 if gate else 16384
    return buffers


def allocation_table(buffers: dict[str, int]) -> list[dict]:
    offset = 0
    result = []
    for name, size in buffers.items():
        if type(size) is not int or size <= 0:
            raise ValueError("allocation sizes must be positive integers")
        result.append({"name": name, "offset": offset, "bytes": size, "aligned_bytes": aligned(size)})
        offset += aligned(size)
    return result


def validate_aliases(aliases: list[dict], buffers: dict[str, int]) -> None:
    """Reject overlapping live intervals; phases are ordered and explicit."""
    for value in aliases:
        if value["buffer"] not in buffers or not set(value["phases"]) <= set(PHASES):
            raise ValueError("unknown buffer or lifetime phase")
        if value["offset"] < 0 or value["bytes"] <= 0 or value["offset"] + value["bytes"] > buffers[value["buffer"]]:
            raise ValueError("alias exceeds its allocation")
    for i, left in enumerate(aliases):
        for right in aliases[i + 1 :]:
            overlap = max(left["offset"], right["offset"]) < min(
                left["offset"] + left["bytes"], right["offset"] + right["bytes"]
            )
            if left["buffer"] == right["buffer"] and overlap and set(left["phases"]) & set(right["phases"]):
                raise ValueError("overlapping live aliases: " + left["name"] + " / " + right["name"])


def aliases(version: int = CONTRACT_VERSION) -> list[dict]:
    if type(version) is not int or version not in (1, 2):
        raise ValueError("unsupported streaming contract version")

    def alias(name, buffer, offset, size, *phases):
        return {"name": name, "buffer": buffer, "offset": offset, "bytes": size, "phases": list(phases)}

    result = [
        alias("decode", "decoded", 0, 65536, "reconstruction"),
        alias("raw_product_0", "decoded", 0, PRODUCT_SLOT_BYTES, "projection"),
        alias("raw_product_1", "decoded", PRODUCT_SLOT_BYTES, PRODUCT_SLOT_BYTES, "projection"),
        alias("swiglu_low", "decoded", 32768, 16384, "nonlinear"),
        alias("swiglu_high", "decoded", 49152, 16384, "nonlinear"),
        alias("quant_padded", "decoded", 0, 1024, "quantization"),
        alias("decode_mask", "mask", 0, 24576, "reconstruction"),
        alias("factors", "mask", 0, 16384, "projection"),
        alias("row_broadcast", "mask", 16384, 1024, "projection"),
        alias("retained_indices", "mask", 24576, 128, *PHASES),
        alias("quant_scratch", "mask", 0, 16384, "quantization"),
        alias("converted_products", "stream_converted", 0, CONVERTED_BYTES, "projection"),
        alias("hidden_tile", "stream_converted", 0, 4096, "quantization", "handoff"),
    ]
    if version == 2:
        result.extend(
            [
                alias("retained_expert_ends", "mask", 24704, 2304, *PHASES),
                alias("scale_gather_indices", "mask", 27008, 1024, *PHASES),
                alias("group_scale_source", "mask", 28032, 128, *PHASES),
            ]
        )
    return result


@dataclass
class SlotLedger:
    """Conservative logical ownership; real hardware event ordering needs a gate."""

    states: tuple[str, str] = ("free", "free")

    def transition(self, slot: int, operation: str) -> None:
        if type(slot) is not int or slot not in (0, 1):
            raise ValueError("slot must be 0 or 1")
        edges = {
            "load": (("free", "consumed"), "operands_ready"),
            "issue": (("operands_ready",), "cube_issued"),
            "readback": (("cube_issued",), "read_back"),
            "consume": (("read_back",), "consumed"),
            "drain": (("consumed", "free"), "free"),
        }
        if operation not in edges or self.states[slot] not in edges[operation][0]:
            raise ValueError("unsafe slot transition: " + operation)
        values = list(self.states)
        values[slot] = edges[operation][1]
        self.states = tuple(values)


def memory_contract(sdk: dict | None = None, *, header_path: Path | None = None) -> dict:
    """Produce current bounds bound to an explicit frozen or default workspace header."""
    header = Path(header_path) if header_path is not None else Path(__file__).with_name("glm_streaming_contract.h")
    version = header_version(header)
    lifetimes = aliases(version)
    capacities = dict(zip(("ub", "l1", "l0a", "l0b", "l0c"), SDK_CAPACITIES))
    if sdk is not None:
        if sdk.get("architecture") != 2002 or sdk.get("capacities") != capacities:
            raise ValueError("installed SDK capacities do not match dav-2002 proof")
    projections = {}
    for gate in (True, False):
        for compact in (False, True):
            buffers = projection_allocations(gate=gate, compact_w4=compact)
            table = allocation_table(buffers)
            total = sum(row["aligned_bytes"] for row in table)
            validate_aliases(lifetimes, buffers)
            local = {"l1": 786432 if gate else 266240, "l0a": 4096, "l0b": 8192, "l0c": 65536}
            if total + SDK_RESERVE > capacities["ub"] or any(local[space] > capacities[space] for space in local):
                raise ValueError("projection exceeds local memory capacity")
            key = ("gate" if gate else "down") + ("_w4" if compact else "_generic")
            projections[key] = {
                "ub_allocations": table,
                "ub_total": total,
                "ub_spare_after_sdk_reserve": capacities["ub"] - SDK_RESERVE - total,
                "local_totals": local,
                "extra_ub_vs_v1021": CONVERTED_BYTES,
            }
    result = {
        "schema_version": 1,
        "architecture": 2002,
        "header_sha256": hashlib.sha256(header.read_bytes()).hexdigest(),
        "capacities": capacities,
        "sdk_ub_reserve": SDK_RESERVE,
        "sdk_verified": sdk is not None,
        "sdk_evidence": sdk,
        "hardware_validated": False,
        "geometry": {
            "rows": ROWS,
            "max_valid_rows": 31,
            "columns": COLUMNS,
            "group": GROUP,
            "pair_k": 64,
            "weight_tile_k": 256,
            "max_k": MAX_K,
            "slots": SLOTS,
            "physical_product_rows": {"count_1_to_16": 32, "count_17_to_31": 64},
        },
        "projections": projections,
        "aliases": lifetimes,
        "events": [
            {"kind": kind, "ids": [0, 1], "producer": producer, "consumer": consumer}
            for kind, producer, consumer in EVENTS
        ],
        "schedule": {
            "owner": "T6 Projection issue loop",
            "prologue": "load/issue pair 0; mark operands and result events; read back to raw slot 0",
            "steady_state": "load/issue j+1 in alternate L0 slot; consume j in vector; read back j+1",
            "drain": "consume final pair; wait/consume outstanding M_MTE1, MTE1_MTE2 and V_M flags for used slots",
            "reconstruction_boundary": "drain both slots and fence PIPE_ALL before noncached next K256 Decode",
            "weight_vector_boundary": "finish previous K256 vector readers before updating its eight scale vectors",
            "gate_up_boundary": "drain gate products; keep output_ FP16 gate live; retain L1 activation cache for up",
            "quantization_boundary": "drain up; nonlinear reads decoded low/high; quantizer uses mask and decoded",
            "handoff_boundary": "after each quantizer run, stage packed tile in dead converted plane; flush all rows",
            "scope": "allocation/order proof only; compiled device event legality and overlap remain unqualified",
        },
        "global_memory_delta": 0,
        "partial_products_if_materialized": 31 * 128 * 128 * 4,
        "admission_scope": "per-core allocations only; whole-rank standalone/live overlap needs separate admission",
    }
    if version == 2:
        result["contract_version"] = version
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sdk-evidence", type=Path)
    args = parser.parse_args()
    sdk = json.loads(args.sdk_evidence.read_text()) if args.sdk_evidence else None
    result = memory_contract(sdk)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
