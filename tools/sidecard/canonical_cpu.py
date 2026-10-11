# SPDX-License-Identifier: Apache-2.0
"""Small canonical packed-weight oracle, loaded only inside CPU worker processes.

This preserves signed codes and block32 scales. It is not the qualified 310P
activation quantizer or an accelerator backend and rejects full GLM dimensions.
"""

import math
import struct
from pathlib import Path

from .bundle import PartitionStore
from .reference import fp32
from .worker import DenseExpert, DenseExpertBank

BLOCK_SIZE = 32
MAX_CPU_WIDTH = 64
DEFAULT_ELEMENT_LIMIT = 100_000


def decode_matrix(codes, scales, rows, columns, bits, scale_dtype):
    if bits not in (2, 3, 4) or rows % BLOCK_SIZE or columns % BLOCK_SIZE or rows <= 0 or columns <= 0:
        raise ValueError("invalid canonical matrix geometry")
    if rows > MAX_CPU_WIDTH or columns > MAX_CPU_WIDTH:
        raise ValueError("CPU oracle is restricted to small matrices")
    if scale_dtype not in ("F16", "F32") or len(codes) != rows * columns * bits // 8:
        raise ValueError("invalid canonical matrix storage")
    scale_struct = struct.Struct("<e" if scale_dtype == "F16" else "<f")
    if len(scales) != rows // BLOCK_SIZE * (columns // BLOCK_SIZE) * scale_struct.size:
        raise ValueError("invalid canonical scale storage")
    factors = tuple(value[0] for value in scale_struct.iter_unpack(scales))
    if any(not math.isfinite(value) or value < 0 for value in factors):
        raise ValueError("invalid canonical scale value")
    fields, mask, sign = [], (1 << bits) - 1, 1 << (bits - 1)
    group_bytes, group_fields = (3, 8) if bits == 3 else (1, 8 // bits)
    for start in range(0, len(codes), group_bytes):
        word = int.from_bytes(codes[start : start + group_bytes], "little")
        for field in range(group_fields):
            value = (word >> (bits * field)) & mask
            fields.append(value - (1 << bits) if value >= sign else value)
    return tuple(
        tuple(
            fp32(
                fields[row * columns + column]
                * factors[row // BLOCK_SIZE * (columns // BLOCK_SIZE) + column // BLOCK_SIZE]
            )
            for column in range(columns)
        )
        for row in range(rows)
    )


class CanonicalCPUBackend:
    """Picklable startup specification; verified owner tensors load in the child."""

    def __init__(self, root: Path, owner, placement, identity, hidden_width, element_limit=DEFAULT_ELEMENT_LIMIT):
        if hidden_width not in (BLOCK_SIZE, MAX_CPU_WIDTH) or type(element_limit) is not int or element_limit <= 0:
            raise ValueError("invalid CPU oracle dimensions or element budget")
        self.root, self.owner, self.placement, self.identity = root, owner, placement, identity
        self.hidden_width, self.element_limit = hidden_width, element_limit
        self.bank = None

    def prepare(self):
        self.bank = None
        store = PartitionStore(self.root, self.owner, self.placement, self.identity)
        if store.hidden_width != self.hidden_width:
            raise ValueError("partition hidden width differs from CPU backend specification")
        keys = tuple(
            (layer, expert)
            for layer in range(
                self.placement.first_moe_layer, self.placement.first_moe_layer + len(self.placement.layers)
            )
            for expert, owner in enumerate(self.placement.layer(layer))
            if owner == self.owner
        )
        geometries, elements = {}, 0
        for layer, expert in keys:
            prefix = f"model.language_model.layers.{layer}.mlp.experts.{expert}."
            intermediate = store.records[prefix + "gate_proj_codes"].shape[0]
            if intermediate not in (BLOCK_SIZE, MAX_CPU_WIDTH):
                raise ValueError("CPU oracle is restricted to small intermediate dimensions")
            elements += 3 * intermediate * self.hidden_width
            geometries[layer, expert] = intermediate
        if elements > self.element_limit:
            raise ValueError("expanded CPU bank exceeds element budget")
        experts = []
        for layer, expert in keys:
            matrices = []
            prefix = f"model.language_model.layers.{layer}.mlp.experts.{expert}."
            intermediate = geometries[layer, expert]
            for projection in ("gate", "up", "down"):
                code_name, scale_name = prefix + projection + "_proj_codes", prefix + projection + "_proj_scale"
                code, scale = store.records[code_name], store.records[scale_name]
                rows, columns = (
                    (self.hidden_width, intermediate) if projection == "down" else (intermediate, self.hidden_width)
                )
                bits = next((bits for bits in (2, 3, 4) if code.shape == (rows, columns * bits // 8)), None)
                if code.dtype != "U8" or bits is None or scale.shape != (rows // BLOCK_SIZE, columns // BLOCK_SIZE):
                    raise ValueError("CPU oracle expert geometry mismatch")
                matrices.append(
                    decode_matrix(
                        store.read_tensor(code_name), store.read_tensor(scale_name), rows, columns, bits, scale.dtype
                    )
                )
            experts.append((layer, expert, DenseExpert(*matrices)))
        self.bank = DenseExpertBank(tuple(experts))
        return self.bank.prepare()

    def compute(self, request):
        if self.bank is None:
            raise RuntimeError("CPU partition backend has not completed preparation")
        return self.bank.compute(request)
