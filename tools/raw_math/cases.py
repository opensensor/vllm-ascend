# SPDX-License-Identifier: Apache-2.0
"""Bounded model-independent cases and precision-aware result validation."""

import math
import statistics
from dataclasses import asdict, dataclass

PRECISIONS = ("fp16", "fp32", "bf16", "int8")
MAX_DIM = 8192
MAX_ELEMENTS = 64 * 1024**2
MAX_COPY_BYTES = 1024**3
SEED = 6000310


@dataclass(frozen=True)
class Case:
    name: str
    op: str
    dtype: str
    m: int
    n: int = 0
    k: int = 0
    pinned: bool = False

    def __post_init__(self):
        if type(self.pinned) is not bool or any(type(x) is not int for x in (self.m, self.n, self.k)):
            raise ValueError("case geometry must use integer dimensions and boolean pinning")
        if self.op == "gemm":
            valid = self.dtype in PRECISIONS and all(1 <= x <= MAX_DIM for x in (self.m, self.n, self.k))
        elif self.op in ("add", "exp"):
            valid = self.dtype == "fp32" and 1 <= self.m <= MAX_ELEMENTS and self.n == self.k == 0
        elif self.op in ("softmax", "rmsnorm"):
            valid = self.dtype == "fp32" and 1 <= self.m <= MAX_DIM and 1 <= self.n <= MAX_DIM and self.k == 0
        elif self.op in ("copy", "h2d", "d2h"):
            valid = self.dtype == "uint8" and 1 <= self.m <= MAX_COPY_BYTES and self.n == self.k == 0
        else:
            valid = False
        if not valid or self.pinned and self.op not in ("h2d", "d2h"):
            raise ValueError("case exceeds the bounded arithmetic contract")

    @property
    def output_dtype(self):
        return "fp16" if self.dtype == "int8" else self.dtype

    @property
    def repeats(self):
        if self.op == "gemm":
            return 1 if self.m >= 4096 else 4 if self.m >= 128 else 32
        if self.op in ("copy", "h2d", "d2h", "add", "exp"):
            return 1 if self.m >= 16 * 1024**2 else 16
        return 4 if self.m >= 128 else 32

    def to_dict(self):
        return asdict(self)


def matrix_cases(precisions=PRECISIONS):
    if not precisions or any(dtype not in PRECISIONS for dtype in precisions):
        raise ValueError("select supported matrix precisions")
    shapes = (
        ("narrow", 1, 128, 4096),
        ("vector", 1, 4096, 4096),
        ("vector_large", 1, 8192, 8192),
        ("batch8", 8, 4096, 4096),
        ("batch128", 128, 4096, 4096),
        ("batch512", 512, 4096, 4096),
        ("batch2048", 2048, 4096, 4096),
        ("square4096", 4096, 4096, 4096),
        ("square8192", 8192, 8192, 8192),
    )
    return [Case(name, "gemm", dtype, m, n, k) for dtype in precisions for name, m, n, k in shapes]


def vector_cases():
    result = [
        Case(f"{op}_{size}", op, "fp32", size)
        for op in ("add", "exp")
        for size in (4096, 1024**2, 8 * 1024**2, MAX_ELEMENTS)
    ]
    result += [
        Case(f"{op}_{rows}x{width}", op, "fp32", rows, width)
        for op in ("softmax", "rmsnorm")
        for rows, width in ((1, 256), (1, 4096), (128, 4096), (2048, 4096))
    ]
    return result


def copy_cases():
    sizes = (8 * 1024, 1024**2, 16 * 1024**2, 256 * 1024**2, MAX_COPY_BYTES)
    return [
        Case(f"{op}_{size}", op, "uint8", size, pinned=pinned)
        for op in ("copy", "h2d", "d2h")
        for size in sizes
        for pinned in ((False,) if op == "copy" else (False, True))
    ]


def summary(values):
    if not values or any(not math.isfinite(x) or x <= 0 for x in values):
        raise ValueError("timings must be positive and finite")
    return dict(median_ms=statistics.median(values), min_ms=min(values), max_ms=max(values), samples_ms=values)


def rates(case, milliseconds):
    if not math.isfinite(milliseconds) or milliseconds <= 0:
        raise ValueError("positive finite timing required")
    if case.op == "gemm":
        return {
            "tera_ops_per_second": 2 * case.m * case.n * case.k / (milliseconds * 1e9),
            "operations": "integer multiply/add" if case.dtype == "int8" else "floating multiply/add",
        }
    if case.op in ("copy", "h2d", "d2h"):
        return {
            "payload_gbytes_per_second": case.m / (milliseconds * 1e6),
            "read_plus_write_gbytes_per_second": 2 * case.m / (milliseconds * 1e6) if case.op == "copy" else None,
        }
    return {"elements_per_second": case.m * (case.n or 1) / (milliseconds / 1000)}
