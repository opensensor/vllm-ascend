# SPDX-License-Identifier: Apache-2.0
"""Admission failures and native integer math versus independent Python math."""

import ctypes
import random
import shutil
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest

from tools.qwen4exp.ple_request import PLERequestGeometry


@pytest.fixture
def geometry():
    sizes = (
        20000003,
        20000023,
        20000033,
        20000047,
        20000059,
        20000063,
        20000069,
        20000077,
        20000081,
        20000083,
        20000107,
        20000111,
        20000117,
        20000119,
        20000143,
        20000149,
    )
    offsets = []
    end = 0
    for size in sizes:
        offsets.append(end)
        end += size
    return PLERequestGeometry(
        12,
        4,
        248044,
        248320,
        (23703573157769, 20109073645365, 8052911324071),
        sizes,
        tuple(offsets),
        (end + 127) // 128 * 128,
    )


@pytest.mark.parametrize(
    "changes",
    [
        {"tokens": 0},
        {"tokens": 2049},
        {"requests": 0},
        {"requests": 5},
        {"tokens": True},
        {"requests": 1.5},
        {"eos": -1},
        {"eos": 248320},
        {"table_rows": 1},
        {"table_rows": 1 << 63},
        {"multipliers": (1, 2)},
        {"multipliers": (1, 2, 1 << 62)},
        {"multipliers": (1, 2, 0)},
        {"sizes": (1,) * 15},
        {"offsets": (0,) * 16},
        {"vocabulary": 0},
    ],
)
def test_invalid_admission(geometry, changes):
    with pytest.raises(ValueError):
        replace(geometry, **changes)


def test_geometry_freezes_sequences(geometry):
    multipliers = list(geometry.multipliers)
    admitted = replace(geometry, multipliers=multipliers)
    multipliers[0] = 0
    assert admitted == geometry
    assert len(geometry.values()) == 39


def test_native_hash_integer_exact(geometry, tmp_path):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("requires C++ compiler")
    source = tmp_path / "hash.cpp"
    source.write_text("""
#include "ple_request_integer.h"
extern "C" void select_rows(int64_t a, int64_t b, int64_t c, int64_t eos,
    const uint64_t* m, const int64_t* sizes, const int64_t* offsets, int64_t* result) {
    for (int head=0; head<16; ++head) {
        const auto mix=ple_request::Mix(a,b,c,eos,m[0],m[1],m[2],head);
        result[head]=ple_request::Row(mix,sizes[head],offsets[head]);
    }
}
""")
    root = Path(__file__).resolve().parents[3]
    library = tmp_path / "hash.so"
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-shared",
            "-fPIC",
            "-O2",
            "-Wall",
            "-Wextra",
            "-Werror",
            f"-I{root / 'tools/qwen4exp'}",
            str(source),
            "-o",
            str(library),
        ],
        check=True,
    )
    function = ctypes.CDLL(str(library)).select_rows
    signed = ctypes.c_int64
    function.argtypes = [signed] * 4 + [ctypes.POINTER(ctypes.c_uint64)] + [ctypes.POINTER(signed)] * 3
    function.restype = None
    cfg = geometry
    multipliers = (ctypes.c_uint64 * 3)(*cfg.multipliers)
    sizes, offsets, actual = (signed * 16)(*cfg.sizes), (signed * 16)(*cfg.offsets), (signed * 16)()
    rng = random.Random(14102026)
    triples = [
        (rng.randrange(cfg.vocabulary), rng.randrange(cfg.vocabulary), rng.randrange(cfg.vocabulary))
        for _ in range(500)
    ]
    triples += [
        (a, b, c)
        for a in (0, cfg.eos, cfg.vocabulary - 1)
        for b in (0, cfg.eos, cfg.vocabulary - 1)
        for c in (0, cfg.eos, cfg.vocabulary - 1)
    ]
    for a, b, c in triples:
        function(a, b, c, cfg.eos, multipliers, sizes, offsets, actual)
        history = [a, b, cfg.eos if b == cfg.eos else c]
        expected = []
        for head in range(16):
            mixed = 0
            for token, multiplier in zip(history[: 2 + head // 8], cfg.multipliers):
                mixed ^= token * multiplier
            expected.append(mixed % cfg.sizes[head] + cfg.offsets[head])
        assert list(actual) == expected
