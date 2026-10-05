"""Host gate for the exact selector shared with the experimental AI CPU op."""

import ctypes
import math
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "tools/qwen4exp/qsa_exact_topk_host.cpp"


@pytest.fixture(scope="module")
def selector_library(tmp_path_factory):
    if shutil.which("g++") is None:
        pytest.skip("g++ is required for the host C++ parity gate")
    library = tmp_path_factory.mktemp("qsa_exact_topk") / "selector.so"
    subprocess.run(
        ["g++", "-O2", "-std=c++17", "-fPIC", "-shared", "-pthread", str(SOURCE), "-o", str(library)],
        check=True,
    )
    return ctypes.CDLL(str(library))


@pytest.fixture(scope="module")
def selector(selector_library):
    function = selector_library.qsa_exact_topk_host
    function.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.POINTER(ctypes.c_int32),
    ]
    function.restype = ctypes.c_int
    return function


@pytest.fixture(scope="module")
def shape_resolver(selector_library):
    function = selector_library.qsa_exact_topk_shape_host
    function.argtypes = [
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.POINTER(ctypes.c_int64),
        ctypes.POINTER(ctypes.c_int64),
    ]
    function.restype = ctypes.c_int
    return function


@pytest.mark.parametrize(
    ("score_elements", "index_elements", "topk", "expected_queries", "expected_groups"),
    [(17568, 1536, 512, 3, 5856), (2048 * 10000, 2048 * 512, 512, 2048, 10000)],
)
def test_flat_aicpu_descriptors_resolve_logical_shape(
    shape_resolver, score_elements, index_elements, topk, expected_queries, expected_groups
):
    queries = ctypes.c_int64()
    groups = ctypes.c_int64()
    assert shape_resolver(score_elements, index_elements, topk, ctypes.byref(queries), ctypes.byref(groups)) == 0
    assert (queries.value, groups.value) == (expected_queries, expected_groups)


@pytest.mark.parametrize(
    ("score_elements", "index_elements", "topk"),
    [(0, 1536, 512), (17568, 1537, 512), (17568, 1536, 0), (300, 1536, 512)],
)
def test_flat_aicpu_descriptors_reject_inconsistent_buffers(shape_resolver, score_elements, index_elements, topk):
    queries = ctypes.c_int64()
    groups = ctypes.c_int64()
    assert shape_resolver(score_elements, index_elements, topk, ctypes.byref(queries), ctypes.byref(groups)) != 0


@pytest.mark.parametrize(
    ("rows", "topk"),
    [
        ([[1.0, 1.0, 1.0, 1.0]], 2),
        ([[float("-inf"), 1.0, float("-inf"), 1.0]], 3),
        ([[0.0, -0.0, 2.0, 2.0], [3.0, 1.0, 3.0, 1.0]], 3),
        ([[float(i % 7) for i in range(257)]], 64),
    ],
)
def test_exact_order_and_cutoff_ties(selector, rows, topk):
    width = len(rows[0])
    flat = (ctypes.c_float * (len(rows) * width))(*(value for row in rows for value in row))
    output = (ctypes.c_int32 * (len(rows) * topk))()
    assert selector(flat, len(rows), width, topk, output) == 0
    expected = [index for row in rows for index in sorted(range(width), key=lambda group: (-row[group], group))[:topk]]
    assert list(output) == expected


def test_rejects_nan_and_invalid_topk(selector):
    scores = (ctypes.c_float * 2)(1.0, math.nan)
    output = (ctypes.c_int32 * 2)()
    assert selector(scores, 1, 2, 1, output) != 0
    scores[1] = 0.0
    assert selector(scores, 1, 2, 0, output) != 0
    assert selector(scores, 1, 2, 3, output) != 0


def test_sharded_rows_match_one_call(selector):
    rows = [[float((row * 13 + column * 7) % 19) for column in range(513)] for row in range(9)]
    width, topk = 513, 64
    flat = (ctypes.c_float * (len(rows) * width))(*(value for row in rows for value in row))
    complete = (ctypes.c_int32 * (len(rows) * topk))()
    sharded = (ctypes.c_int32 * (len(rows) * topk))()
    assert selector(flat, len(rows), width, topk, complete) == 0
    for first, last in ((0, 2), (2, 6), (6, 9)):
        score_slice = ctypes.cast(
            ctypes.byref(flat, first * width * ctypes.sizeof(ctypes.c_float)), ctypes.POINTER(ctypes.c_float)
        )
        output_slice = ctypes.cast(
            ctypes.byref(sharded, first * topk * ctypes.sizeof(ctypes.c_int32)), ctypes.POINTER(ctypes.c_int32)
        )
        assert selector(score_slice, last - first, width, topk, output_slice) == 0
    assert list(sharded) == list(complete)
