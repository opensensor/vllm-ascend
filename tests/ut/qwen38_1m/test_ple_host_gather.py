# SPDX-License-Identifier: Apache-2.0
"""Native request admission and scattered gather, including untouched error output."""

import ctypes
import fcntl
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.qwen4exp import benchmark_ple_host_gather_310 as benchmark
from tools.qwen4exp.ple_sources import IMMUTABLE_SEALS


@pytest.fixture(scope="module")
def native(tmp_path_factory):
    root = Path(__file__).resolve().parents[3]
    tmp = tmp_path_factory.mktemp("ple-host-gather")
    source = tmp / "test.cpp"
    source.write_text("""
#include "ple_host_gather.h"
extern "C" int gather(const void* table, size_t bytes, const void* packet, size_t tokens, void* output) {
    return ple_host::gather(table, bytes, packet, tokens, output);
}
""")
    library = tmp / "test.so"
    subprocess.run(
        [
            "c++",
            "-std=c++17",
            "-O2",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-shared",
            "-fPIC",
            f"-I{root / 'tools/qwen4exp'}",
            str(source),
            "-o",
            str(library),
        ],
        check=True,
    )
    function = ctypes.CDLL(str(library)).gather
    function.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p]
    function.restype = ctypes.c_int
    return function


@pytest.mark.parametrize("tokens", [1, 3, 12])
def test_scattered_duplicates_and_page_crossing(native, tokens):
    rows = [12, 0, 63, 12, *range(20, 32)] * tokens
    table = bytes((i * 17 + 13) % 251 for i in range(64 * 320))
    source = ctypes.create_string_buffer(table)
    packet = (ctypes.c_int64 * (tokens * 20))()
    for i, row in enumerate(rows):
        packet[i] = row
    output = ctypes.create_string_buffer(tokens * 5120)
    assert native(source, len(table), packet, tokens, output) == 0
    assert output.raw == b"".join(table[row * 320 : (row + 1) * 320] for row in rows)


@pytest.mark.parametrize(
    "bad", ["negative", "end", "overflow", "receipt", "zero_tokens", "too_many", "unaligned_table"]
)
def test_rejects_entire_request_before_writing(native, bad):
    tokens = 3
    table = ctypes.create_string_buffer(64 * 320)
    packet = (ctypes.c_int64 * (tokens * 20))()
    if bad in {"negative", "end", "overflow"}:
        packet[tokens * 16 - 1] = {"negative": -1, "end": 64, "overflow": (1 << 63) - 1}[bad]
    if bad == "receipt":
        receipts = ctypes.cast(ctypes.byref(packet, tokens * 16 * 8), ctypes.POINTER(ctypes.c_int32))
        receipts[2 * 8] = 2
    output = ctypes.create_string_buffer(b"x" * (tokens * 5120), tokens * 5120)
    count = 0 if bad == "zero_tokens" else 13 if bad == "too_many" else tokens
    size = 63 if bad == "unaligned_table" else len(table)
    assert native(table, size, packet, count, output) != 0
    assert output.raw == b"x" * len(output)


@pytest.mark.parametrize("fail_worker", [False, True])
def test_runner_shares_sealed_source_and_restores_budget(tmp_path, monkeypatch, fail_worker):
    parameter = tmp_path / "parameter"
    parameter.write_text("5\n")
    monkeypatch.setattr(benchmark, "PARAMETER", str(parameter))
    calls = []
    descriptors = []

    def run(command, **kwargs):
        calls.append(command)
        if command[:2] == ["sudo", "tee"]:
            parameter.write_text(kwargs["input"])
        else:
            (fd,) = kwargs["pass_fds"]
            descriptors.append(fd)
            assert os.fstat(fd).st_size == 64 * 320
            assert fcntl.fcntl(fd, fcntl.F_GET_SEALS) & IMMUTABLE_SEALS == IMMUTABLE_SEALS
            assert "--native-producer" in command
            if fail_worker:
                raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(benchmark.subprocess, "run", run)
    args = SimpleNamespace(
        output=tmp_path / "run",
        synthetic=True,
        model=None,
        build=tmp_path,
        library=tmp_path / "native.so",
        devices=[0, 1],
        native_producer=True,
        native_only=False,
        spin_budgets=[0, 5],
        cpu=8,
    )
    if fail_worker:
        with pytest.raises(subprocess.CalledProcessError):
            benchmark.parent(args)
    else:
        benchmark.parent(args)
        assert len(descriptors) == 4
    assert parameter.read_text() == "5\n"
    assert len(set(descriptors)) == 1
    with pytest.raises(OSError):
        os.fstat(descriptors[0])


def test_actual_native_producer_wait_policy(tmp_path):
    root = Path(__file__).resolve().parents[3]
    source = (root / "tools/qwen4exp/native_ple_host_gather.cpp").read_text()
    start = source.index("static int produce_fetch(")
    (tmp_path / "producer_wait.inc").write_text(source[start:])
    subprocess.run(
        [
            "c++",
            "-std=c++17",
            "-Wall",
            "-Wextra",
            "-Werror",
            f"-I{tmp_path}",
            str(Path(__file__).parent / "cpp/ple_producer_wait.cpp"),
            "-o",
            str(tmp_path / "test"),
        ],
        check=True,
    )
    subprocess.run([str(tmp_path / "test")], check=True)


def test_populated_mapping_shared_lifetime(tmp_path):
    root = Path(__file__).resolve().parents[3]
    subprocess.run(
        [
            "c++",
            "-std=c++17",
            "-Wall",
            "-Wextra",
            "-Werror",
            f"-I{root / 'tools/qwen4exp'}",
            str(Path(__file__).parent / "cpp/ple_shared_mapping.cpp"),
            "-o",
            str(tmp_path / "mapping-test"),
        ],
        check=True,
    )
    subprocess.run([str(tmp_path / "mapping-test")], check=True)
