# SPDX-License-Identifier: Apache-2.0
"""CPU-only validation of the bounded ARM process access diagnostic."""

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture(scope="module")
def probe(tmp_path_factory):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("requires C++ compiler")
    directory = tmp_path_factory.mktemp("ple_access")
    source = directory / "probe.cpp"
    source.write_text(r"""
#include "ple_device_access.h"
#include <cassert>
#include <cstring>
int main(int argc, char** argv) {
    assert(argc == 3);
    char bytes[8193];
    std::memset(bytes, 'X', sizeof(bytes));
    const bool small = std::strcmp(argv[1], "small") == 0;
    const size_t capacity = small ? 16 : 8192;
    PleProbeText out(bytes, capacity);
    if (std::strcmp(argv[1], "file") == 0) ple_access_file(out, argv[2]);
    else ple_access_node(out, argv[2]);
    assert(bytes[capacity] == 'X');
    assert(out.size() < capacity);
    assert(bytes[out.size()] == 0);
    if (small) assert(out.truncated());
    std::fwrite(bytes, 1, out.size(), stdout);
}
""")
    binary = directory / "probe"
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-Wall",
            "-Wextra",
            "-Werror",
            f"-I{ROOT / 'tools/qwen4exp'}",
            str(source),
            "-o",
            str(binary),
        ],
        check=True,
    )
    return binary


def run_probe(probe, mode, path):
    return subprocess.run([str(probe), mode, str(path)], check=True, capture_output=True, text=True).stdout


def test_missing_path_reports_failure(probe, tmp_path):
    output = run_probe(probe, "node", tmp_path / "missing")
    assert "stat_errno=2" in output
    assert "open_readonly_errno=2" in output
    assert "uid=" not in output


def test_regular_file_is_identified_without_reading_or_mutation(probe, tmp_path):
    path = tmp_path / "private"
    path.write_text("private contents must not enter the node report")
    path.chmod(0o600)
    output = run_probe(probe, "node", path)
    assert "mode=0600 character_device=0" in output
    assert "open_readonly_errno=0" in output
    assert "private contents" not in output
    assert path.read_text() == "private contents must not enter the node report"


def test_file_read_is_bounded_and_reported(probe, tmp_path):
    path = tmp_path / "status"
    path.write_text("x" * 4096 + "excluded tail")
    output = run_probe(probe, "file", path)
    assert "file_limit_reached=1" in output
    assert "excluded tail" not in output
    path.write_text("Groups: 1000\n")
    output = run_probe(probe, "file", path)
    assert "Groups: 1000" in output
    assert "file_limit_reached=0" in output
    assert "open_errno=2" in run_probe(probe, "file", tmp_path / "missing")


def test_tiny_receipt_preserves_guard(probe, tmp_path):
    assert len(run_probe(probe, "small", tmp_path / "missing")) == 15


def test_standalone_compiles_without_ascend_sdk(tmp_path):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("requires C++ compiler")
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-c",
            str(ROOT / "tools/qwen4exp/ple_device_access.cpp"),
            "-o",
            str(tmp_path / "probe.o"),
        ],
        check=True,
    )
