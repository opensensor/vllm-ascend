# SPDX-License-Identifier: Apache-2.0
"""Compile the actual M16 boundary helper with bounded CPU DMA semantics."""

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
STUBS = Path(__file__).parent / "decode_end_cache_cpu_stubs"
HELPER = ROOT / "tools/glm_perf/glm_decode_end_cache.h"


def compile_checker(output, *, helper_dir=None, sanitizers=False):
    compiler = (shutil.which("clang++") or shutil.which("c++")) if sanitizers else shutil.which("c++")
    if compiler is None:
        pytest.skip("requires host C++ compiler" + (" with sanitizer support" if sanitizers else ""))
    flags = ["-O1", "-fsanitize=address,undefined", "-fno-omit-frame-pointer"] if sanitizers else ["-O2"]
    command = [compiler, "-std=c++17", "-Wall", "-Wextra", "-Werror", *flags, f"-I{STUBS}"]
    if helper_dir is not None:
        command.append(f"-I{helper_dir}")
    command += [f"-I{ROOT / 'tools/glm_perf'}", str(STUBS / "check_decode_end_cache.cpp"), "-o", str(output)]
    subprocess.run(command, check=True, capture_output=True, text=True)
    return output


@pytest.fixture(scope="module")
def checker(tmp_path_factory):
    return compile_checker(tmp_path_factory.mktemp("decode-end-cache") / "checker")


@pytest.mark.parametrize("experts", range(1, 289))
@pytest.mark.parametrize("pattern", [0, 1, 2], ids=["all_peer", "monotonic_empty", "exact_int64_bits"])
def test_every_expert_length_refreshes_exact_bits_and_preserves_mask(checker, experts, pattern):
    result = subprocess.run([str(checker), str(experts), str(pattern), "1"], capture_output=True, text=True, check=True)
    assert "exact cache bits, DMA bounds/fences, neighbors, replay and no-allocation passed" in result.stdout


@pytest.mark.parametrize(
    "mutation,experts,error",
    [
        ("overround", 73, "DMA overread"),
        ("dma_offset", 72, "DMA address alignment"),
        ("neighbor_tail", 73, "cache neighbor"),
        ("first_fence", 73, "before DMA fence"),
        ("final_fence", 72, "incomplete prologue fences"),
        ("stale_tail", 73, "wrong scalar tail count"),
        ("gm_reader", 72, "GM tail read before DMA fence"),
    ],
)
def test_actual_helper_fault_injections_are_rejected(tmp_path, mutation, experts, error):
    source = HELPER.read_text()
    if mutation == "overround":
        source = source.replace(
            "experts / ENDS_PER_DMA * ENDS_PER_DMA", "(experts + ENDS_PER_DMA - 1) / ENDS_PER_DMA * ENDS_PER_DMA"
        )
    elif mutation == "dma_offset":
        source = source.replace("DataCopy(cached, ends, aligned)", "DataCopy(cached[1], ends, aligned)")
    elif mutation == "neighbor_tail":
        source = source.replace(
            "cached.SetValue(expert, ends.GetValue(expert))", "cached.SetValue(expert + 1, ends.GetValue(expert))"
        )
    elif mutation == "first_fence":
        source = source.replace("  PipeBarrier<PIPE_ALL>();\n", "", 1)
    elif mutation == "final_fence":
        before, separator, after = source.rpartition("  PipeBarrier<PIPE_ALL>();\n")
        assert separator
        source = before + after
    elif mutation == "stale_tail":
        source = source.replace("expert < experts;", "expert + 1 < experts;")
    else:
        source = source.replace(
            "return cached.GetValue(expert);",
            "GlobalTensor<int64_t> wrong{cached.data, cached.bytes}; return wrong.GetValue(expert);",
        )
    assert source != HELPER.read_text()
    (tmp_path / HELPER.name).write_text(source)
    executable = compile_checker(tmp_path / "mutant", helper_dir=tmp_path)
    result = subprocess.run([str(executable), str(experts), "2", "1"], capture_output=True, text=True)
    assert result.returncode != 0
    assert error in result.stderr


def test_boundary_replays_under_address_and_undefined_sanitizers(tmp_path):
    checker = compile_checker(tmp_path / "sanitizer", sanitizers=True)
    result = subprocess.run([str(checker), "sanitizer"], capture_output=True, text=True, check=True)
    assert "no-allocation passed" in result.stdout
