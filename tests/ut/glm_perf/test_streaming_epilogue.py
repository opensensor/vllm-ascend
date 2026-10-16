# SPDX-License-Identifier: Apache-2.0
"""Byte-exact checks of the production packed-code helper, without device timing."""

import json
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]


@pytest.fixture(scope="module", params=["optimized", "sanitized"])
def epilogue_checker(tmp_path_factory, request):
    executable = tmp_path_factory.mktemp("streaming-epilogue") / "check"
    compiler_flags = ["-O2"]
    if request.param == "sanitized":
        compiler_flags = ["-O1", "-fsanitize=address,undefined", "-fno-omit-frame-pointer", "-no-pie"]
    subprocess.run(
        [
            "c++",
            "-std=c++17",
            *compiler_flags,
            f"-I{Path(__file__).parent / 'streaming_epilogue_cpu_stubs'}",
            f"-I{REPO / 'tools/glm_perf'}",
            str(Path(__file__).parent / "reduce_kernel_cpu_stubs/check_streaming_epilogue.cpp"),
            "-o",
            str(executable),
        ],
        check=True,
    )
    return executable


@pytest.mark.parametrize("count", range(32))
def test_actual_helper_preserves_records_padding_ownership_and_copy_counts(epilogue_checker, count):
    result = subprocess.run([str(epilogue_checker), str(count)], check=True, capture_output=True, text=True)
    model = json.loads(result.stdout)
    assert model == {
        "count": count,
        "old_gm_calls": 2 + 4 * count,
        "old_gm_bytes": 4096 + 128 * count,
        "new_gm_calls": 2,
        "new_gm_bytes": 4096,
        "old_local_calls": 4 * count,
        "local_calls": 4 * ((count + 3) // 4),
        "local_bytes": 128 * count,
    }


def test_changing_route_counts_clear_all_padding(epilogue_checker):
    result = subprocess.run([str(epilogue_checker), "replay"], check=True, capture_output=True, text=True)
    rows = [json.loads(line) for line in result.stdout.splitlines()]
    assert [row["count"] for row in rows[:4]] == [31, 5, 4, 0]


def test_zero_quantized_activations_remain_zero(epilogue_checker):
    result = subprocess.run([str(epilogue_checker), "zero"], check=True, capture_output=True, text=True)
    assert [json.loads(line)["count"] for line in result.stdout.splitlines()] == [31, 5]


def test_header_has_no_tensor_allocation_or_quantizer_change():
    source = (REPO / "tools/glm_perf/glm_streaming_epilogue.h").read_text()
    assert "TBuf" not in source and "InitBuffer" not in source
    assert "GlmFusedQuant::" not in source and "GlmFusedDownScales::" not in source
    assert "PipeBarrier<PIPE_ALL>" not in source[source.index("inline void Stage") : source.index("inline void Store")]
