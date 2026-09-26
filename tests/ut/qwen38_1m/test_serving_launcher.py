# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import shlex
import subprocess
from pathlib import Path

import pytest

LAUNCHER = Path(__file__).resolve().parents[3] / "examples/start_qwen38_flash_next_310p.sh"


def _run(tmp_path, *args):
    checkpoint = tmp_path / "checkpoint with spaces"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text("{}")
    (checkpoint / "model.safetensors.index.json").write_text("{}")
    return subprocess.run(
        ["bash", str(LAUNCHER), "--model", str(checkpoint), "--dry-run", *args],
        text=True,
        capture_output=True,
        check=False,
    )


def test_default_launcher_preserves_graph_mtp_and_memory_cap(tmp_path):
    result = _run(tmp_path)
    assert result.returncode == 0, result.stderr
    argv = shlex.split(result.stdout)
    assert argv[:2] == ["vllm", "serve"]
    assert argv[2].endswith("checkpoint with spaces")
    assert argv[argv.index("--gpu-memory-utilization") + 1] == "0.965"
    assert argv[argv.index("--max-model-len") + 1] == "160000"
    assert argv[argv.index("--max-num-seqs") + 1] == "1"
    assert argv[argv.index("--tensor-parallel-size") + 1] == "4"
    assert "--enable-prompt-tokens-details" in argv
    assert "--no-async-scheduling" in argv
    assert json.loads(argv[argv.index("--speculative-config") + 1]) == {"method": "mtp", "num_speculative_tokens": 1}
    assert json.loads(argv[argv.index("--compilation-config") + 1]) == {"cudagraph_mode": "FULL_DECODE_ONLY"}


def test_launcher_accepts_lower_context_and_localhost(tmp_path):
    result = _run(tmp_path, "--max-model-len", "131072", "--host", "127.0.0.1", "--port", "8002")
    assert result.returncode == 0, result.stderr
    argv = shlex.split(result.stdout)
    assert argv[argv.index("--max-model-len") + 1] == "131072"
    assert argv[argv.index("--port") + 1] == "8002"
    assert argv[argv.index("--host") + 1] == "127.0.0.1"


@pytest.mark.parametrize(
    "args",
    [
        ("--gpu-memory-utilization", "0.99"),
        ("--max-model-len", "320000"),
        ("--max-model-len", "0"),
        ("--max-model-len", "131072;false"),
        ("--port", "65536"),
        ("--port", "01"),
        ("--port",),
    ],
)
def test_launcher_rejects_invalid_or_unvalidated_overrides(tmp_path, args):
    assert _run(tmp_path, *args).returncode == 2


def test_launcher_requires_local_checkpoint():
    result = subprocess.run(
        ["bash", str(LAUNCHER), "--model", "matteiuspi/Qwen3.8-Flash-Next-W8A8-DYNAMIC-300i", "--dry-run"],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 2
    assert "local checkpoint" in result.stderr
