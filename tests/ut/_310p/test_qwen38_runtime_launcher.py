# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
LAUNCHER = REPO_ROOT / "examples" / "start_qwen38_flash_next_w4_310p.sh"


def test_launcher_defaults_to_the_coherency_qualified_runtime_snapshot():
    launcher = LAUNCHER.read_text()

    assert "QWEN38_PLUGIN_ROOT:-/srv/ai/src/qwen38-head-unified-runtime-20261001" in launcher


def test_custom_opp_order_survives_plugin_bootstrap():
    launcher = LAUNCHER.read_text()

    assert "PACKAGED_OPP=${RUNTIME_ROOT}/vllm_ascend/_cann_ops_custom/vendors/custom_transformer" in launcher
    assert (
        'ASCEND_CUSTOM_OPP_PATH="${COHERENT_OPP}:${PACKAGED_OPP}:${RETAINED_OPP}'
        '${ASCEND_CUSTOM_OPP_PATH:+:${ASCEND_CUSTOM_OPP_PATH}}"'
    ) in launcher
    assert (
        'LD_LIBRARY_PATH="${COHERENT_OPP}/op_api/lib:${PACKAGED_OPP}/op_api/lib:'
        '${RETAINED_OPP}/op_api/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"'
    ) in launcher

    coherent_index = launcher.index("${COHERENT_OPP}:${PACKAGED_OPP}:${RETAINED_OPP}")
    bootstrap_comment_index = launcher.index("Plugin bootstrap prepends PACKAGED_OPP")
    assert bootstrap_comment_index < coherent_index


def test_matching_qsa_abi_precedes_retained_fallback():
    launcher = LAUNCHER.read_text()

    packaged_index = launcher.index("${PACKAGED_OPP}:${RETAINED_OPP}")
    qsa_abi_comment_index = launcher.index("QSA host API")
    assert qsa_abi_comment_index < packaged_index


def test_coherent_opp_supports_the_qualified_prefill_chunk():
    launcher = LAUNCHER.read_text()

    assert "qwen38-coherent-opp-20261001-r2" in launcher
    assert "20,480 rows" in launcher
    assert "MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-2048}" in launcher


@pytest.mark.parametrize(
    ("num_spec_tokens", "max_num_seqs", "expected_sizes"),
    [
        (2, 4, [3, 6]),
        (1, 3, [2, 4]),
        (3, 1, [4]),
    ],
)
def test_graph_capture_sizes_keep_interactive_mtp_shapes_exact(num_spec_tokens, max_num_seqs, expected_sizes):
    env = os.environ.copy()
    env.update(NUM_SPEC_TOKENS=str(num_spec_tokens), MAX_NUM_SEQS=str(max_num_seqs))
    result = subprocess.run(
        ["bash", str(LAUNCHER), "--show"],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    argv = shlex.split(result.stdout)
    compilation_config = json.loads(argv[argv.index("--compilation-config") + 1])
    assert compilation_config["cudagraph_capture_sizes"] == expected_sizes
    assert compilation_config["cudagraph_capture_sizes"][-1] == min(max_num_seqs, 2) * (num_spec_tokens + 1)
    assert "VLLM_ASCEND_LOG_REQUEST_TIMINGS=1" in LAUNCHER.read_text()


def test_graph_capture_sizes_cover_c3_c4_for_gpqa():
    result = subprocess.run(
        ["bash", str(LAUNCHER), "--c3-c4-graphs", "--show"],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    argv = shlex.split(result.stdout)
    compilation_config = json.loads(argv[argv.index("--compilation-config") + 1])
    assert compilation_config["cudagraph_capture_sizes"] == [9, 12]


def test_c3_c4_graph_profile_rejects_unqualified_shape():
    env = os.environ.copy()
    env["MAX_NUM_SEQS"] = "3"
    result = subprocess.run(
        ["bash", str(LAUNCHER), "--c3-c4-graphs", "--show"],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 2
    assert "requires NUM_SPEC_TOKENS=2 and MAX_NUM_SEQS=4" in result.stderr


def _write_qwen_runtime(runtime_root: Path, *, extended_formatter: bool) -> None:
    model_dir = runtime_root / "vllm_ascend" / "models" / "qwen4_exp"
    model_dir.mkdir(parents=True)
    if extended_formatter:
        formatter_args = "model, extra_projection_types=()"
    else:
        formatter_args = "model"
    (model_dir / "model.py").write_text(f"def _format_eager_linear_weights_npu({formatter_args}):\n    pass\n")
    (model_dir / "mtp.py").write_text("_format_eager_linear_weights_npu(model, (Predictor, MoE))\n")
    (model_dir / "w4_moe.py").write_text("class DeferredReduceStream:\n    pass\n")
    (model_dir / "w4a8_int4.py").write_text(
        "import torch\n"
        "def pack_native_weight(tensor):\n"
        "    try:\n"
        "        torch.set_num_threads(6)\n"
        "    finally:\n"
        "        torch.set_num_threads(1)\n"
    )


def test_runtime_coherence_check_accepts_matching_formatter(tmp_path):
    _write_qwen_runtime(tmp_path, extended_formatter=True)
    env = os.environ.copy()
    env.update(QWEN38_PLUGIN_ROOT=str(tmp_path), QWEN38_PYTHON_BIN=sys.executable)

    result = subprocess.run(
        ["bash", str(LAUNCHER), "--check-runtime"],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "runtime coherence check passed" in result.stdout


def test_runtime_coherence_check_rejects_stale_formatter(tmp_path):
    _write_qwen_runtime(tmp_path, extended_formatter=False)
    env = os.environ.copy()
    env.update(QWEN38_PLUGIN_ROOT=str(tmp_path), QWEN38_PYTHON_BIN=sys.executable)

    result = subprocess.run(
        ["bash", str(LAUNCHER), "--check-runtime"],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "stale _format_eager_linear_weights_npu signature" in result.stderr


def test_runtime_coherence_check_rejects_missing_w4_symbol(tmp_path):
    _write_qwen_runtime(tmp_path, extended_formatter=True)
    model_path = tmp_path / "vllm_ascend" / "models" / "qwen4_exp" / "model.py"
    model_path.write_text(model_path.read_text() + "from .w4_moe import MissingStream\n")
    env = os.environ.copy()
    env.update(QWEN38_PLUGIN_ROOT=str(tmp_path), QWEN38_PYTHON_BIN=sys.executable)

    result = subprocess.run(
        ["bash", str(LAUNCHER), "--check-runtime"],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "imports ['MissingStream'] missing" in result.stderr


def test_runtime_coherence_check_rejects_serial_weight_packer(tmp_path):
    _write_qwen_runtime(tmp_path, extended_formatter=True)
    w4a8_path = tmp_path / "vllm_ascend" / "models" / "qwen4_exp" / "w4a8_int4.py"
    w4a8_path.write_text("def pack_native_weight(tensor):\n    return tensor\n")
    env = os.environ.copy()
    env.update(QWEN38_PLUGIN_ROOT=str(tmp_path), QWEN38_PYTHON_BIN=sys.executable)

    result = subprocess.run(
        ["bash", str(LAUNCHER), "--check-runtime"],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "lost the qualified parallel native-weight packer" in result.stderr
