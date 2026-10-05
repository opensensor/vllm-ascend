# SPDX-License-Identifier: Apache-2.0
"""Host-only checks for the deferred Qwen W4 finalizer profile gates."""

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

from tools.qwen4exp.benchmark_w4_finalize_routing_310 import capture_npu_profile

REPO_ROOT = Path(__file__).resolve().parents[3]


def test_profile_capture_records_three_calls_and_flushes(tmp_path):
    profiler = MagicMock()
    profiler_api = MagicMock()
    profiler_api.ProfilerActivity = SimpleNamespace(CPU="cpu", NPU="npu")
    profiler_api.profile.return_value.__enter__.return_value = profiler
    torch_npu = SimpleNamespace(profiler=profiler_api)
    torch = SimpleNamespace(npu=SimpleNamespace(synchronize=MagicMock()))
    function = MagicMock()

    capture_npu_profile(function, tmp_path / "torch", torch, torch_npu)

    profiler_api.tensorboard_trace_handler.assert_called_once_with(str(tmp_path / "torch"))
    assert function.call_count == profiler.step.call_count == 3
    torch.npu.synchronize.assert_called_once_with()


def test_profile_gate_dry_runs_leave_trace_paths_absent(tmp_path):
    cases = (
        ("benchmark_w4_finalize_routing_310", ["--trace-dir", str(tmp_path / "epilogue")]),
        ("benchmark_w4_finalize_layer_310", ["--trace-dir", str(tmp_path / "layer")]),
        ("benchmark_w4_finalize_service", ["--skip-warmup", "--arm", "torch"]),
    )
    for module, options in cases:
        output = tmp_path / f"{module}.jsonl"
        result = subprocess.run(
            [sys.executable, "-m", f"tools.qwen4exp.{module}", "--dry-run", "--output", str(output), *options],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        report = json.loads(result.stdout)
        assert not output.exists()
        if module == "benchmark_w4_finalize_service":
            assert report["http_used"] is False
            assert report["warmup"] is False
        else:
            assert report["npu_used"] is False
            assert report["trace_capture"] is True
    assert not (tmp_path / "epilogue").exists()
    assert not (tmp_path / "layer").exists()


def test_prefill_swiglu_pack_gate_dry_run_is_host_only(tmp_path):
    output = tmp_path / "prefill-swiglu.jsonl"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "tools.qwen4exp.benchmark_w4_prefill_swiglu_pack_310",
            "--dry-run",
            "--output",
            str(output),
            "--trace-dir",
            str(tmp_path / "prefill-traces"),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(result.stdout) == {
        "rows": [5120, 15360, 20480],
        "width": 640,
        "trace_rows": 15360,
        "trace_capture": True,
        "npu_used": False,
    }
    assert not output.exists()
    assert not (tmp_path / "prefill-traces").exists()
