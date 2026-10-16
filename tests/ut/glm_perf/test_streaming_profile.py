# SPDX-License-Identifier: Apache-2.0
"""Offline profiler API, pipe-counter interpretation and allocation gates."""

import csv
import json
import sys
from types import SimpleNamespace

import pytest

from tools.glm_perf import streaming_profile as profile
from tools.glm_perf.streaming_protocol import file_digest


def write_kernel_csv(path, *, missing=None, value="1000", include_down=True):
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = ["Name", "Duration(us)", "aicore_time(us)", *profile.PIPE_FIELDS]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for name in (
            ["glm_fused_gate_up_w4_v1", "glm_fused_down_w4_v1"] if include_down else ["glm_fused_gate_up_w4_v1"]
        ):
            row = {"Name": name, "Duration(us)": "2000", "aicore_time(us)": "1900"}
            row.update({key: value for key in profile.PIPE_FIELDS})
            if missing:
                row[missing] = "N/A"
            writer.writerow(row)


def test_overlapping_pipe_times_are_reported_separately_from_task_duration(tmp_path):
    source = tmp_path / "kernel_details.csv"
    write_kernel_csv(source)
    result = profile.summarize_pipe_csv(source)
    gate = result["stages"]["moe_gate_up"]
    assert gate["task_count"] == 1
    assert gate["task_duration_ms"] == 2.0
    assert sum(gate["pipe_attribution_ms"].values()) == 6.0
    assert gate["tasks"][0]["aicore_time_ms"] == 1.9
    assert "overlaps" in result["semantics"]


@pytest.mark.parametrize("field", profile.PIPE_FIELDS[:3])
def test_missing_native_vec_scalar_or_mac_counters_is_not_silently_accepted(tmp_path, field):
    source = tmp_path / "kernel_details.csv"
    write_kernel_csv(source, missing=field)
    with pytest.raises(ValueError, match="missing required"):
        profile.summarize_pipe_csv(source)


@pytest.mark.parametrize("value", ["-1", "invalid"])
def test_invalid_pipe_counters_are_rejected(tmp_path, value):
    source = tmp_path / "kernel_details.csv"
    write_kernel_csv(source, value=value)
    with pytest.raises(ValueError):
        profile.summarize_pipe_csv(source)


def test_incomplete_profile_missing_down_is_rejected(tmp_path):
    source = tmp_path / "kernel_details.csv"
    write_kernel_csv(source, include_down=False)
    with pytest.raises(ValueError, match="missing gate/up or down"):
        profile.summarize_pipe_csv(source)


def test_native_profile_scope_has_no_qualification_or_coverage_shortcut():
    assert len(profile.profile_cases()) == 6
    names = ["w4_a4_k4096_rows31_balanced", "w3_a4_k4096_rows5_balanced"]
    assert [case["id"] for case in profile.profile_cases(names)] == names
    for invalid in (
        "w4_a8_k4096_rows31_balanced",
        "w2_a4_k4096_rows31_balanced",
        "w4_a4_k2048_rows31_balanced",
        "w4_a4_k4096_rows31_skewed",
    ):
        with pytest.raises(ValueError, match="native profile supports"):
            profile.profile_cases([invalid])


def test_event_medians_are_recomputed_and_distinct_from_profiler_counter_sums():
    candidate = {"gate": {"samples_ms": [80, 100, 90], "median_ms": 1234}}
    reference = {"gate": {"samples_ms": [50, 60, 40], "median_ms": 1234}}
    result = profile.stage_comparison(candidate, reference)
    assert result["gate"] == {"candidate_median_ms": 90, "reference_median_ms": 50, "candidate_over_reference": 1.8}
    with pytest.raises(ValueError, match="coverage differs"):
        profile.stage_comparison(candidate, {})
    with pytest.raises(ValueError, match="at least three"):
        profile.stage_comparison({"gate": {"samples_ms": [1, 2]}}, reference)
    with pytest.raises(ValueError, match="positive"):
        profile.stage_comparison({"gate": {"samples_ms": [0, 0, 0]}}, reference)


def test_profile_once_uses_installed_level1_pipeutilization_and_synchronous_export(tmp_path):
    seen = {}
    csv_file = tmp_path / "trace" / "ASCEND_PROFILER_OUTPUT" / "kernel_details.csv"

    class Session:
        def __init__(self, **kwargs):
            seen["profile"] = kwargs

        def __enter__(self):
            return self

        def __exit__(self, *args):
            write_kernel_csv(csv_file)

        def step(self):
            seen["step"] = True

    def config(**kwargs):
        seen["config"] = kwargs
        return "config"

    def handler(directory, **kwargs):
        seen["handler"] = (directory, kwargs)
        return "handler"

    profiler = SimpleNamespace(
        _ExperimentalConfig=config,
        ProfilerLevel=SimpleNamespace(Level1=1),
        AiCMetrics=SimpleNamespace(PipeUtilization=2),
        ExportType=SimpleNamespace(Text="text"),
        tensorboard_trace_handler=handler,
        ProfilerActivity=SimpleNamespace(CPU="cpu", NPU="npu"),
        schedule=lambda **kwargs: kwargs,
        profile=Session,
    )
    fake_torch = SimpleNamespace(npu=SimpleNamespace(synchronize=lambda: None))
    result, record = profile.profile_once(lambda: "output", tmp_path, fake_torch, SimpleNamespace(profiler=profiler))
    assert result == "output"
    assert seen["config"] == {
        "profiler_level": 1,
        "aic_metrics": 2,
        "export_type": "text",
        "data_simplification": False,
    }
    assert seen["profile"]["activities"] == ["cpu", "npu"]
    assert seen["handler"] == (str(tmp_path), {"analyse_flag": True, "async_mode": False})
    assert seen["step"]
    assert record["kernel_csv_sha256"] == file_digest(csv_file)
    assert "profiled one-pass" in record["timing_semantics"]


def test_profile_memory_gate_blocks_before_device_initialization_or_worker(tmp_path, monkeypatch):
    raw = tmp_path / "raw-memory.json"
    raw.write_text("measured\n")
    memory = tmp_path / "memory.json"
    memory.write_text(
        json.dumps(
            {
                "free_bytes": 1,
                "resident_allocated_bytes": 10,
                "runtime_context_bytes": 1610612736,
                "safety_bytes": 268435456,
                "raw_provenance": [{"path": raw.name, "sha256": file_digest(raw)}],
            }
        )
    )
    reference = tmp_path / "reference.json"
    reference.write_text(json.dumps({"reference_sha256": "a" * 64, "binaries": {}}))
    output = tmp_path / "profile.json"
    monkeypatch.setattr(profile, "admit_candidate", lambda *a, **kw: {"reference_sha256": "a" * 64})
    monkeypatch.setattr(profile, "_verify_reference_build", lambda *a: None)
    forbidden = lambda *a, **kw: pytest.fail("memory rejection reached execution")
    monkeypatch.setattr(profile, "worker", forbidden)
    monkeypatch.setattr(profile.subprocess, "run", forbidden)
    argv = [
        "profile",
        "--build-dir",
        str(tmp_path),
        "--reference-build-dir",
        str(tmp_path),
        "--reference",
        str(reference),
        "--checkpoint",
        str(tmp_path),
        "--memory-evidence",
        str(memory),
        "--output",
        str(output),
    ]
    monkeypatch.setattr(sys, "argv", argv)
    profile.main()
    report = json.loads(output.read_text())
    assert report["state"] == "blocked"
    assert report["admissible_native_qualification"] is False
    assert "before torch_npu" in report["errors"][0]
