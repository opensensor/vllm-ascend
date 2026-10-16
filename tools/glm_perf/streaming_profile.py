# SPDX-License-Identifier: Apache-2.0
"""Separate repeat stage timings from one-pass native PipeUtilization traces."""

from __future__ import annotations

import argparse
import csv
import gc
import importlib
import json
import statistics
import subprocess
import sys
import time
from functools import partial
from pathlib import Path

from tools.glm_perf.analyze_trace import kernel_category
from tools.glm_perf.build_streaming import admit_candidate
from tools.glm_perf.streaming_probe import (
    MAX_TOKENS,
    TOP_K,
    _bank,
    _clear,
    _same,
    _verify_reference_build,
    memory_envelope,
    route_ids,
    select_cases,
)
from tools.glm_perf.streaming_protocol import file_digest

PIPE_FIELDS = ("vec_time(us)", "scalar_time(us)", "mac_time(us)", "mte1_time(us)", "mte2_time(us)", "mte3_time(us)")
PROFILE_ROWS = (5, 17, 31)
PROFILE_BITS = (4, 3)


def profile_cases(names=()):
    if not names:
        names = [f"w{bits}_a4_k4096_rows{rows}_balanced" for bits in PROFILE_BITS for rows in PROFILE_ROWS]
    selected = select_cases(names)
    if any(
        case["bits"] not in PROFILE_BITS
        or case["activation"] != 4
        or case["k"] != 4096
        or case["rows"] not in PROFILE_ROWS
        or case["pattern"] != "balanced"
        for case in selected
    ):
        raise ValueError("native profile supports balanced W3/W4 A4 K4096 rows5/17/31 fixtures")
    return selected


def summarize_pipe_csv(path: Path) -> dict:
    """Preserve overlapping per-pipe attribution; never add it into latency."""
    stages = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            category = kernel_category(row.get("Name", ""), row.get("Accelerator Core", ""))
            if category not in ("moe_gate_up", "moe_down", "moe_prepare", "moe_reduce"):
                continue
            duration = float(row["Duration(us)"].strip())
            if duration <= 0:
                raise ValueError("nonpositive native profiler task duration")
            stage = stages.setdefault(
                category,
                {
                    "tasks": [],
                    "task_count": 0,
                    "task_duration_ms": 0.0,
                    "pipe_attribution_ms": {name: 0.0 for name in PIPE_FIELDS},
                },
            )
            task = {"name": row["Name"], "duration_ms": duration / 1000}
            for field in PIPE_FIELDS:
                value = row.get(field, "")
                if value.strip() in ("", "N/A"):
                    if category in ("moe_gate_up", "moe_down") and field in PIPE_FIELDS[:3]:
                        raise ValueError(f"native profile missing required {field}")
                    task[field] = None
                else:
                    numeric = float(value.strip())
                    if numeric < 0:
                        raise ValueError("negative native pipe metric")
                    task[field] = numeric / 1000
                    stage["pipe_attribution_ms"][field] += numeric / 1000
            aicore = row.get("aicore_time(us)", "")
            task["aicore_time_ms"] = float(aicore) / 1000 if aicore.strip() not in ("", "N/A") else None
            stage["tasks"].append(task)
            stage["task_count"] += 1
            stage["task_duration_ms"] += duration / 1000
    if not all(name in stages for name in ("moe_gate_up", "moe_down")):
        raise ValueError("native profile is incomplete: missing gate/up or down task")
    return {
        "stages": stages,
        "semantics": "per-pipe task attribution overlaps; sums are not independent recoverable latency",
    }


def stage_comparison(candidate: dict, reference: dict) -> dict:
    if set(candidate) != set(reference) or not candidate:
        raise ValueError("candidate/reference event stage coverage differs")
    result = {}
    for name in candidate:
        c, r = candidate[name], reference[name]
        if len(c["samples_ms"]) < 3 or len(r["samples_ms"]) < 3:
            raise ValueError("stage comparison requires at least three unprofiled event samples")
        c_ms, r_ms = statistics.median(c["samples_ms"]), statistics.median(r["samples_ms"])
        if c_ms <= 0 or r_ms <= 0:
            raise ValueError("native stage medians must be positive")
        result[name] = {
            "candidate_median_ms": c_ms,
            "reference_median_ms": r_ms,
            "candidate_over_reference": c_ms / r_ms,
        }
    return result


def profile_once(pipeline, directory: Path, torch, torch_npu):
    """Installed torch_npu API; no timing wrapper is active during this trace."""
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError("native profile trace directory is append-only")
    directory.mkdir(parents=True, exist_ok=True)
    profiler = torch_npu.profiler
    config = profiler._ExperimentalConfig(
        profiler_level=profiler.ProfilerLevel.Level1,
        aic_metrics=profiler.AiCMetrics.PipeUtilization,
        export_type=profiler.ExportType.Text,
        data_simplification=False,
    )
    handler = profiler.tensorboard_trace_handler(str(directory), analyse_flag=True, async_mode=False)
    torch.npu.synchronize()
    start = time.perf_counter()
    with profiler.profile(
        activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.NPU],
        schedule=profiler.schedule(wait=0, warmup=0, active=1, repeat=1),
        on_trace_ready=handler,
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
        experimental_config=config,
    ) as session:
        result = pipeline()
        torch.npu.synchronize()
        session.step()
    elapsed = time.perf_counter() - start
    tables = list(directory.rglob("kernel_details.csv"))
    if len(tables) != 1:
        raise ValueError("profiler did not produce exactly one complete kernel_details.csv")
    return result, {
        "wall_seconds_with_profiler_and_export": elapsed,
        "pipe_summary": summarize_pipe_csv(tables[0]),
        "kernel_csv": str(tables[0]),
        "kernel_csv_sha256": file_digest(tables[0]),
        "timing_semantics": "profiled one-pass attribution; excludes unprofiled event medians",
    }


def worker(args, identity, reference, memory, report):
    import torch
    import torch_npu

    from tools.glm_perf.fused_moe_profile import frozen_helper, measure

    torch.set_num_threads(4)
    torch.npu.set_device(args.device)
    torch_npu.npu.set_compile_mode(jit_compile=False)
    helper_c, options_c = frozen_helper(args.build_dir)
    helper_r, options_r = frozen_helper(args.reference_build_dir)
    # Hash the installed primary API source which supplied this profiler.
    api_modules = (
        importlib.import_module("torch_npu.profiler.profiler"),
        importlib.import_module("torch_npu.profiler.experimental_config"),
    )
    report["profiler_api"] = [
        {"path": module.__file__, "sha256": file_digest(Path(module.__file__))} for module in api_modules
    ]
    report["records"] = []
    selected = profile_cases(args.case_id)
    try:
        for bits in PROFILE_BITS:
            selected_bits = [case for case in selected if case["bits"] == bits]
            if not selected_bits:
                continue
            cpu_bank, bank_provenance = _bank(args.checkpoint, reference, bits, 4096)
            bank = tuple(value.npu() for value in cpu_bank)
            for case in selected_bits:
                candidate = helper_c.NativeFusedMoE(
                    args.build_dir,
                    namespace=options_c["namespace"],
                    activation_bits=4,
                    prepared_weight_layout=True,
                    fp16_route_workspace=True,
                    shared_prefill_max_tokens=MAX_TOKENS,
                )
                baseline = helper_r.NativeFusedMoE(
                    args.reference_build_dir,
                    namespace=options_r["namespace"],
                    activation_bits=4,
                    prepared_weight_layout=True,
                    fp16_route_workspace=True,
                    shared_prefill_max_tokens=MAX_TOKENS,
                )
                generator = torch.Generator().manual_seed(8800 + bits + case["tokens"])
                x_cpu = torch.randn(case["tokens"], 4096, generator=generator).half() * 0.1
                x = x_cpu.npu()
                weights = torch.full((case["tokens"], TOP_K), 0.125, dtype=torch.float32).npu()
                ids = torch.tensor(route_ids(case), dtype=torch.int64).npu()
                tensors = (x, *bank, weights, ids)
                pipe_c, pipe_r = partial(candidate, *tensors), partial(baseline, *tensors)
                _same(torch, pipe_c(), pipe_r())
                # Separate bounded event medians from all profiler traces.
                stages_c = measure(candidate, pipe_c, args.warmups, args.samples)
                stages_r = measure(baseline, pipe_r, args.warmups, args.samples)
                record = {
                    "case": case,
                    "checkpoint": bank_provenance,
                    "same_input_bitwise": True,
                    "unprofiled_stage_ms": {"candidate": stages_c, "reference": stages_r},
                    "unprofiled_stage_comparison": stage_comparison(stages_c, stages_r),
                    "profiles": {},
                }
                case_root = args.output.parent / (args.output.stem + "-traces") / case["id"]
                actual_c, prof_c = profile_once(pipe_c, case_root / "candidate", torch, torch_npu)
                actual_r, prof_r = profile_once(pipe_r, case_root / "reference", torch, torch_npu)
                _same(torch, actual_c, actual_r)
                record["profiles"] = {"candidate": prof_c, "reference": prof_r}
                record["peak_allocated_bytes"] = torch.npu.max_memory_allocated()
                report["records"].append(record)
                args.output.write_text(json.dumps(report, indent=2) + "\n")
                print(
                    json.dumps({"case": case["id"], "stage_comparison": record["unprofiled_stage_comparison"]}),
                    flush=True,
                )
                del actual_c, actual_r, pipe_c, pipe_r, x, weights, ids, tensors
                _clear(candidate, torch)
                _clear(baseline, torch)
                del candidate, baseline
            del bank, cpu_bank
            gc.collect()
            torch.npu.empty_cache()
        report["state"] = "profiled"
    except Exception as error:
        report["state"] = "failed"
        report["errors"].append(f"{type(error).__name__}: {error}")
        raise
    finally:
        args.output.write_text(json.dumps(report, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("build-dir", "reference", "reference-build-dir", "checkpoint", "memory-evidence", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--case-id", action="append", default=[])
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--watchdog-seconds", type=int, default=900)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.warmups < 1 or args.samples < 3 or args.watchdog_seconds <= 0:
        parser.error("positive warmups/watchdog and at least three event samples required")
    selected = profile_cases(args.case_id)
    if args.output.exists() and not args.worker:
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    identity = admit_candidate(args.build_dir, live=False)
    reference = json.loads(args.reference.read_text())
    if reference["reference_sha256"] != identity["reference_sha256"]:
        raise ValueError("frozen v1021 reference mismatch")
    _verify_reference_build(args.reference_build_dir, reference)
    evidence = json.loads(args.memory_evidence.read_text())
    for item in evidence.get("raw_provenance", []):
        path = Path(item["path"])
        if not path.is_absolute():
            path = args.memory_evidence.parent / path
        if file_digest(path) != item["sha256"]:
            raise ValueError("memory measurement provenance changed")
    memory = memory_envelope(evidence)
    report = {
        "schema_version": 1,
        "kind": "native_profile_diagnostic",
        "state": "pending",
        "identity": identity,
        "reference_binary_hashes": reference["binaries"],
        "memory_envelope": memory,
        "memory_evidence_sha256": file_digest(args.memory_evidence),
        "cases": [case["id"] for case in selected],
        "errors": [],
        "records": [],
        "scope": "standalone same-input 72-expert stage diagnostics; no full-model performance or native admission",
        "timing_semantics": "unprofiled event medians separate from profiled one-pass pipe counters",
        "admissible_native_qualification": False,
    }
    if not memory["passed"]:
        report["state"] = "blocked"
        report["errors"].append("memory admission failed before torch_npu import/device initialization")
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        return
    if args.worker:
        worker(args, identity, reference, memory, report)
        return
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    try:
        subprocess.run(
            [sys.executable, "-m", "tools.glm_perf.streaming_profile", *sys.argv[1:], "--worker"],
            check=True,
            timeout=args.watchdog_seconds,
        )
        report = json.loads(args.output.read_text())
        if report["state"] != "profiled" or len(report["records"]) != len(selected):
            raise RuntimeError("native diagnostic profile did not complete selected cases")
        report["watchdog_passed"] = True
    except Exception as error:
        report = json.loads(args.output.read_text())
        report["state"] = "failed"
        report["errors"].append(f"supervisor: {type(error).__name__}: {error}")
        raise
    finally:
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
