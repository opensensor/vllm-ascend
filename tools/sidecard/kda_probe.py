# SPDX-License-Identifier: Apache-2.0
"""Exclusive-device qualification probe for the isolated native recurrence.

`--help` does not import torch/torch_npu. This reports operator-only evidence;
even a passing probe cannot qualify complete-model correctness or service speed.
"""

import argparse
import hashlib
import json
import os
import statistics
import time
from dataclasses import asdict
from pathlib import Path

from .npu_kda import BOUNDS, HEAD_DIM, OPERATOR_NAME, OPERATOR_NAMESPACE, native, validate_inputs


def configuration(config):
    if not isinstance(config, dict):
        raise ValueError("probe config must be an object")
    unknown = set(config) - {
        "library",
        "library_sha256",
        "base_library",
        "base_library_sha256",
        "physical_chips",
        "heads",
        "warmup",
        "repeats",
        "sequence",
    }
    if unknown:
        raise ValueError(f"unknown KDA probe options: {sorted(unknown)}")
    if config.get("physical_chips") != [2, 3]:
        raise ValueError("probe may allocate only assigned physical chips [2,3]")
    path = Path(config["library"])
    if not path.is_absolute() or not path.is_file():
        raise ValueError("probe library must be an existing absolute path")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if config.get("library_sha256") != digest:
        raise ValueError("probe library SHA256 mismatch")
    if "base_library" in config or "base_library_sha256" in config:
        base = Path(config.get("base_library", ""))
        if not base.is_absolute() or not base.is_file():
            raise ValueError("probe base library must be an existing absolute path")
        if config.get("base_library_sha256") != hashlib.sha256(base.read_bytes()).hexdigest():
            raise ValueError("probe base library SHA256 mismatch")
    for name, default, lower, upper in (
        ("warmup", 10, 1, 1000),
        ("repeats", 100, 3, 10000),
        ("sequence", 256, 2, 4096),
    ):
        value = config.get(name, default)
        if type(value) is not int or not lower <= value <= upper:
            raise ValueError(f"{name} must be an integer in [{lower},{upper}]")
    heads = config.get("heads", [16, 32])
    if not isinstance(heads, list) or not heads or any(type(h) is not int or not 8 <= h <= 128 or h % 8 for h in heads):
        raise ValueError("heads must be nonempty 8..128 multiples of eight")
    return path, digest, heads


def assigned_device(rank):
    """Resolve the leased physical pair to its process-local device ordinals."""
    if type(rank) is not int or rank not in (0, 1) or os.environ.get("ASCEND_RT_VISIBLE_DEVICES") != "2,3":
        raise ValueError("KDA probe requires leased physical chips 2,3 with local rank 0/1")
    return f"npu:{rank}"


def run(config, output, rank):
    path, digest, head_counts = configuration(config)
    device = assigned_device(rank)
    destination = Path(output)
    if destination.exists():
        raise ValueError("KDA report already exists; retain prior evidence")
    # Lazy process-local imports are required for accelerator isolation.
    import torch
    import torch_npu  # noqa: F401

    from .layer_math import recurrent_kda

    torch.npu.set_device(device)
    if "base_library" in config:
        torch.ops.load_library(config["base_library"])
    torch.ops.load_library(str(path))
    warmup, repeats, sequence = config.get("warmup", 10), config.get("repeats", 100), config.get("sequence", 256)
    report = {
        "rank": rank,
        "physical_chip": config["physical_chips"][rank],
        "operator": OPERATOR_NAME,
        "operator_namespace": OPERATOR_NAMESPACE,
        "library_sha256": digest,
        "base_library_sha256": config.get("base_library_sha256"),
        "bounds": asdict(BOUNDS),
        "measurement_scope": "operator-only synchronized wall time including launch; no server throughput",
        "full_model_qualified": False,
        "cases": [],
    }
    torch.manual_seed(812 + rank)
    for heads in head_counts:

        def inputs(phase, heads=heads):
            scale = (0.0001, 1.0, 8.0, 0.0)[phase % 4]
            q, k, v = [torch.randn(1, heads, HEAD_DIM, device=device).half() * scale for _ in range(3)]
            gate = -torch.rand(1, heads, HEAD_DIM, device=device) * 5
            beta = torch.rand(1, heads, device=device)
            return q, k, v, gate, beta

        original = torch.randn(heads, HEAD_DIM, HEAD_DIM, device=device) * 0.2
        state, expected_state = original.clone(), original.clone()
        first = inputs(0)
        validate_inputs(*first, state, torch)
        torch.npu.synchronize()
        started = time.perf_counter()
        native(*first, state, torch)
        torch.npu.synchronize()
        first_call_ms = (time.perf_counter() - started) * 1000
        state.copy_(original)
        comparisons = []
        for index in range(sequence):
            fields = inputs(index)
            # Independently test every changed input against an identical
            # incoming carry, in addition to cumulative sequence drift.
            one_step_state = expected_state.clone()
            one_step_output = native(*fields, one_step_state, torch)
            actual = native(*fields, state, torch)
            expected = recurrent_kda(*fields, expected_state, torch)
            # CPU transfers/checks are deliberately outside measured loops.
            state_error = float((state - expected_state).abs().max().cpu())
            output_error = float((actual.float() - expected.float()).abs().max().cpu())
            state_atol = BOUNDS.state_atol if index == 0 else BOUNDS.sequential_state_atol
            state_rtol = BOUNDS.state_rtol if index == 0 else BOUNDS.sequential_state_rtol
            comparisons.append(
                {
                    "step": index,
                    "state_max_abs": state_error,
                    "output_max_abs": output_error,
                    "finite": bool(torch.isfinite(state).all().cpu()) and bool(torch.isfinite(actual).all().cpu()),
                    "state_close": torch.allclose(state, expected_state, atol=state_atol, rtol=state_rtol),
                    "one_step_state_close": torch.allclose(
                        one_step_state, expected_state, atol=BOUNDS.state_atol, rtol=BOUNDS.state_rtol
                    ),
                    "one_step_output_close": torch.allclose(
                        one_step_output, expected, atol=BOUNDS.output_atol, rtol=BOUNDS.output_rtol
                    ),
                    "output_close": torch.allclose(actual, expected, atol=BOUNDS.output_atol, rtol=BOUNDS.output_rtol),
                }
            )

        fields = inputs(1)
        timings = {}
        for label, operation in (("native", native), ("eager", recurrent_kda)):
            timed_state = original.clone()
            for _ in range(warmup):
                operation(*fields, timed_state, torch)
            samples = []
            for _ in range(repeats):
                torch.npu.synchronize()
                started = time.perf_counter()
                operation(*fields, timed_state, torch)
                torch.npu.synchronize()
                samples.append((time.perf_counter() - started) * 1000)
            timings[label] = {"median_ms": statistics.median(samples), "samples_ms": samples}
        report["cases"].append(
            {
                "heads": heads,
                "first_call_ms": first_call_ms,
                "sequence": comparisons,
                "steady_state": timings,
                "ok": all(
                    case["finite"]
                    and case["state_close"]
                    and case["output_close"]
                    and case["one_step_state_close"]
                    and case["one_step_output_close"]
                    for case in comparisons
                ),
            }
        )
    report["ok"] = all(case["ok"] for case in report["cases"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rank", type=int, choices=(0, 1), required=True)
    args = parser.parse_args()
    report = run(json.loads(args.config.read_text()), args.output, args.rank)
    if not report["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
