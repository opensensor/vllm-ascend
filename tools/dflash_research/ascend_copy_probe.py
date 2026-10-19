#!/usr/bin/env python3
"""Reproduce queued DFlash prefix writes with computed broadcast/where sources.

No tensor reads occur between the first prefix write and the final snapshots.
Use --sync-between-writes only as a separate synchronization-control experiment.
"""

import argparse
import hashlib
import json
import time
import traceback
from pathlib import Path

import torch


def run_sequence(device, k, mode, computed, sync):
    def transfer(value):
        result = torch.empty_strided(value.shape, value.stride(), dtype=value.dtype, device=device)
        result.copy_(value)
        return result

    positions = torch.tensor([100, 100, 20, 21, 21, 45, 80, 80, 81], dtype=torch.int32)
    slots = torch.tensor([9, 10, 55, 56, 57, 58, 80, 81, 82], dtype=torch.int64)
    next_ids = torch.tensor([11, 22, 33], dtype=torch.int64)
    starts = torch.tensor([0, 2, 6, 9], dtype=torch.int32)
    lengths = torch.tensor([15, 24, 31], dtype=torch.int32)
    table = torch.arange(96, dtype=torch.int32).reshape(3, 32)[:, :16]
    query_count = 3 * (k + 1)
    expected = {
        "context_positions": torch.full((14,), -777, dtype=torch.int32),
        "context_slots": torch.full((14,), -777, dtype=torch.int32),
        "input_ids": torch.full((query_count + 5,), -777, dtype=torch.int64),
        "query_positions": torch.full((query_count + 5,), -777, dtype=torch.int32),
        "query_slots": torch.full((query_count + 5,), -777, dtype=torch.int32),
        "sample_indices": torch.full((3 * k + 5,), -777, dtype=torch.int32),
    }
    outputs = {name: transfer(value) for name, value in expected.items()}
    expected["context_positions"][:9] = positions
    expected["context_slots"][:9] = slots
    for req in range(3):
        for step in range(k + 1):
            idx = req * (k + 1) + step
            pos = int(lengths[req]) + step
            expected["input_ids"][idx] = int(next_ids[req]) if step == 0 else 999
            expected["query_positions"][idx] = int(positions[int(starts[req + 1]) - 1]) + 1 + step
            expected["query_slots"][idx] = int(table[req, pos // 4]) * 4 + pos % 4
            if step:
                expected["sample_indices"][req * k + step - 1] = idx
    inputs = {
        name: transfer(value)
        for name, value in {
            "positions": positions,
            "slots": slots,
            "next_ids": next_ids,
            "starts": starts,
            "lengths": lengths,
            "table": table,
        }.items()
    }
    active_counts = {
        "context_positions": 9,
        "context_slots": 9,
        "input_ids": query_count,
        "query_positions": query_count,
        "query_slots": query_count,
        "sample_indices": 3 * k,
    }
    prebuilt = {name: transfer(value[: active_counts[name]]) for name, value in expected.items()}
    # Preflight snapshot is before the entire queued sequence, never between ops.
    before = {name: value.cpu().clone() for name, value in outputs.items()}

    def write(name, source):
        target = outputs[name]
        flat = source.flatten()
        if mode == "clone_copy":
            flat = flat.clone()
        elif mode == "contiguous_copy":
            flat = flat.contiguous()
        if mode == "index_copy":
            indices = torch.arange(flat.numel(), dtype=torch.int64, device=device)
            target.index_copy_(0, indices, flat.to(target.dtype))
        else:
            target[: flat.numel()].copy_(flat)
        if sync and str(device).startswith("npu"):
            torch.npu.synchronize()

    write("context_positions", inputs["positions"])
    write("context_slots", inputs["slots"])
    # Same sequence of gathers/broadcasts as expansion, without host observation.
    begin = inputs["starts"][:3].to(torch.int64)
    end = inputs["starts"][1:].to(torch.int64)
    valid = end > begin
    last = torch.where(valid, end - 1, begin)
    first = inputs["positions"][last] + valid.to(torch.int32)
    offsets = torch.arange(k + 1, dtype=torch.int32, device=device)
    query_positions = first[:, None] + offsets
    kv_positions = inputs["lengths"][:, None] + offsets
    columns = torch.div(kv_positions, 4, rounding_mode="floor").to(torch.int64)
    blocks = inputs["table"].gather(1, columns).to(torch.int64)
    query_slots = blocks * 4 + kv_positions.remainder(4)
    input_ids = torch.where(offsets[None, :] == 0, inputs["next_ids"].reshape(3, 1), 999)
    write("input_ids", input_ids if computed else prebuilt["input_ids"])
    write("query_positions", query_positions if computed else prebuilt["query_positions"])
    write("query_slots", query_slots if computed else prebuilt["query_slots"])
    indices = torch.arange(query_count, dtype=torch.int32, device=device).reshape(3, k + 1)[:, 1:]
    write("sample_indices", indices if computed else prebuilt["sample_indices"])
    actual = {name: value.cpu().clone() for name, value in outputs.items()}
    mismatch = [name for name in expected if not torch.equal(expected[name], actual[name])]
    return {
        "passed": not mismatch,
        "mismatched_buffers": mismatch,
        "before": {name: value.tolist() for name, value in before.items()},
        "expected": {name: value.tolist() for name, value in expected.items()},
        "actual": {name: value.tolist() for name, value in actual.items()},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--sync-between-writes", action="store_true")
    args = parser.parse_args()
    result = {
        "completed": False,
        "device": args.device,
        "torch_version": torch.__version__,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "sync_between_writes": args.sync_between_writes,
        "cases": [],
    }
    started = time.perf_counter()
    try:
        if args.device.startswith("npu"):
            import torch_npu

            result["torch_npu_version"] = torch_npu.__version__
            torch.npu.set_device(args.device)
            torch.npu.set_compile_mode(jit_compile=False)
        elif args.device != "cpu":
            raise ValueError("Only cpu and npu accepted")
        for repeat in range(args.repeats):
            for k in (1, 3, 7):
                for computed in (False, True):
                    for mode in (
                        "plain_copy",
                        "clone_copy",
                        "contiguous_copy",
                        "index_copy",
                    ):
                        entry = {
                            "repeat": repeat,
                            "k": k,
                            "computed": computed,
                            "mode": mode,
                        }
                        entry.update(run_sequence(args.device, k, mode, computed, args.sync_between_writes))
                        result["cases"].append(entry)
                        print(
                            json.dumps(
                                {
                                    key: entry[key]
                                    for key in (
                                        "repeat",
                                        "k",
                                        "computed",
                                        "mode",
                                        "passed",
                                    )
                                }
                            ),
                            flush=True,
                        )
        result["completed"] = True
    except Exception:
        result["error"] = traceback.format_exc()
    result["elapsed_s"] = time.perf_counter() - started
    result["passed"] = result["completed"] and all(entry["passed"] for entry in result["cases"])
    result["case_count"] = len(result["cases"])
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
