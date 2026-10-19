#!/usr/bin/env python3
"""Small eager DFlash/DSpark primitive parity probe; never loads a model or vLLM.

Example: python dflash-device-probe.py --ops /path/to/dflash_ops.py
         --device npu:0 --output /path/to/result.json
Use --device cpu to validate the harness without initializing any accelerator.
"""

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import time
import traceback
from pathlib import Path

import torch


def load_ops(path):
    spec = importlib.util.spec_from_file_location("isolated_dflash_ops", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def move_preserving_stride(value, device):
    if not isinstance(value, torch.Tensor):
        return value
    result = torch.empty_strided(value.shape, value.stride(), dtype=value.dtype, device=device)
    result.copy_(value)
    return result


def tensor_record(value):
    cpu = value.detach().cpu()
    return {
        "shape": list(value.shape),
        "stride": list(value.stride()),
        "dtype": str(value.dtype),
        "values": cpu.tolist(),
    }


def expansion_case(k, rejection, mrope, sample_from_anchor=False):
    batch, block_size, count = 3, 4, 9
    positions = torch.tensor([100, 100, 20, 21, 21, 45, 80, 80, 81], dtype=torch.int32)
    target_positions = positions
    if mrope:
        backing = torch.full((3, 16), -999, dtype=torch.int32)
        target_positions = backing[:, :count]
        target_positions.copy_(torch.stack([positions, positions + 1000, positions + 2000]))
    starts = torch.tensor([0, 2, 6, 9], dtype=torch.int32)
    lengths = torch.tensor([15, 24, 31], dtype=torch.int32)
    slots = torch.tensor([9, 10, 55, 56, 57, 58, 80, 81, 82], dtype=torch.int64)
    table = torch.arange(96, dtype=torch.int32).reshape(3, 32)[:, :16]
    bonus = torch.tensor([11, 22, 33], dtype=torch.int64)
    queries_per_req = k + (0 if sample_from_anchor else 1)
    total_queries = batch * queries_per_req
    outputs = {
        "out_input_ids_ptr": torch.full((total_queries + 5,), -777, dtype=torch.int64),
        "out_context_positions_ptr": torch.full((count + 5,), -777, dtype=torch.int32),
        "out_query_positions_ptr": torch.full((total_queries + 5,), -777, dtype=torch.int32),
        "out_context_slot_mapping_ptr": torch.full((count + 5,), -777, dtype=torch.int32),
        "out_query_slot_mapping_ptr": torch.full((total_queries + 5,), -777, dtype=torch.int32),
        "out_token_indices_ptr": torch.full((batch * k + 5,), -777, dtype=torch.int32),
    }
    arguments = dict(
        next_token_ids_ptr=bonus,
        target_positions_ptr=target_positions,
        context_slot_mapping_ptr=slots,
        block_table_ptr=table,
        block_table_stride=table.stride(0),
        query_start_loc_ptr=starts,
        seq_lens_ptr=lengths,
        num_rejected_tokens_ptr=0 if rejection is None else torch.tensor(rejection, dtype=torch.int32),
        parallel_drafting_token_id=999,
        block_size=block_size,
        num_query_per_req=queries_per_req,
        SAMPLE_FROM_ANCHOR=sample_from_anchor,
        num_speculative_tokens=k,
        total_input_tokens=count,
        batch_size=batch,
        HAS_NUM_REJECTED=rejection is not None,
        **outputs,
    )
    expected = {name: value.clone() for name, value in outputs.items()}
    expected["out_context_positions_ptr"][:count] = positions
    expected["out_context_slot_mapping_ptr"][:count] = slots
    # Scalar reference independent of the implementation's broadcast/gather path.
    for req in range(batch):
        rejected = 0 if rejection is None else rejection[req]
        begin = int(starts[req])
        end = int(starts[req + 1]) - rejected
        first = int(positions[end - 1]) + 1 if end > begin else int(positions[begin])
        for step in range(queries_per_req):
            output_idx = req * queries_per_req + step
            linear = int(lengths[req]) - rejected + step
            expected["out_input_ids_ptr"][output_idx] = int(bonus[req]) if step == 0 else 999
            expected["out_query_positions_ptr"][output_idx] = first + step
            expected["out_query_slot_mapping_ptr"][output_idx] = (
                int(table[req, linear // block_size]) * block_size + linear % block_size
            )
            if sample_from_anchor or step:
                sample_offset = step if sample_from_anchor else step - 1
                expected["out_token_indices_ptr"][req * k + sample_offset] = output_idx
    return arguments, expected


def selector_case(batch, steps, dtype, ties, seed):
    top_k = 16
    generator = torch.Generator().manual_seed(seed)
    candidates = torch.randint(100, 2000, (batch, steps, top_k), generator=generator)
    shape = (batch, steps, top_k, top_k)
    scores = (
        torch.randint(-2, 3, shape, generator=generator).to(dtype)
        if ties
        else torch.randn(shape, generator=generator).to(dtype)
    )
    if ties:
        # Guarantee a smallest-index tie at the anchor, independent of seed.
        scores[:, 0, 0] = -1
        scores[:, 0, 0, 2] = 5
        scores[:, 0, 0, 7] = 5
    expected = torch.empty((batch, steps), dtype=candidates.dtype)
    for req in range(batch):
        previous = 0
        for step in range(steps):
            row = scores[req, step, previous].float().tolist()
            previous = row.index(max(row))
            expected[req, step] = candidates[req, step, previous]
    return candidates, scores, expected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ops", type=Path, required=True)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = {
        "completed": False,
        "passed": False,
        "mode": "eager",
        "device": args.device,
        "pid": os.getpid(),
        "torch_version": torch.__version__,
        "python": platform.python_version(),
        "ops_path": str(args.ops.resolve()),
        "ops_sha256": hashlib.sha256(args.ops.read_bytes()).hexdigest(),
        "probe_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "cases": [],
    }
    start = time.perf_counter()
    try:
        if args.device.startswith("npu"):
            import torch_npu

            result["torch_npu_version"] = torch_npu.__version__
            torch_npu.npu.set_compile_mode(jit_compile=False)
            torch.npu.set_device(args.device)
            result["device_name"] = torch.npu.get_device_name(torch.device(args.device))
        elif args.device != "cpu":
            raise ValueError("Only cpu and npu devices are accepted")
        ops = load_ops(args.ops)

        def run_case(name, metadata, callback):
            record = {"name": name, **metadata}
            case_start = time.perf_counter()
            try:
                record.update(callback())
                record.setdefault("passed", True)
            except Exception:
                record["passed"] = False
                record["error"] = traceback.format_exc()
            record["elapsed_s"] = time.perf_counter() - case_start
            result["cases"].append(record)
            print(json.dumps({"case": name, "passed": record["passed"]}), flush=True)
            # Stop after the first device failure: later kernels may be poisoned.
            if not record["passed"]:
                raise RuntimeError(f"Case failed: {name}")

        for sample_from_anchor in (False, True):
            for k in (1, 3, 7):
                for rejection in (None, [0, 2, 3], [2, 4, 3]):
                    for mrope in (False, True):
                        arguments, expected = expansion_case(k, rejection, mrope, sample_from_anchor)

                        def check_expansion(arguments=arguments, expected=expected):
                            on_device = {
                                key: move_preserving_stride(value, args.device) for key, value in arguments.items()
                            }
                            before = {
                                key: value.detach().cpu().clone()
                                for key, value in on_device.items()
                                if isinstance(value, torch.Tensor)
                            }
                            initial_mismatches = [
                                key
                                for key, value in arguments.items()
                                if isinstance(value, torch.Tensor)
                                and (on_device[key].stride() != value.stride() or not torch.equal(before[key], value))
                            ]
                            details = {
                                "inputs": {
                                    key: tensor_record(value) if isinstance(value, torch.Tensor) else value
                                    for key, value in arguments.items()
                                    if key not in expected
                                },
                                "expected_outputs": {key: tensor_record(value) for key, value in expected.items()},
                                "initial_device_values": {key: tensor_record(value) for key, value in before.items()},
                            }
                            if initial_mismatches:
                                return {
                                    **details,
                                    "passed": False,
                                    "phase": "initial_transfer",
                                    "mismatched_buffers": initial_mismatches,
                                }
                            ops.copy_and_expand_dflash_inputs(**on_device)
                            outputs = {key: on_device[key].detach().cpu().clone() for key in expected}
                            mismatches = [key for key in expected if not torch.equal(outputs[key], expected[key])]
                            return {
                                **details,
                                "passed": not mismatches,
                                "phase": "helper_output",
                                "mismatched_buffers": mismatches,
                                "outputs": {key: tensor_record(value) for key, value in outputs.items()},
                            }

                        run_case(
                            f"expand-anchor{sample_from_anchor}-k{k}-reject{rejection}-mrope{mrope}",
                            {
                                "sample_from_anchor": sample_from_anchor,
                                "draft_tokens": k,
                                "rejections": rejection,
                                "mrope": mrope,
                            },
                            check_expansion,
                        )

        for batch in (1, 3, 17):
            for steps in (1, 3, 7):
                for dtype in (torch.float16, torch.float32):
                    for ties in (False, True):
                        seed = 91 + batch + steps
                        candidates, scores, expected = selector_case(batch, steps, dtype, ties, seed)

                        def check_selector(candidates=candidates, scores=scores, expected=expected):
                            device_candidates = move_preserving_stride(candidates, args.device)
                            device_scores = move_preserving_stride(scores, args.device)
                            initial = {
                                "candidate_ids": device_candidates.cpu().clone(),
                                "scores": device_scores.cpu().clone(),
                            }
                            mismatches = [
                                name
                                for name, reference in (
                                    ("candidate_ids", candidates),
                                    ("scores", scores),
                                )
                                if not torch.equal(initial[name], reference)
                            ]
                            if mismatches:
                                return {
                                    "passed": False,
                                    "phase": "initial_transfer",
                                    "mismatched_buffers": mismatches,
                                    "initial_device_values": {
                                        key: tensor_record(value) for key, value in initial.items()
                                    },
                                }
                            actual = ops.greedy_select_path_310p(device_candidates, device_scores).cpu().clone()
                            return {
                                "passed": torch.equal(actual, expected),
                                "phase": "selector_output",
                                "mismatched_buffers": [] if torch.equal(actual, expected) else ["selector_tokens"],
                                "expected_tokens": expected.tolist(),
                                "actual_tokens": actual.tolist(),
                            }

                        run_case(
                            f"selector-b{batch}-k{steps}-{dtype}-ties{ties}",
                            {
                                "batch": batch,
                                "steps": steps,
                                "top_k": 16,
                                "dtype": str(dtype),
                                "ties": ties,
                                "seed": seed,
                            },
                            check_selector,
                        )
        result["completed"] = True
        result["passed"] = all(case["passed"] for case in result["cases"])
    except Exception:
        result["error"] = traceback.format_exc()
    finally:
        result["elapsed_s"] = time.perf_counter() - start
        result["case_count"] = len(result["cases"])
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(json.dumps(result, indent=2) + "\n")
        temporary.replace(args.output)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
