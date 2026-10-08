# SPDX-License-Identifier: Apache-2.0
"""Qualify a separate full KDA package against saved resident-baseline outputs."""

import argparse
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path

import torch
import torch_npu  # noqa: F401

from vllm_ascend.utils import enable_custom_op


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("label", choices=("baseline", "cached", "control"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    torch.set_num_threads(4)
    torch.npu.set_device(0)
    torch.npu.set_compile_mode(jit_compile=False)
    enable_custom_op()
    references = torch.load(root / "full-kda-baseline.pt", weights_only=True) if args.label != "baseline" else {}
    saved, records = {}, []
    configurations = [
        (length, layout, True, -7.0) for length in (1, 16, 63, 64, 65, 134, 640) for layout in ("BSND", "BNSD")
    ]
    configurations += [(640, "BSND", True, raw) for raw in (-3.0, 0.0, 5.0)]
    configurations += [(64, "BSND", False, 0.0), (65, "BSND", False, 0.0)]
    for index, (length, layout, safe, raw_mean) in enumerate(configurations):
        name = f"{length}_{layout}_{safe}_{raw_mean}"
        generator = torch.Generator().manual_seed(6310 + index)
        shape = (1, length, 16, 128)
        q_cpu, k_cpu = [
            torch.nn.functional.normalize(torch.randn(shape, generator=generator), dim=-1).half() for _ in range(2)
        ]
        v_cpu = (torch.randn(shape, generator=generator) * 0.1).half()
        raw_cpu = (
            (raw_mean + torch.randn(shape, generator=generator) * 0.03).float()
            if safe
            else -torch.rand(shape, generator=generator) * 0.005
        )
        beta_cpu = torch.rand(1, length, 16, generator=generator) * 0.8 + 0.1
        cu = [0, 0, 64, length] if length == 134 else [0, length]
        state_cpu = torch.randn(len(cu) - 1, 16, 128, 128, generator=generator) * 0.01
        a_log = (torch.randn(16, generator=generator) * 0.1).npu()
        bias = (torch.randn(16 * 128, generator=generator) * 0.1).npu()
        if layout == "BNSD":
            q_cpu, k_cpu, v_cpu, raw_cpu = [
                tensor.transpose(1, 2).contiguous() for tensor in (q_cpu, k_cpu, v_cpu, raw_cpu)
            ]
            beta_cpu = beta_cpu.transpose(1, 2).contiguous()
        inputs_cpu = [q_cpu, k_cpu, v_cpu, raw_cpu, beta_cpu, state_cpu]
        inputs = [tensor.npu() for tensor in inputs_cpu]

        def execute(inputs=inputs, layout=layout, cu=cu, safe=safe, a_log=a_log, bias=bias):
            q, k, v, raw, beta, state = inputs
            return torch.ops._C_ascend.chunk_kda_fwd(
                q,
                k,
                v,
                raw,
                beta,
                128**-0.5,
                64,
                layout=layout,
                initial_state=state,
                output_final_state=True,
                cu_seqlens=cu,
                safe_gate=safe,
                lower_bound=-5.0,
                use_gate_in_kernel=safe,
                A_log=a_log,
                dt_bias=bias,
                disable_recompute=True,
                return_intermediate_states=True,
                state_v_first=True,
            )

        outputs = execute()
        torch.npu.synchronize()
        actual = [tensor.cpu().contiguous() if tensor is not None else None for tensor in outputs]
        saved[name] = actual
        finite = all(bool(torch.isfinite(tensor).all()) for tensor in actual if tensor is not None)
        input_unchanged = all(torch.equal(device.cpu(), cpu) for device, cpu in zip(inputs, inputs_cpu))
        record = {
            "case": name,
            "finite": finite,
            "inputs_unchanged": input_unchanged,
            "finite_by_output": [
                bool(torch.isfinite(tensor).all()) if tensor is not None else None for tensor in actual
            ],
            "sha256": [
                hashlib.sha256(tensor.view(torch.uint8).numpy().tobytes()).hexdigest() if tensor is not None else None
                for tensor in actual
            ],
        }
        if args.label != "baseline":
            expected = references[name]
            assert len(actual) == len(expected)
            record["bit_mismatches"] = [
                int((a.view(torch.uint8) != b.view(torch.uint8)).sum())
                if a is not None and b is not None
                else int(a is not b)
                for a, b in zip(actual, expected)
            ]
        records.append(record)
        (root / f"full-kda-{args.label}.json").write_text(json.dumps(records, indent=2) + "\n")
        print(json.dumps(record), flush=True)
        if args.label == "baseline":
            torch.save(saved, root / "full-kda-baseline.pt")
        assert input_unchanged, record
        # Production uses safe gating. Record the broader existing operator
        # behavior too, including a nonfinite nonsafe baseline discovered by
        # this sweep; do not describe those cases as finite qualification.
        repeat_outputs = execute()
        torch.npu.synchronize()
        repeated = [tensor.cpu().contiguous() if tensor is not None else None for tensor in repeat_outputs]
        record["repeat_bit_mismatches"] = [
            int((a.view(torch.uint8) != b.view(torch.uint8)).sum())
            if a is not None and b is not None
            else int(a is not b)
            for a, b in zip(actual, repeated)
        ]
        samples = []
        for _ in range(7):
            start = time.perf_counter()
            execute()
            torch.npu.synchronize()
            samples.append((time.perf_counter() - start) * 1000)
        record["samples_ms"] = samples
        record["median_ms"] = statistics.median(samples)
        (root / f"full-kda-{args.label}.json").write_text(json.dumps(records, indent=2) + "\n")
    if args.label == "baseline":
        torch.save(saved, root / "full-kda-baseline.pt")
    passed = all(
        row["finite"] and sum(row.get("bit_mismatches", [])) == 0 and sum(row["repeat_bit_mismatches"]) == 0
        for row in records
        if "_True_" in row["case"]
    )
    print(
        json.dumps(
            {
                "safe_gate_passed": passed,
                "cases": len(records),
                "nonfinite_cases": [row["case"] for row in records if not row["finite"]],
                "label": args.label,
            }
        ),
        flush=True,
    )
    if not passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
