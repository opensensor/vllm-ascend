# SPDX-License-Identifier: Apache-2.0
"""Standalone fusion parity and changing-input graph gates before resident load."""

import argparse
import importlib.util
import json
from pathlib import Path

import torch
import torch_npu  # noqa: F401


def prepare():
    root = Path(__file__).resolve().parent
    spec = importlib.util.spec_from_file_location("glm_resident_fusions_v1", root / "native.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Fusions(root)


@torch.inference_mode()
def validate(operation, full=False):
    records = []
    generator = torch.Generator().manual_seed(310206)
    device = f"npu:{torch.npu.current_device()}"
    for tokens in (1, 2, 4, 8, 9, 63, 640) if full else (1, 2, 4, 8):
        width, hidden, top_k = 2048, 4096, 8
        gate_up = (torch.randn(tokens * top_k, 2 * width, generator=generator) * 3).half().to(device)
        gate, up = gate_up.chunk(2, -1)
        expected_swiglu = (torch.nn.functional.silu(gate.float()) * up.float()).half()
        actual_swiglu = operation.swiglu(gate_up)
        torch.testing.assert_close(actual_swiglu.cpu(), expected_swiglu.cpu(), rtol=1e-3, atol=2e-6)
        record = {
            "tokens": tokens,
            "swiglu_fp16_bit_mismatches": int(
                (actual_swiglu.cpu().view(torch.int16) != expected_swiglu.cpu().view(torch.int16)).sum()
            ),
        }

        x = (torch.randn(tokens, hidden, generator=generator) * 0.3).half().to(device)
        residual = torch.randn(tokens, 4, hidden, generator=generator).half().float().to(device)
        post = torch.rand(tokens, 4, 1, generator=generator).half().float().to(device)
        comb = torch.rand(tokens, 4, 4, generator=generator).half().float().to(device)
        # Same upstream einsum and FP16 round trip as the serving post path.
        expected_post = (
            (torch.einsum("...ij,...ih->...jh", comb, residual) + post * x.float().unsqueeze(-2)).half().float()
        )
        actual_post = operation.mhc_post(x, residual, post, comb)
        torch.testing.assert_close(actual_post.cpu(), expected_post.cpu(), rtol=1e-3, atol=2e-6)
        record["mhc_post_max_abs"] = float((actual_post - expected_post).abs().max().cpu())
        record["mhc_post_fp32_bit_mismatches"] = int(
            (actual_post.cpu().view(torch.int32) != expected_post.cpu().view(torch.int32)).sum()
        )

        rows = tokens * top_k
        inverse_cpu = torch.randperm(rows, generator=generator)
        weights_cpu = torch.rand(tokens, top_k, generator=generator)
        for fraction in (0, 0.25, 1):
            live = int(rows * fraction)
            routed_cpu = torch.randn(rows, hidden, generator=generator).half()
            routed_cpu[live:] = float("nan")
            weights_cpu[:, -1] = 0
            ordered = routed_cpu[inverse_cpu].reshape(tokens, top_k, hidden).double()
            active = (inverse_cpu.reshape(tokens, top_k) < live) & (weights_cpu != 0)
            products = torch.where(active.unsqueeze(-1), ordered, 0) * weights_cpu.double().unsqueeze(-1)
            expected = products.sum(1)
            ends = torch.tensor([0, live // 2, live // 2, live], dtype=torch.int64, device=device)
            actual = operation.combine(
                routed_cpu.to(device), inverse_cpu.to(device), weights_cpu.to(device), ends
            ).cpu()
            bound = 2 * top_k * torch.finfo(torch.float32).eps * products.abs().sum(1) + 1e-7
            assert torch.isfinite(actual).all() and torch.all((actual.double() - expected).abs() <= bound)
        records.append(record)

        if tokens <= 8:
            graph = torch.npu.NPUGraph()
            routed = torch.ones(rows, hidden, dtype=torch.float16, device=device)
            inverse = torch.arange(rows, dtype=torch.int64, device=device)
            weights = torch.full((tokens, top_k), 0.125, device=device)
            ends = torch.tensor([0, rows], dtype=torch.int64, device=device)
            # Populate configs and lazy runtime resources before capture.
            operation.combine(routed, inverse, weights, ends)
            torch.npu.synchronize()
            with torch.npu.graph(graph):
                captured_swiglu = operation.swiglu(gate_up)
                captured_post = operation.mhc_post(x, residual, post, comb)
                captured_combine = operation.combine(routed, inverse, weights, ends)
            for value in (-3.0, 0.0, 2.0):
                gate_up.fill_(value)
                x.fill_(value)
                residual.fill_(value)
                post.fill_(0.25)
                comb.fill_(0.125)
                routed.fill_(value)
                ends.fill_(rows if value else 0)
                graph.replay()
                g, u = gate_up.chunk(2, -1)
                reference = (torch.nn.functional.silu(g.float()) * u.float()).half()
                torch.testing.assert_close(captured_swiglu.cpu(), reference.cpu(), rtol=1e-3, atol=2e-6)
                reference = (
                    (torch.einsum("...ij,...ih->...jh", comb, residual) + post * x.float().unsqueeze(-2)).half().float()
                )
                torch.testing.assert_close(captured_post.cpu(), reference.cpu(), rtol=1e-3, atol=2e-6)
                torch.testing.assert_close(captured_combine.cpu(), torch.full((tokens, hidden), value), rtol=0, atol=0)
            del graph
        print(json.dumps(record), flush=True)
    return {
        "passed": True,
        "cases": records,
        "graph_shapes": [1, 2, 4, 8],
        "precision": "SwiGLU/mHC post FP16 step tolerance; combine bounded against independent FP64 oracle",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.npu.set_device(args.device)
    torch.npu.set_compile_mode(jit_compile=False)
    torch.ops.load_library("/home/matteius/experiments/glm-eager-fusion-20261006/glm_eager_fusions_v1.so")
    result = validate(prepare(), full=True)
    result["device"] = args.device
    Path(__file__).resolve().with_name(f"qualification-device-{args.device}.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
