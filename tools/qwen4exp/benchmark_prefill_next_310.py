# SPDX-License-Identifier: Apache-2.0
"""Explicit idle-device gates for paged QSA, fused WY and local route operands.

Run only in a later authorized NPU session with external thermal supervision.
Does not contact, change, start or stop a serving engine. Component gates do
not qualify real-weight generation, throughput or sustained thermal behavior.
"""

import argparse
import hashlib
import json
from functools import partial
from pathlib import Path

import torch
import torch.nn.functional as F

from tools.qwen4exp.native_prefill import NativeLocalRouteGather, NativeLocalSwigluPack, NativeWY


def load_resources(build: Path, load_library=True):
    provenance = json.loads((build / "provenance.json").read_text())
    for entry in [provenance["bridge"], *provenance["binaries"].values()]:
        if hashlib.sha256(Path(entry["path"]).read_bytes()).hexdigest() != entry["sha256"]:
            raise ValueError("native binary digest mismatch")
    if load_library:
        torch.ops.load_library(provenance["bridge"]["path"])
    namespace = provenance["namespace"]
    kernel = getattr(torch.classes, namespace).Kernel
    launch = getattr(torch.ops, namespace).launch
    resources = {}
    for name, cls, symbol in (
        ("wy", NativeWY, "qwen_fused_wy_v1"),
        ("route_gather", NativeLocalRouteGather, "qwen_local_route_gather_v1"),
        ("local_swiglu", NativeLocalSwigluPack, "qwen_local_swiglu_pack_v1"),
    ):
        filename = {"wy": "native_wy", "route_gather": "native_route_gather", "local_swiglu": "native_local_swiglu"}[
            name
        ]
        resources[name] = cls(kernel(provenance["binaries"][filename]["path"], symbol), launch)
    return resources


def error_metrics(actual, expected):
    a, b = actual.detach().float().cpu(), expected.detach().float().cpu()
    torch.testing.assert_close(a, b, rtol=0.02, atol=0.003)
    cosine = F.cosine_similarity(a.double().flatten(), b.double().flatten(), dim=0).item()
    if cosine < 0.9999:
        raise ValueError(f"component cosine below gate: {cosine}")
    return {"max_absolute_error": (a - b).abs().max().item(), "cosine": cosine}


def wy_gate(resources, tokens=128, timing=None):
    from vllm_ascend._310p.ops.fla.chunk_gated_delta_rule import (
        _compute_kernel_inputs_from_torch_wy,
        chunk_gated_delta_rule_310,
    )

    results = []
    generator = torch.Generator().manual_seed(813)
    for key_heads, value_heads in ((4, 12), (16, 48)):
        q = F.normalize(torch.randn(1, tokens, key_heads, 128, generator=generator), dim=-1).half().npu()
        k = F.normalize(torch.randn(1, tokens, key_heads, 128, generator=generator), dim=-1).half().npu()
        v = torch.randn(1, tokens, value_heads, 128, generator=generator).half().npu()
        g = F.logsigmoid(torch.randn(1, tokens, value_heads, generator=generator)).npu()
        beta = torch.rand(1, tokens, value_heads, generator=generator).half().npu()
        expected = _compute_kernel_inputs_from_torch_wy(q, k, v, g, beta, 64)
        actual = resources["wy"](q, k, v, g, beta, 64)
        errors = [error_metrics(a, b) for a, b in zip(actual, expected)]
        # Test the complete existing H/O kernels and FP32 recurrent state,
        # rather than declaring the operator correct from U/W cosine alone.
        state = (torch.randn(1, value_heads, 128, 128, generator=generator) * 0.01).npu()
        options = dict(initial_state=state, output_final_state=True)
        reference_out, reference_state = chunk_gated_delta_rule_310(q, k, v, g, beta, **options)
        candidate_out, candidate_state = chunk_gated_delta_rule_310(
            q, k, v, g, beta, wy_prepare=resources["wy"], **options
        )
        if candidate_state.dtype != torch.float32:
            raise ValueError("recurrent state precision changed")
        times = {}
        if timing is not None:
            from tools.qwen4exp.benchmark_w4_projection_310 import timed_ms

            times = {
                "reference_wy_ms": timed_ms(
                    partial(_compute_kernel_inputs_from_torch_wy, q, k, v, g, beta, 64), *timing
                ),
                "candidate_wy_ms": timed_ms(partial(resources["wy"], q, k, v, g, beta, 64), *timing),
            }
        results.append(
            {
                "key_heads": key_heads,
                "value_heads": value_heads,
                "tokens": tokens,
                **times,
                "kernel_inputs": errors,
                "output": error_metrics(candidate_out, reference_out),
                "state": error_metrics(candidate_state, reference_state),
            }
        )
    return {"passed": True, "cases": results, "real_model_quality_validated": False}


def routes_gate(resources, tokens=128, timing=None):
    from vllm_ascend.models.qwen4_exp.w4a8_int4 import pack_activation_device, swiglu_pack_activation_device

    generator = torch.Generator().manual_seed(914)
    inputs = torch.randn(tokens, 2560, generator=generator).half().npu()
    prepared = pack_activation_device(inputs)
    route_rows = torch.arange(tokens * 10, dtype=torch.int32).remainder(tokens).npu()
    projected = torch.randn(tokens * 10, 2560, generator=generator).half().npu()
    full_swiglu = swiglu_pack_activation_device(projected)
    cases = []
    for active in (0, 1, tokens * 10 // 4, tokens * 10):
        ends = torch.tensor([active // 2, active], dtype=torch.int64, device=inputs.device)
        gathered = resources["route_gather"](prepared, route_rows, ends)
        packed = resources["local_swiglu"](projected, ends)
        for actual, operand in zip(gathered, prepared):
            expected = operand.index_select(0, route_rows[:active].long())
            torch.testing.assert_close(actual[:active].cpu(), expected.cpu(), rtol=0, atol=0)
        for actual, expected in zip(packed, full_swiglu):
            torch.testing.assert_close(actual[:active].cpu(), expected[:active].cpu(), rtol=0, atol=0)
        times = {}
        if timing is not None:
            from tools.qwen4exp.benchmark_w4_projection_310 import timed_ms

            times = {
                "reference_gather_ms": timed_ms(
                    lambda: tuple(t.index_select(0, route_rows.long()) for t in prepared), *timing
                ),
                "candidate_gather_ms": timed_ms(
                    partial(resources["route_gather"], prepared, route_rows, ends), *timing
                ),
                "reference_swiglu_ms": timed_ms(lambda: swiglu_pack_activation_device(projected), *timing),
                "candidate_swiglu_ms": timed_ms(partial(resources["local_swiglu"], projected, ends), *timing),
            }
        cases.append({"active_rows": active, "capacity_rows": tokens * 10, "packed_prefix_exact": True, **times})
    return {"passed": True, "cases": cases, "real_weight_moe_validated": False}


def qsa_inputs(tokens, context):
    from vllm_ascend.models.qwen4_exp.ops.qsa_indexer import QSAGroupSelection

    generator = torch.Generator().manual_seed(1015)
    pages = (context + 127) // 128
    query = torch.randn(tokens, 6, 256, generator=generator).half().npu()
    key = torch.randn(pages + 3, 16, 128, 16, generator=generator).half().npu()
    value = torch.randn_like(key)
    table = torch.randperm(pages + 3, generator=generator)[:pages].int()[None].npu()
    # Retain the native operator's group order; each row has independent sparse
    # selections, followed by a tail outside the compressed groups.
    groups = (
        torch.stack([torch.randperm((context - 4) // 4, generator=generator)[:512] for _ in range(tokens)]).int().npu()
    )
    counts = torch.full((tokens,), groups.shape[1], dtype=torch.int32, device=query.device)
    tails = torch.full_like(counts, context - 4)
    tail_counts = torch.arange(tokens, dtype=torch.int32, device=query.device).remainder(4)
    selection = QSAGroupSelection(groups, counts, tails, tail_counts)
    bounds = torch.tensor([0, tokens], dtype=torch.int32, device=query.device)
    return query, key, value, selection, table, bounds


def qsa_gate(tokens=128, context=8192):
    from vllm_ascend.models.qwen4_exp.ops.qsa_batched_attention_310 import qsa_batched_prefill_310
    from vllm_ascend.models.qwen4_exp.ops.qsa_sparse_attention_310 import qsa_sparse_attention_310

    inputs = qsa_inputs(tokens, context)
    native = qsa_sparse_attention_310(*inputs)
    gathered = qsa_batched_prefill_310(*inputs, query_lens=[tokens])
    return {
        "passed": True,
        "errors": error_metrics(native, gathered),
        "tokens": tokens,
        "context_tokens": context,
        "selected_kv_global_scratch": False,
        "real_model_quality_validated": False,
    }


def validate_resources(resources):
    return {"passed": True, "wy": wy_gate(resources), "routes": routes_gate(resources)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("qsa", "wy", "routes"), required=True)
    parser.add_argument("--native-build", type=Path)
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--context-tokens", type=int, default=8192)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.tokens <= 0 or args.tokens % 64 or args.tokens > 2560:
        parser.error("choose a new output and a positive 64-aligned token count up to 2560")
    if args.iterations <= 0 or args.repeats <= 0 or args.context_tokens < 2052:
        parser.error("positive repetitions and context >=2052 required")
    if args.case != "qsa" and args.native_build is None:
        parser.error("native-build required for WY/routes")
    # Deferred until an explicit invocation. --help and offline imports do not
    # initialize hardware packages or execute requests.
    import torch_npu

    from tools.qwen4exp.benchmark_w4_projection_310 import timed_ms
    from vllm_ascend.utils import enable_custom_op

    torch.npu.set_device(0)
    if "310" not in torch_npu.npu.get_device_name(0):
        raise RuntimeError("requires Ascend 310P")
    torch_npu.npu.set_compile_mode(jit_compile=False)
    enable_custom_op()
    with torch.inference_mode():
        resources = load_resources(args.native_build) if args.native_build else None
        if args.case == "qsa":
            receipt = qsa_gate(args.tokens, args.context_tokens)
            from vllm_ascend.models.qwen4_exp.ops.qsa_batched_attention_310 import qsa_batched_prefill_310
            from vllm_ascend.models.qwen4_exp.ops.qsa_sparse_attention_310 import qsa_sparse_attention_310

            inputs = qsa_inputs(args.tokens, args.context_tokens)
            baseline = lambda: qsa_batched_prefill_310(*inputs, query_lens=[args.tokens])
            candidate = lambda: qsa_sparse_attention_310(*inputs)
        elif args.case == "wy":
            receipt = wy_gate(resources, args.tokens, (args.iterations, args.repeats))
            # D2H assertions and H/O correctness checks stay outside timing.
            baseline = candidate = None
        else:
            receipt = routes_gate(resources, args.tokens, (args.iterations, args.repeats))
            baseline = candidate = None
        if baseline is not None:
            receipt["baseline_ms"] = timed_ms(baseline, args.iterations, args.repeats)
            receipt["candidate_ms"] = timed_ms(candidate, args.iterations, args.repeats)
        receipt.update(
            case=args.case, hardware_component_gate=True, sustained_thermal_validated=False, server_modified=False
        )
        args.output.write_text(json.dumps(receipt, indent=2) + "\n")
        print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
