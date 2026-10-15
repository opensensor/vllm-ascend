# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Changing-route NPU gate for the isolated W8A16 draft dispatch candidate.

Uses one NPU, synthetic tiny banks, and no distributed process group. Timings
are component wall time, not serving throughput. The parent controller owns
hardware admission and supplies the qualified SDK/runtime environment.
"""

import argparse
import json
import statistics
import time
from pathlib import Path
from types import SimpleNamespace

import torch

from tools.qwen4exp.resident_candidates import draft_graph_experts as candidate


class SplitCapture:
    """Minimal ACL graph/eager sequencing for a single-device component gate."""

    def __init__(self):
        self.graphs = []
        self.segments = []
        self._capturing = False
        self.current_graph = None

    def begin(self):
        self.current_graph = torch.npu.NPUGraph()
        self.current_graph.capture_begin()
        self._capturing = True

    def end(self):
        self.current_graph.capture_end()
        self.graphs.append(self.current_graph)
        self.segments.append(self.current_graph.replay)
        self._capturing = False

    def add_eager(self, callback):
        self.end()
        callback()
        self.segments.append(callback)
        self.begin()

    def replay(self):
        for segment in self.segments:
            segment()

    def reset(self):
        for graph in self.graphs:
            graph.reset()
        self.segments.clear()
        self.graphs.clear()


def timed(function, iterations, repeats):
    durations = []
    for _ in range(repeats):
        torch.npu.synchronize()
        started = time.perf_counter()
        for _ in range(iterations):
            function()
        torch.npu.synchronize()
        durations.append((time.perf_counter() - started) * 1000 / iterations)
    return {"median_ms": statistics.median(durations), "samples_ms": durations}


@torch.inference_mode()
def run(args):
    # Worker-local imports must follow the parent's frozen SDK/PYTHONPATH.
    import torch_npu

    from vllm_ascend.models.qwen4_exp.dtype_policy import Qwen4ExpDtypePolicy
    from vllm_ascend.models.qwen4_exp.mtp import _linear, _MTPFP16MoE, route_topk, swiglu_gate_up
    from vllm_ascend.utils import weak_ref_tensor

    torch.npu.set_device(args.device)
    if "310" not in torch_npu.npu.get_device_name(args.device):
        raise RuntimeError("requires Ascend 310P")
    torch.npu.set_compile_mode(jit_compile=False)
    torch.manual_seed(907)
    config = SimpleNamespace(
        num_experts=8,
        num_experts_per_tok=2,
        hidden_size=args.hidden,
        moe_intermediate_size=args.intermediate,
        shared_expert_intermediate_size=args.intermediate,
        ascend_expert_quantization={},
    )
    bank = _MTPFP16MoE(config, Qwen4ExpDtypePolicy(), (0, 1), quantize_experts=True)
    for expert in range(bank.num_local_experts):
        bank.load_expert_weight("gate_up_proj", expert, torch.randn(2 * args.intermediate, args.hidden) * 0.03)
        bank.load_expert_weight("down_proj", expert, torch.randn(args.hidden, args.intermediate) * 0.03)
    bank.gate.normal_(std=0.1)
    bank.shared_gate_up.normal_(std=0.03)
    bank.shared_down.normal_(std=0.03)
    bank.shared_expert_gate.normal_(std=0.03)
    # Simulate a rank owning only the latter half of the logical experts.
    # Both paths use the same local weight arrays; no distributed group exists.
    bank.expert_offset = 4
    bank.num_local_experts = 4
    bank = bank.to(device=f"npu:{args.device}")
    active_capture = [None]
    forward = candidate.make_forward(
        linear=_linear,
        route_topk=route_topk,
        swiglu=swiglu_gate_up,
        current_capture=lambda: active_capture[0],
        weak_ref_tensor=weak_ref_tensor,
        fallback=_MTPFP16MoE.forward,
    )
    records = []
    for tokens in args.tokens:
        inputs = (torch.randn(tokens, args.hidden) * 0.2).half().to(f"npu:{args.device}")
        for _ in range(3):
            bank._forward_eager(inputs)
            forward(bank, inputs)
        torch.npu.synchronize()
        stream = torch.npu.Stream()
        stream.wait_stream(torch.npu.current_stream())
        baseline = SplitCapture()
        proposal = SplitCapture()
        baseline_output = torch.empty_like(inputs)
        with torch.npu.stream(stream):
            baseline.begin()
            baseline.add_eager(lambda output=baseline_output, x=inputs: output.copy_(bank._forward_eager(x)))
            baseline.end()
            active_capture[0] = proposal
            proposal.begin()
            try:
                candidate_output = forward(bank, inputs)
            finally:
                proposal.end()
                active_capture[0] = None
        torch.npu.current_stream().wait_stream(stream)
        torch.npu.synchronize()
        maximum_difference = 0.0
        local_route_counts = []
        for step in range(args.steps):
            current = (torch.randn(tokens, args.hidden) * 0.2).half()
            current[:, 0] = 1
            inputs.copy_(current)
            gate = torch.zeros(8, args.hidden, dtype=torch.float16)
            if step % 3 < 2:
                direction = 1 if step % 3 == 0 else -1
                gate[:4, 0] = direction * 2
                gate[4:, 0] = -direction * 2
            else:
                gate.normal_(std=0.2)
            bank.gate.copy_(gate)
            # Readback is outside timing and verifies the replay actually
            # covers empty-local, full-local and changing mixed routing.
            _, ids = route_topk(_linear(inputs, bank.gate, bank.policy.router_dtype), bank.top_k)
            local_route_counts.append(int(((ids >= 4) & (ids < 8)).sum().cpu()))
            baseline.replay()
            proposal.replay()
            expected, actual = baseline_output.cpu(), candidate_output.cpu()
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            maximum_difference = max(maximum_difference, float((actual.float() - expected.float()).abs().max()))
        assert 0 in local_route_counts and tokens * bank.top_k in local_route_counts
        record = {
            "scope": "synthetic_draft_only_no_tp_not_serving_throughput",
            "tokens": tokens,
            "hidden": args.hidden,
            "intermediate": args.intermediate,
            "changing_steps": args.steps,
            "exact_output": True,
            "max_abs_difference": maximum_difference,
            "local_route_counts": local_route_counts,
            "baseline_graphs": len(baseline.graphs),
            "candidate_graphs": len(proposal.graphs),
            "baseline": timed(baseline.replay, args.iterations, args.repeats),
            "candidate": timed(proposal.replay, args.iterations, args.repeats),
        }
        records.append(record)
        torch.npu.synchronize()
        baseline.reset()
        proposal.reset()
        print(json.dumps(record), flush=True)
    return {"passed": True, "records": records}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--tokens", type=int, nargs="+", default=[1, 3, 6])
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--intermediate", type=int, default=128)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output must be new")
    if not all(0 < n <= candidate.MAX_TOKENS for n in args.tokens) or args.steps < 3:
        parser.error("tokens must be 1..12 and changing steps at least 3")
    if min(args.hidden, args.intermediate, args.iterations, args.repeats) < 1:
        parser.error("dimensions and iteration counts must be positive")
    with args.output.open("x") as output:
        result = run(args)
        json.dump(result, output, indent=2)
        output.write("\n")


if __name__ == "__main__":
    main()
