# SPDX-License-Identifier: Apache-2.0
"""One-card gate for the opt-in Qwen W4 grouped MoE finalizer.

This measures the epilogue only. It does not launch a model server or change
the qualified runtime configuration.
"""

import argparse
import hashlib
import json
import statistics
import time
from functools import partial
from pathlib import Path

CASES = (3, 12, 512, 2048)
PATTERNS = ("mixed", "all_peer")
TOP_K = 10
HIDDEN = 2560
GLOBAL_EXPERTS = 512
LOCAL_EXPERTS = 128
EXPERT_OFFSET = 128


def elapsed_ms(function, iterations, trials, torch):
    for _ in range(2):
        function()
    torch.npu.synchronize()
    samples = []
    for _ in range(trials):
        start = time.perf_counter()
        for _ in range(iterations):
            function()
        torch.npu.synchronize()
        samples.append((time.perf_counter() - start) * 1000 / iterations)
    return {"median_ms": statistics.median(samples), "trials_ms": samples}


def capture_npu_profile(function, path, torch, torch_npu):
    """Capture three calls after timing, keeping profiler overhead separate."""
    with torch_npu.profiler.profile(
        activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU],
        record_shapes=True,
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(path)),
    ) as profiler:
        for _ in range(3):
            function()
            profiler.step()
        torch.npu.synchronize()


def prepare_case(tokens, pattern, seed, torch, build_dispatch):
    torch.manual_seed(seed)
    ids = torch.randint(GLOBAL_EXPERTS, (tokens, TOP_K), dtype=torch.int64, device="npu")
    if pattern == "all_peer":
        ids.zero_()
    else:
        ids[0, 0] = EXPERT_OFFSET
    weights = torch.rand((tokens, TOP_K), dtype=torch.float32, device="npu")
    weights /= weights.sum(1, keepdim=True)
    dispatch = build_dispatch(
        weights,
        ids,
        num_local_experts=LOCAL_EXPERTS,
        expert_offset=EXPERT_OFFSET,
        weight_dtype=torch.float32,
    )
    sorted_ids = ids.flatten().index_select(0, dispatch.order)
    local = (sorted_ids >= EXPERT_OFFSET) & (sorted_ids < EXPERT_OFFSET + LOCAL_EXPERTS)
    routed = torch.randn((tokens * TOP_K, HIDDEN), dtype=torch.float16, device="npu")
    routed = torch.where(local[:, None], routed, 0)
    return routed, weights, dispatch


def replace_inputs(target, replacement):
    routed, weights, dispatch = target
    new_routed, new_weights, new_dispatch = replacement
    routed.copy_(new_routed)
    weights.copy_(new_weights)
    dispatch.order.copy_(new_dispatch.order)
    dispatch.inverse_order.copy_(new_dispatch.inverse_order)
    dispatch.route_weights.copy_(new_dispatch.route_weights)


def check_parity(reference, candidate, torch, atol, rtol, expect_zero):
    expected = reference().cpu()
    actual = candidate().cpu()
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
    if expect_zero and torch.count_nonzero(actual):
        raise AssertionError("peer-owned routes produced a nonzero partial sum")
    return float((actual - expected).abs().max())


def check_graph_replay(candidate, reference, current, tokens, pattern, torch, build_dispatch, atol, rtol):
    if tokens > 12:
        return "skipped_large_shape"
    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(stream):
        for _ in range(3):
            candidate()
    torch.npu.current_stream().wait_stream(stream)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, stream=stream):
        captured = candidate()
    replacement = prepare_case(tokens, pattern, 4000 + tokens, torch, build_dispatch)
    replace_inputs(current, replacement)
    graph.replay()
    torch.npu.synchronize()
    actual = captured.cpu()
    torch.testing.assert_close(actual, reference().cpu(), atol=atol, rtol=rtol)
    if pattern == "all_peer" and torch.count_nonzero(actual):
        raise AssertionError("graph replay produced a nonzero peer-only partial sum")
    return "passed"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--atol", type=float, default=0.003)
    parser.add_argument("--rtol", type=float, default=0.003)
    parser.add_argument("--graph-replay", action="store_true")
    parser.add_argument("--trace-dir", type=Path, help="new directory for 2,048-token mixed-route NPU traces")
    args = parser.parse_args()
    if args.iterations <= 0 or args.trials <= 0 or args.atol < 0 or args.rtol < 0:
        parser.error("iterations/trials must be positive and tolerances nonnegative")
    if args.dry_run:
        print(
            json.dumps(
                {
                    "cases": CASES,
                    "patterns": PATTERNS,
                    "graph_replay": args.graph_replay,
                    "trace_capture": args.trace_dir is not None,
                    "npu_used": False,
                }
            )
        )
        return
    if args.output is None or args.output.exists() or (args.trace_dir and args.trace_dir.exists()):
        parser.error("--output and --trace-dir must name new paths")

    import torch
    import torch_npu

    from vllm_ascend.models.qwen4_exp.grouped_expert_dispatch import build_grouped_expert_dispatch
    from vllm_ascend.models.qwen4_exp.w4_moe import finalize_grouped_routes

    if not torch_npu.npu.is_available() or "310" not in torch_npu.npu.get_device_name(0):
        raise RuntimeError("requires one Ascend 310P; no gate was run")
    torch.npu.set_device(0)
    torch_npu.npu.set_compile_mode(jit_compile=False)
    source = Path(__file__).resolve().parents[2] / "vllm_ascend/models/qwen4_exp/w4_moe.py"
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    if args.trace_dir:
        args.trace_dir.mkdir(parents=True)

    with args.output.open("x") as output:
        for tokens in CASES:
            for pattern in PATTERNS:
                current = prepare_case(tokens, pattern, 1000 + tokens, torch, build_grouped_expert_dispatch)
                routed, weights, dispatch = current
                reference = partial(finalize_grouped_routes, routed, dispatch, weights, torch.float32, "torch")
                candidate = partial(finalize_grouped_routes, routed, dispatch, weights, torch.float32, "cann_v2")
                max_abs = check_parity(reference, candidate, torch, args.atol, args.rtol, pattern == "all_peer")
                replacement = prepare_case(tokens, pattern, 2000 + tokens, torch, build_grouped_expert_dispatch)
                replace_inputs(current, replacement)
                changed_max_abs = check_parity(reference, candidate, torch, args.atol, args.rtol, pattern == "all_peer")
                reference_time = elapsed_ms(reference, args.iterations, args.trials, torch)
                candidate_time = elapsed_ms(candidate, args.iterations, args.trials, torch)
                replay = (
                    check_graph_replay(
                        candidate,
                        reference,
                        current,
                        tokens,
                        pattern,
                        torch,
                        build_grouped_expert_dispatch,
                        args.atol,
                        args.rtol,
                    )
                    if args.graph_replay
                    else "not_requested"
                )
                trace_roots = None
                if args.trace_dir and tokens == max(CASES) and pattern == "mixed":
                    trace_roots = {
                        "torch": str(args.trace_dir / "torch"),
                        "cann_v2": str(args.trace_dir / "cann_v2"),
                    }
                    capture_npu_profile(reference, Path(trace_roots["torch"]), torch, torch_npu)
                    capture_npu_profile(candidate, Path(trace_roots["cann_v2"]), torch, torch_npu)
                record = {
                    "tokens": tokens,
                    "routes": tokens * TOP_K,
                    "hidden": HIDDEN,
                    "pattern": pattern,
                    "reference_ms": reference_time,
                    "candidate_ms": candidate_time,
                    "max_abs": max_abs,
                    "changed_max_abs": changed_max_abs,
                    "graph_replay": replay,
                    "trace_roots": trace_roots,
                    "source_sha256": source_hash,
                    "model_ttft_measured": False,
                }
                line = json.dumps(record)
                output.write(line + "\n")
                output.flush()
                print(line, flush=True)


if __name__ == "__main__":
    main()
