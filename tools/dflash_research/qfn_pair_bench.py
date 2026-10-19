# SPDX-License-Identifier: Apache-2.0
"""Matched eager QFN trials; synchronized profiling is a separate request."""

import argparse
import dataclasses
import json
import time
from pathlib import Path

PROMPTS = {
    "merge": "Write a Python function merge_sorted(a, b) to merge two sorted lists without sort(). Include tests.",
    "debug": (
        "Fix this Python function and explain the bug: def total(xs): "
        "s = 0; for i in range(len(xs) - 1): s += xs[i]; return s"
    ),
    "reasoning": "Explain why a thread-safe queue needs a shutdown protocol. Give a short Python example.",
    "retrieval": (
        "Remember this exact marker: amber-582. "
        + "The unrelated build completed successfully. " * 90
        + "Return only the exact marker from the beginning."
    ),
}


def install_profile(worker):
    """Temporarily time worker stages; synchronization deliberately adds overhead."""
    import time

    import torch

    if hasattr(worker, "_qfn_pair_profile"):
        raise RuntimeError("profile already installed")
    runner = worker.model_runner
    state = {"samples": {}, "originals": []}
    worker._qfn_pair_profile = state

    def wrap(owner, name, label):
        original = getattr(owner, name)
        local = name in vars(owner)

        def timed(*args, **kwargs):
            torch.npu.synchronize()
            start = time.perf_counter()
            try:
                return original(*args, **kwargs)
            finally:
                torch.npu.synchronize()
                state["samples"].setdefault(label, []).append(time.perf_counter() - start)

        state["originals"].append((owner, name, original, local))
        setattr(owner, name, timed)

    wrap(runner, "_model_forward", "target_forward")
    if runner.drafter is not None:
        wrap(runner.drafter, "_propose", "draft_total_inclusive")
        draft = runner.drafter.model
        unwrap = getattr(draft, "unwrap", None)
        if callable(unwrap):
            draft = unwrap()
        wrap(draft, "forward", "draft_forward")
        for name in ("precompute_and_store_context_kv", "compute_draft_logits", "markov_bias"):
            if hasattr(draft, name):
                wrap(draft, name, name)
    return {"rank": worker.rank, "stages": [x[1] for x in state["originals"]]}


def collect_profile(worker):
    state = worker._qfn_pair_profile
    for owner, name, original, local in reversed(state["originals"]):
        if local:
            setattr(owner, name, original)
        else:
            delattr(owner, name)
    del worker._qfn_pair_profile
    return {"rank": worker.rank, "seconds": state["samples"]}


def metrics(llm):
    return [dataclasses.asdict(m) for m in llm.get_metrics() if "spec_decode" in m.name]


def install_oracle(worker, token_ids, prompt_tokens, prefixes):
    """Correctness-only override after real drafting, using a plain-run trace."""
    import torch

    drafter = worker.model_runner.drafter
    if hasattr(worker, "_qfn_pair_oracle"):
        raise RuntimeError("oracle already installed")
    original = drafter._propose
    local = "_propose" in vars(drafter)
    state = {"original": original, "local": local, "calls": [], "target_originals": [], "logits": []}
    worker._qfn_pair_oracle = state

    if worker.rank == 0:
        runner = worker.model_runner
        forward = runner._model_forward
        compute_logits = runner.model.compute_logits
        state["target_originals"] = [
            (runner, "_model_forward", forward, "_model_forward" in vars(runner)),
            (runner.model, "compute_logits", compute_logits, "compute_logits" in vars(runner.model)),
        ]

        def traced_forward(*args, **kwargs):
            positions = kwargs["positions"] if "positions" in kwargs else args[2]
            state["positions"] = (positions[0] if positions.ndim == 2 else positions).cpu().tolist()
            return forward(*args, **kwargs)

        def traced_logits(*args, **kwargs):
            logits = compute_logits(*args, **kwargs)
            positions = state["positions"][-logits.shape[0] :]
            offsets = [p + 1 - prompt_tokens for p in positions]
            expected = [token_ids[i] if 0 <= i < len(token_ids) else 248046 for i in offsets]
            values, indices = logits.topk(5, dim=-1)
            scores = logits.gather(1, torch.tensor(expected, device=logits.device)[:, None])
            state["logits"].append(
                {
                    "output_offsets": offsets,
                    "top_ids": indices.cpu().tolist(),
                    "top_scores": values.float().cpu().tolist(),
                    "reference_ids": expected,
                    "reference_scores": scores.float().cpu().tolist(),
                }
            )
            return logits

        runner._model_forward = traced_forward
        runner.model.compute_logits = traced_logits

    def propose(*args, **kwargs):
        proposed = original(*args, **kwargs)
        assert isinstance(proposed, torch.Tensor) and proposed.shape[0] == 1
        anchor = int(drafter.positions[0].item()) - prompt_tokens
        prefix = prefixes[len(state["calls"]) % len(prefixes)]
        assert anchor >= 0
        replacement = []
        for i in range(proposed.shape[1]):
            index = anchor + i + 1
            correct = token_ids[index] if index < len(token_ids) else 248046
            replacement.append(correct if i < prefix else (correct + 1) % 248320)
        proposed.copy_(torch.tensor([replacement], dtype=proposed.dtype, device=proposed.device))
        state["calls"].append({"anchor_offset": anchor, "correct_prefix_requested": prefix, "tokens": replacement})
        return proposed

    drafter._propose = propose
    return {"rank": worker.rank}


def collect_oracle(worker):
    state = worker._qfn_pair_oracle
    drafter = worker.model_runner.drafter
    if state["local"]:
        drafter._propose = state["original"]
    else:
        del drafter._propose
    for owner, name, original, local in reversed(state["target_originals"]):
        if local:
            setattr(owner, name, original)
        else:
            delattr(owner, name)
    del worker._qfn_pair_oracle
    return {"rank": worker.rank, "calls": state["calls"], "logits": state["logits"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("plain", "mtp", "dspark"), required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--draft")
    parser.add_argument("--draft-k", type=int, choices=(1, 2, 3, 7), default=3)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--oracle-reference", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output must be new")
    if args.mode == "dspark" and not args.draft:
        parser.error("DSpark requires --draft")
    if args.oracle_reference and args.mode != "dspark":
        parser.error("oracle rejection checks require DSpark")

    from vllm import LLM, SamplingParams

    spec = None
    if args.mode == "mtp":
        spec = {"method": "mtp", "num_speculative_tokens": 2, "enforce_eager": True}
    elif args.mode == "dspark":
        spec = {
            "model": args.draft,
            "method": "dspark",
            "num_speculative_tokens": args.draft_k,
            "draft_tensor_parallel_size": 1,
            "enforce_eager": True,
        }
    report = {
        "mode": args.mode,
        "scope": "matched eager sync TP6 batch1; no retained resident optimizations",
        "samples": [],
        "passed": False,
        "speculative_config": spec,
    }

    def save():
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    save()
    llm = LLM(
        model=args.target,
        dtype="half",
        tensor_parallel_size=6,
        enforce_eager=True,
        max_model_len=2048,
        max_num_batched_tokens=128,
        max_num_seqs=1,
        enable_prefix_caching=False,
        async_scheduling=False,
        mamba_cache_mode="align",
        language_model_only=True,
        mm_encoder_tp_mode="data",
        enable_expert_parallel=True,
        enable_ep_weight_filter=True,
        disable_custom_all_reduce=True,
        gpu_memory_utilization=0.9,
        seed=0,
        disable_log_stats=False,
        speculative_config=spec,
        additional_config={"qfn_dspark_eager_probe": True} if args.mode == "dspark" else {},
    )

    def complete(prompt, limit):
        start = time.perf_counter()
        response = llm.chat(
            [{"role": "user", "content": prompt}],
            sampling_params=SamplingParams(temperature=0, max_tokens=limit),
            chat_template_kwargs={"enable_thinking": False},
            use_tqdm=False,
        )[0]
        elapsed = time.perf_counter() - start
        out = response.outputs[0]
        assert out.token_ids and out.text.strip()
        return {
            "elapsed_s": elapsed,
            "prompt_tokens": len(response.prompt_token_ids),
            "text": out.text,
            "token_ids": list(out.token_ids),
            "finish_reason": out.finish_reason,
            "request_metrics": vars(response.metrics) if response.metrics is not None else None,
        }

    report["warmup"] = complete("Say hello.", 16)
    for repeat in range(args.repeats):
        for name, prompt in PROMPTS.items():
            before = metrics(llm)
            result = complete(prompt, args.max_tokens)
            report["samples"].append(
                {"name": name, "repeat": repeat, "spec_before": before, "spec_after": metrics(llm), **result}
            )
            save()
            print(
                json.dumps(
                    {
                        "name": name,
                        "repeat": repeat,
                        "tokens": len(result["token_ids"]),
                        "elapsed_s": result["elapsed_s"],
                    }
                ),
                flush=True,
            )
    if args.profile:
        report["profile_scope"] = "separate synchronized request; nested stages overlap; not throughput samples"
        report["profile_installed"] = llm.collective_rpc(install_profile)
        try:
            report["profile_request"] = complete(PROMPTS["merge"], 32)
        finally:
            report["profile"] = llm.collective_rpc(collect_profile)
            save()
    if args.oracle_reference:
        reference = json.loads(args.oracle_reference.read_text())
        assert reference["mode"] == "plain" and reference["passed"]
        sample = next(s for s in reference["samples"] if s["name"] == "merge")
        report["oracle_scope"] = "correctness only; scripted proposals exclude learned acceptance/performance claims"
        report["oracle"] = []
        patterns = [[n] for n in range(args.draft_k + 1)] + [[0, args.draft_k, 1, args.draft_k // 2]]
        for prefixes in patterns:
            llm.collective_rpc(install_oracle, args=(sample["token_ids"], sample["prompt_tokens"], prefixes))
            try:
                before = metrics(llm)
                result = complete(PROMPTS["merge"], min(32, len(sample["token_ids"])))
            finally:
                calls = llm.collective_rpc(collect_oracle)
            assert all(worker["calls"] for worker in calls), "oracle hook was not exercised"
            exact = result["token_ids"] == sample["token_ids"][: len(result["token_ids"])]
            report["oracle"].append(
                {
                    "prefixes": prefixes,
                    "exact_tokens": exact,
                    "workers": calls,
                    "result": result,
                    "spec_before": before,
                    "spec_after": metrics(llm),
                }
            )
            save()
        report["oracle_all_exact"] = all(case["exact_tokens"] for case in report["oracle"])
        # Shape-dependent greedy differences are recorded, not silently
        # labelled rollback correctness or used as throughput observations.
        save()
    report["passed"] = True
    save()


if __name__ == "__main__":
    main()
