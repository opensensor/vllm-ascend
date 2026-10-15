# SPDX-License-Identifier: Apache-2.0
"""Controlled QFN comparison with explicit all-rank collective teardown.

Uses the qualified checkpoint/cache/operator stack supplied by the launcher
process. This harness does not install kernels or change environment variables.
Reported wall throughput includes prefill; it is not isolated decode speed.
"""

import argparse
import json
import signal
import time
from pathlib import Path
from types import MappingProxyType

from .qfn_collective_worker import validate_graph_snapshot

MODEL = "/srv/ai/models/Qwen3.8-Flash-Next-W4A16-G128-300i"
KV_CACHE_MEMORY_BYTES = 88673894400
GRAPH_PROFILES = MappingProxyType({"eager": (), "c1-c2": (3, 6), "c3-c4": (9, 12)})
GRAPH_EVIDENCE_COUNTERS = frozenset(
    {"registered_graphs", "captured_graphs", "captured_calls", "native_leases_acquired", "native_leases_released"}
)
PROMPTS = (
    "Explain why a computer cache improves performance. Give a clear technical explanation with examples.",
    "Describe how an electric motor works, including the roles of current and magnetic fields.",
    "Explain how a database index makes queries faster and describe the tradeoffs involved.",
    "Describe how TCP delivers a reliable ordered byte stream over an unreliable network.",
)


def graph_sizes(profile):
    if type(profile) is not str or profile not in GRAPH_PROFILES:
        raise ValueError("graph_profile must be eager, c1-c2 or c3-c4")
    return GRAPH_PROFILES[profile]


def expects_graph_replay(profile, concurrency):
    """Expected primary decode dispatch; C4 tails can enter smaller graphs."""
    sizes = graph_sizes(profile)
    return bool(sizes) and concurrency * 3 <= max(sizes)


def engine_options(collective_config=None, *, graph_profile="eager"):
    sizes = graph_sizes(graph_profile)
    options = dict(
        model=MODEL,
        dtype="float16",
        tensor_parallel_size=4,
        async_scheduling=False,
        disable_log_stats=False,
        disable_custom_all_reduce=True,
        max_model_len=262144,
        max_num_batched_tokens=2048,
        max_num_seqs=4,
        gpu_memory_utilization=0.965,
        kv_cache_memory_bytes=KV_CACHE_MEMORY_BYTES,
        language_model_only=True,
        enable_expert_parallel=True,
        enable_ep_weight_filter=True,
        mamba_cache_mode="align",
        enable_chunked_prefill=True,
        enable_prefix_caching=True,
        speculative_config=dict(method="mtp", num_speculative_tokens=2),
        compilation_config=dict(mode=0, cudagraph_mode="NONE"),
        enforce_eager=True,
        limit_mm_per_prompt=dict(image=0, video=0),
        seed=0,
        worker_extension_cls="tools.sidecard.qfn_collective_worker.QfnCollectiveWorkerExtension",
        hf_overrides=dict(
            text_config=dict(
                ascend_expert_quantization=dict(
                    backend="cube_310_int4_a8",
                    activation_quantization="int8_per_group",
                    bits=4,
                    format="qwen4exp_w4a16_group_v1",
                    group_size=128,
                    lm_head_execution="w8a8_dynamic",
                    ple_projection_execution="w8a8_dynamic",
                    offset_dtype="int8",
                    packing="signed_int4_low_nibble_first_in_axis",
                    scale_dtype="float16",
                    shared_expert_execution="tp_sharded",
                    symmetric=False,
                )
            )
        ),
    )
    if collective_config is not None:
        options["additional_config"] = dict(npu_collective=collective_config)
    if sizes:
        options["enforce_eager"] = False
        options["compilation_config"] = dict(
            mode=0, cudagraph_mode="FULL_DECODE_ONLY", cudagraph_capture_sizes=list(sizes)
        )
    return options


def output_receipt(output):
    metrics = getattr(output, "metrics", None)
    values = {}
    for name in ("arrival_time", "scheduled_ts", "first_token_ts", "last_token_ts"):
        value = getattr(metrics, name, None)
        if isinstance(value, (int, float)):
            values[name] = value
    return dict(
        prompt_tokens=len(output.prompt_token_ids),
        metrics=values,
        num_cached_tokens=getattr(output, "num_cached_tokens", None),
        completions=[
            dict(text=item.text, token_ids=list(item.token_ids), finish_reason=item.finish_reason)
            for item in output.outputs
        ],
    )


def verify_workers(workers, *, custom, closed=False, require_calls=False):
    if type(workers) is not list or len(workers) != 4:
        raise RuntimeError("exactly four worker status receipts are required")
    for worker in workers:
        if type(worker) is not dict or worker.get("enabled") is not custom:
            raise RuntimeError("worker collective backend differs from requested mode")
        if custom:
            if worker.get("state") != ("closed" if closed else "active"):
                raise RuntimeError("custom worker collective state differs")
            if require_calls and not sum(g["calls"] for g in worker["geometries"]):
                raise RuntimeError("custom collective did not execute on every rank")


def coverage_deltas(before, after, *, custom, require_calls=True):
    verify_workers(before, custom=custom)
    verify_workers(after, custom=custom)
    if not custom:
        return []
    results = []
    for old, new in zip(before, after, strict=True):

        def counters(packet, field, keys):
            return {tuple(value.get(k) for k in keys): value["calls"] for value in packet.get(field, [])}

        rank = {}
        for field, keys in (
            ("geometries", ("dtype", "elements")),
            ("fallback_schemas", ("dtype", "elements", "contiguous")),
        ):
            first, last = counters(old, field, keys), counters(new, field, keys)
            if set(first) - set(last):
                raise RuntimeError("collective counter schema disappeared")
            delta = {key: value - first.get(key, 0) for key, value in last.items()}
            if any(value < 0 for value in delta.values()):
                raise RuntimeError("collective counter moved backwards")
            rank[field] = [
                dict(zip(keys, key, strict=True), calls=value) for key, value in sorted(delta.items()) if value
            ]
        if require_calls and not sum(item["calls"] for item in rank["geometries"]):
            raise RuntimeError("measurement did not execute custom SUM on every rank")
        results.append(rank)
    if any(result != results[0] for result in results):
        raise RuntimeError("rank collective call deltas differ")
    return results


def validate_replay_evidence(coverage, *, required):
    """Validate capture-only production evidence or explicit diagnostic counts."""
    if type(coverage) is not list or len(coverage) != 4:
        raise ValueError("graph evidence is required on all four ranks")
    for rank in coverage:
        if (
            type(rank) is not dict
            or set(rank)
            != GRAPH_EVIDENCE_COUNTERS | {"tracking_enabled", "count_semantics", "replay_submissions", "replayed_calls"}
            or type(rank["tracking_enabled"]) is not bool
            or any(type(rank[name]) is not int or rank[name] < 0 for name in GRAPH_EVIDENCE_COUNTERS)
            or any(
                rank[name] == 0
                for name in ("registered_graphs", "captured_graphs", "captured_calls", "native_leases_acquired")
            )
        ):
            raise ValueError("positive captured custom graph and native lease evidence required")
        semantics = "capture-and-replay-submissions" if rank["tracking_enabled"] else "capture-only"
        if rank["count_semantics"] != semantics:
            raise ValueError("graph evidence count semantics differ")
        for name in ("replay_submissions", "replayed_calls"):
            value = rank[name]
            if rank["tracking_enabled"]:
                if type(value) is not int or value < 0 or required and value == 0:
                    raise ValueError("expected diagnostic graph replay requires positive exact counters")
            elif value is not None:
                raise ValueError("production graph replays are unobserved, not counted")
    for field in ("tracking_enabled", "count_semantics", "replay_submissions", "replayed_calls"):
        if any(rank[field] != coverage[0][field] for rank in coverage):
            raise ValueError("graph replay evidence mode or counters differ across ranks")


def replay_deltas(before, after, *, required):
    """No replay counter is invented when production tracking is disabled."""
    deltas = []
    for old, new in zip(before, after, strict=True):
        first, last = old.get("graph_observer"), new.get("graph_observer")
        if type(first) is not dict or type(last) is not dict:
            raise RuntimeError("graph profile requires every rank's graph observer evidence")
        try:
            first, last = validate_graph_snapshot(first), validate_graph_snapshot(last)
        except ValueError as error:
            raise RuntimeError("invalid graph observer snapshot") from error
        for packet in (first, last):
            if (
                packet.get("lifecycle_only") is not True
                or packet.get("retired") is not False
                or packet.get("active_capture") is not False
                or packet.get("failed_graphs") != 0
            ):
                raise RuntimeError("graph lifecycle is failed, retired or still capturing")
        if first["tracking_enabled"] is not last["tracking_enabled"]:
            raise RuntimeError("graph tracking mode changed during a measurement")
        values = {name: last[name] for name in GRAPH_EVIDENCE_COUNTERS}
        values.update(tracking_enabled=last["tracking_enabled"], count_semantics=last["count_semantics"])
        for field in ("replay_submissions", "replayed_calls"):
            if any(type(packet.get(field)) is not int or packet[field] < 0 for packet in (first, last)):
                raise RuntimeError("exact nonnegative graph replay counters are required")
            delta = last[field] - first[field]
            values[field] = delta if last["tracking_enabled"] else None
            if delta < 0 or last["tracking_enabled"] and required and delta == 0:
                raise RuntimeError("expected graph replay did not advance on every rank")
        deltas.append(values)
    try:
        validate_replay_evidence(deltas, required=required)
    except ValueError as error:
        raise RuntimeError("invalid graph capture/replay evidence") from error
    return deltas


def verify_completion_budget(outputs, count, tokens):
    if len(outputs) != count:
        raise RuntimeError("generation returned a different request count")
    for output in outputs:
        if len(output["completions"]) != 1:
            raise RuntimeError("generation returned a different completion count")
        completion = output["completions"][0]
        if len(completion["token_ids"]) != tokens or completion["finish_reason"] != "length":
            raise RuntimeError("generation did not complete the requested token budget")


def run(output, collective_config=None, *, llm_factory=None, sampling_factory=None, graph_profile="eager"):
    options = engine_options(collective_config, graph_profile=graph_profile)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    (output / "engine-options.json").write_text(json.dumps(options, indent=2) + "\n")
    if llm_factory is None:
        from vllm import LLM, SamplingParams

        llm_factory, sampling_factory = LLM, SamplingParams
    started = time.monotonic()
    llm = llm_factory(**options)
    result = dict(
        mode="custom" if collective_config else "hccl",
        eager=not bool(graph_sizes(graph_profile)),
        graph_profile=graph_profile,
        startup_seconds=time.monotonic() - started,
        measurements=[],
        scope="prefix-warmed completed generation including prefill; not isolated decode",
        model_numerical_qualification=False,
        graph_evidence_scope=(
            "cold custom captures/native leases; replay counts are unobserved unless diagnostic "
            "tracking is explicitly enabled; verify qualified runtime graph dispatch logs"
        ),
    )
    result["initial_workers"] = llm.collective_rpc("npu_collective_status")
    verify_workers(result["initial_workers"], custom=collective_config is not None)
    if graph_profile != "eager" and collective_config is not None:
        tracking = collective_config.get("graph_replay_tracking", False)
        if type(tracking) is not bool or any(
            worker.get("graph_observer", {}).get("tracking_enabled") is not tracking
            for worker in result["initial_workers"]
        ):
            raise RuntimeError("graph observer tracking differs from the explicitly configured mode")
    tokenizer = llm.get_tokenizer()
    prompts = [
        tokenizer.apply_chat_template(
            [dict(role="user", content=p)], tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        for p in PROMPTS
    ]
    c1_tokens = 256 if graph_profile == "eager" else 512
    for count, tokens, label in (
        (4, 32, "warmup"),
        (1, c1_tokens, "c1-a"),
        (4, 128, "c4-a"),
        (1, c1_tokens, "c1-b"),
        (4, 128, "c4-b"),
    ):
        params = sampling_factory(temperature=0, max_tokens=tokens, min_tokens=tokens, ignore_eos=True)
        before_workers = llm.collective_rpc("npu_collective_status")
        begin = time.monotonic()
        outputs = llm.generate(prompts[:count], params, use_tqdm=False)
        elapsed = time.monotonic() - begin
        receipts = [output_receipt(item) for item in outputs]
        verify_completion_budget(receipts, count, tokens)
        generated = sum(len(c["token_ids"]) for item in receipts for c in item["completions"])
        result["measurements"].append(
            dict(
                label=label,
                concurrency=count,
                elapsed_seconds=elapsed,
                generated_tokens=generated,
                wall_tokens_per_second=generated / elapsed,
                outputs=receipts,
            )
        )
        result["workers"] = llm.collective_rpc("npu_collective_status")
        result["measurements"][-1]["collective_coverage"] = coverage_deltas(
            before_workers,
            result["workers"],
            custom=collective_config is not None,
            require_calls=graph_profile == "eager",
        )
        if graph_profile != "eager":
            measurement = result["measurements"][-1]
            measurement["expected_graph_replay"] = expects_graph_replay(graph_profile, count)
            if collective_config is not None:
                measurement["graph_replay_coverage"] = replay_deltas(
                    before_workers, result["workers"], required=measurement["expected_graph_replay"]
                )
                measurement["replay_counter_observed"] = measurement["graph_replay_coverage"][0]["tracking_enabled"]
        verify_workers(result["workers"], custom=collective_config is not None, require_calls=graph_profile == "eager")
        (output / "progress.json").write_text(json.dumps(result, indent=2) + "\n")
    result["closed_workers"] = llm.collective_rpc("npu_collective_close")
    verify_workers(
        result["closed_workers"],
        custom=collective_config is not None,
        closed=True,
        require_calls=graph_profile == "eager",
    )
    llm.llm_engine.engine_core.shutdown()
    result["ownership_released"] = True
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--collective-config", type=Path)
    parser.add_argument("--graph-profile", choices=tuple(GRAPH_PROFILES), default="eager")
    args = parser.parse_args()
    config = json.loads(args.collective_config.read_text()) if args.collective_config else None
    try:
        result = run(args.output, config, graph_profile=args.graph_profile)
        print(json.dumps(dict(mode=result["mode"], ownership_released=result["ownership_released"])))
    except BaseException:
        # Preserve partial SDK owners on failure. Recovery is an explicit action.
        import traceback

        traceback.print_exc()
        while True:
            signal.pause()


if __name__ == "__main__":
    main()
