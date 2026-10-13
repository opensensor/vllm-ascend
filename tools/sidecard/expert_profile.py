# SPDX-License-Identifier: Apache-2.0
"""Bounded complete-model attribution without adding synchronization barriers.

Host partitions are disjoint. GPU/NPU event spans include stream idle/launch
gaps and overlap host waits; never sum them into the host latency waterfall.
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

from .cold_dispatch import HEADER
from .dma_protocol import COMMAND
from .expert_actor import ExpertCoordinator
from .expert_timing import TimedPeer
from .layer_server import Generation, request_tokens
from .layer_weights import NPU_LAYERS, prefill_chunk_size, prefill_window_size
from .prefill_benchmark import exact_prompt

OUTPUT_TOKENS = 64


class ProfileStage:
    def __init__(self, stage, owner, name):
        self.stage, self.owner, self.name = stage, owner, name

    def forward(self, state):
        if not self.owner.enabled:
            return self.stage.forward(state)
        start = self.owner.event()
        begin = time.perf_counter()
        result = self.stage.forward(state)
        self.owner.add(self.name + "_host_submit", time.perf_counter() - begin)
        self.owner.pair(self.name, start)
        return result

    def close(self):
        self.stage.close()


class ProfileDispatch:
    def __init__(self, dispatch, owner):
        self.dispatch, self.owner = dispatch, owner

    def with_shared(self, layer, x, ids, weights):
        if not self.owner.enabled:
            return self.dispatch.with_shared(layer, x, ids, weights)
        events = []
        for peer in self.dispatch.connections:
            peer.record = events
        start = self.owner.event()
        begin = time.perf_counter()
        try:
            result = self.dispatch.with_shared(layer, x, ids, weights)
        finally:
            for peer in self.dispatch.connections:
                peer.record = None
        end = time.perf_counter()
        self.owner.pair("expert_dispatch", start, layer.index)
        sends = [event for event in events if event["direction"] == "send"]
        receives = [event for event in events if event["direction"] == "receive"]
        ready = min(event["begin"] for event in sends) - begin if sends else 0
        send = sum(event["end"] - event["begin"] for event in sends)
        receive = sum(event["end"] - event["begin"] for event in receives)
        self.owner.add("input_readiness", ready)
        self.owner.add("socket_send", send)
        self.owner.add("socket_receive_wait", receive)
        self.owner.add("hot_shared_reply_merge_submit", end - begin - ready - send - receive)
        rows = x.shape[0]
        cold = None
        if self.owner.spec.get("expert_transport") != "direct-dma":
            offset = rows * 4096 * 2
            host_ids = self.dispatch.host[offset : offset + rows * 8 * 4].view(self.owner.torch.int32).numpy()
            hot = set(self.dispatch.hot[str(layer.index)])
            cold = {
                rank: int(sum(expert not in hot and expert // 144 == rank for expert in host_ids.flat))
                for rank in (0, 1)
            }
        self.owner.frame["dispatches"].append(
            dict(layer=layer.index, rows=rows, cold_routes=cold, begin=begin, end=end, socket_events=events)
        )
        return result


class ProfileMiddle:
    def __init__(self, stage, owner):
        self.stage, self.owner = stage, owner

    def forward(self, state):
        if not self.owner.enabled:
            return self.stage.forward(state)
        for layer in self.stage.layers:
            start = self.owner.event()
            begin = time.perf_counter()
            captured = self.stage.graphs.get(layer.index)
            dsa = getattr(self.stage, "dsa_graphs", {}).get(layer.index)
            if dsa is not None and state[0].shape[0] == 1:
                inputs = dsa.forward(state)
            elif captured is None or state[0].shape[0] != 1:
                prepared = layer.prepare_forward(*state)
                inputs = (*prepared, *layer.route(prepared[0]))
            else:
                graph, inputs, outputs = captured
                for target, source in zip(inputs, state, strict=True):
                    target.copy_(source)
                graph.replay()
                inputs = outputs
            self.owner.add("middle_backbone_host_submit", time.perf_counter() - begin)
            self.owner.pair("middle_backbone", start, layer.index)
            begin = time.perf_counter()
            before = sum(self.owner.frame["host_seconds"].values())
            state = layer.finish_forward(*inputs)
            dispatch = sum(self.owner.frame["host_seconds"].values()) - before
            self.owner.add("middle_finish_host_submit", time.perf_counter() - begin - dispatch)
        return state

    def close(self):
        self.stage.close()


class ProfileCoordinator(ExpertCoordinator):
    def __init__(self, config, *, decode_only=False):
        super().__init__(config)
        self.decode_only = decode_only
        self.workload_logits = None
        self.capture_on_forward = self.capture_active = False
        if self.prefix_graphs is None or self.middle_graphs is None or self.suffix_graphs is None:
            self.close()
            raise ValueError("profile requires the qualified CUDA graph architecture")
        self.enabled, self.frame, self.frames = False, None, []
        header = COMMAND if self.spec.get("expert_transport") == "direct-dma" else HEADER
        self.experts.connections = [
            TimedPeer(peer, rank, header_size=header.size, header_decoder=header.unpack)
            for rank, peer in enumerate(self.experts.connections)
        ]
        dispatch = ProfileDispatch(self.experts, self)
        for index in NPU_LAYERS:
            self.layers[index].experts = dispatch
        self.prefix_graphs = ProfileStage(self.prefix_graphs, self, "prefix")
        self.suffix_graphs = ProfileStage(self.suffix_graphs, self, "suffix")
        self.middle_graphs = ProfileMiddle(self.middle_graphs, self)

    def observe_logits(self, logits):
        if self.workload_logits is not None:
            self.workload_logits.append(hashlib.sha256(logits.cpu().numpy().tobytes()).hexdigest())

    def prefill(self, tokens, **kwargs):
        # Cold windows do not pass through the staged forward observer. Keep
        # their production geometry and observe only the subsequent decode.
        enabled = self.enabled
        if self.decode_only:
            self.enabled = False
        try:
            result = super().prefill(tokens, **kwargs)
        finally:
            self.enabled = enabled
        self.observe_logits(result)
        return result

    def event(self):
        event = self.torch.cuda.Event(enable_timing=True)
        event.record()
        return event

    def pair(self, name, start, layer=None):
        self.frame["event_pairs"].append((name, layer, start, self.event()))

    def add(self, name, seconds):
        self.frame["host_seconds"][name] = self.frame["host_seconds"].get(name, 0) + seconds

    def forward(self, tokens):
        if not self.enabled:
            return super().forward(tokens)
        if self.capture_on_forward:
            self.torch.cuda.profiler.start()
            self.capture_on_forward, self.capture_active = False, True
        self.frame = dict(
            sequence=self.sequence,
            position=self.position,
            rows=len(tokens),
            host_seconds={},
            event_pairs=[],
            dispatches=[],
        )
        begin = time.perf_counter()
        result = super().forward(tokens)
        total = time.perf_counter() - begin
        self.add("embedding_head_final_completion", total - sum(self.frame["host_seconds"].values()))
        self.frame["total_ms"] = total * 1000
        self.frame["host_ms"] = {key: value * 1000 for key, value in self.frame.pop("host_seconds").items()}
        self.frame["device_stream_ms"] = [
            dict(name=name, layer=layer, ms=start.elapsed_time(end))
            for name, layer, start, end in self.frame.pop("event_pairs")
        ]
        self.frames.append(self.frame)
        self.frame = None
        # Vocabulary checks occur after the timed forward and its existing sync.
        self.observe_logits(result)
        return result


def run(
    config,
    reference,
    *,
    decode_only=False,
    chat_request=None,
    benchmark_reference=None,
    benchmark_sha256=None,
    capture_cuda=False,
):
    """Qualify observers and collect bounded real-model attribution."""
    from transformers import AutoTokenizer

    spec = json.loads(Path(config).read_bytes())
    if spec.get("expert_transport") != "direct-dma" and not spec.get("expert_worker_profile"):
        raise ValueError("profile snapshot must enable worker device-event receipts")
    if capture_cuda and not decode_only:
        raise ValueError("CUDA capture requires decode-only profiling")
    if not decode_only and prefill_window_size(spec) > prefill_chunk_size(spec):
        raise ValueError("cold batch profiling requires decode-only observation")
    benchmark = None
    if decode_only:
        if spec.get("prompt_prefix_cache", False):
            raise ValueError("decode profiling requires fresh model state without prefix snapshots")
        if chat_request is None or benchmark_reference is None or benchmark_sha256 is None:
            raise ValueError("decode-only profiling requires its full prompt and authenticated vocabulary reference")
        raw = Path(benchmark_reference).read_bytes()
        if hashlib.sha256(raw).hexdigest() != benchmark_sha256:
            raise ValueError("profile benchmark reference failed authentication")
        benchmark = json.loads(raw)
        if not benchmark.get("passed"):
            raise ValueError("profile benchmark reference did not pass")
    reference_bytes = Path(reference).read_bytes()
    if (
        spec.get("transport_reference_sha256")
        and hashlib.sha256(reference_bytes).hexdigest() != spec["transport_reference_sha256"]
    ):
        raise ValueError("profile transport reference failed authentication")
    output, engine = Path(spec["output"]), None
    report = dict(
        passed=False,
        production_speed_validated=False,
        transport=spec.get("expert_transport", "socket"),
        decode_only=decode_only,
        cuda_capture_workload="nsight-2K" if capture_cuda else None,
        vocabulary_checks_outside_forward_timing=decode_only,
        workloads=[],
    )
    try:
        engine = ProfileCoordinator(config, decode_only=decode_only)
        tokenizer = AutoTokenizer.from_pretrained(spec["checkpoint"], local_files_only=True, trust_remote_code=False)
        expected = json.loads(reference_bytes)
        prompt = tokenizer.encode('def factorial(n):\n    """Compute factorial recursively."""\n')
        engine.reset(context_limit=len(prompt) + OUTPUT_TOKENS)
        engine.enabled = True
        engine.fingerprints = []
        engine.prefill(prompt)
        for token in expected["teacher_forced_tokens"][: OUTPUT_TOKENS - 1]:
            engine.forward([token])
        report["full_vocabulary_reference_bitwise_equal"] = engine.fingerprints == expected["traces"][0]
        if not report["full_vocabulary_reference_bitwise_equal"]:
            raise AssertionError("profile changed full-vocabulary output bytes")
        engine.fingerprints = None
        engine.frames.clear()
        generation = Generation(engine, tokenizer)
        if decode_only:
            seed = tokenizer.encode("Inspect this module and explain its control flow. ")
            full, _ = request_tokens(
                json.loads(Path(chat_request).read_bytes()), tokenizer, engine.max_context, chat=True
            )
            workloads = [("2K", exact_prompt(seed, 2048)), ("8K", exact_prompt(seed, 8192)), ("full-code", full)]
            if capture_cuda:
                workloads.append(("nsight-2K", exact_prompt(seed, 2048)))
        else:
            seed = tokenizer.encode(
                "Inspect this module for profiler test 20261011 and explain the control flow. ",
                add_special_tokens=False,
            )
            workloads = [("matched2048", exact_prompt(seed, 2048)), ("python", prompt)]
        for name, tokens in workloads:
            result = {}
            first = engine.sequence
            engine.workload_logits = [] if decode_only else None
            engine.capture_on_forward = capture_cuda and name == "nsight-2K"
            limit = OUTPUT_TOKENS // 2 if name == "nsight-2K" else OUTPUT_TOKENS
            list(generation.generate(tokens, limit, result, ignore_eos=True))
            if engine.capture_active:
                engine.torch.cuda.profiler.stop()
                engine.capture_active = False
            row = dict(
                name=name,
                first_sequence=first,
                end_sequence=engine.sequence,
                prompt_tokens=len(tokens),
                cached_prompt_tokens=result["cached_prompt_tokens"],
            )
            if decode_only:
                reference_row = next(
                    x
                    for x in benchmark["measurements"]
                    if x["case"] == ("2K" if name == "nsight-2K" else name)
                    and x["window_rows"] == prefill_window_size(spec)
                )
                token_hash = hashlib.sha256(json.dumps(tokens, separators=(",", ":")).encode()).hexdigest()
                equal = engine.workload_logits == reference_row["logits_sha256"][:limit]
                if token_hash != reference_row["prompt_ids_sha256"] or result["cached_prompt_tokens"] != 0 or not equal:
                    raise AssertionError("decode profile changed its full prompt, cache state, or vocabulary logits")
                row.update(
                    prompt_ids_sha256=token_hash,
                    full_vocabulary_bitwise_equal=equal,
                    logits_sha256=engine.workload_logits,
                    output_tokens=len(result["output_ids"]),
                    cuda_kernel_capture=name == "nsight-2K",
                )
            else:
                row.update(output_ids=result["output_ids"], text=result["text"])
            engine.workload_logits = None
            report["workloads"].append(row)
            (output / "profile-run.json").write_text(json.dumps(report, indent=2) + "\n")
        report["passed"] = True
    except Exception as error:
        report["error"] = dict(type=type(error).__name__, message=str(error))
        raise
    finally:
        if engine is not None:
            if engine.capture_active:
                engine.torch.cuda.profiler.stop()
                engine.capture_active = False
            (output / "parent-profile.jsonl").write_text("".join(json.dumps(frame) + "\n" for frame in engine.frames))
            engine.close()
        (output / "profile-run.json").write_text(json.dumps(report, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--decode-only", action="store_true")
    parser.add_argument("--chat-request", type=Path)
    parser.add_argument("--benchmark-reference", type=Path)
    parser.add_argument("--benchmark-sha256")
    parser.add_argument("--capture-cuda", action="store_true")
    args = parser.parse_args()
    run(
        args.config,
        args.reference,
        decode_only=args.decode_only,
        chat_request=args.chat_request,
        benchmark_reference=args.benchmark_reference,
        benchmark_sha256=args.benchmark_sha256,
        capture_cuda=args.capture_cuda,
    )


if __name__ == "__main__":
    main()
