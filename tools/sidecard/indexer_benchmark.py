# SPDX-License-Identifier: Apache-2.0
"""Matched full-model fused indexer qualification and completed context probe."""

import argparse
import hashlib
import json
import time
from contextlib import contextmanager
from pathlib import Path

from .expert_actor import ExpertCoordinator
from .layer_indexer import cuda_rotation_enabled
from .layer_prefix import PromptPrefix
from .layer_server import request_tokens
from .prefill_benchmark import exact_prompt
from .prefill_gate import validate

OUTPUT_TOKENS = 64


@contextmanager
def rotation_mode(engine, enabled):
    enabled = cuda_rotation_enabled({"cuda_indexer_hadamard": enabled})
    indexers = [layer.indexer for layer in engine.layers.values() if not layer.linear]
    previous = [indexer.fused_cuda_rotation for indexer in indexers]
    try:
        for indexer in indexers:
            indexer.fused_cuda_rotation = enabled
        yield
    finally:
        for indexer, value in zip(indexers, previous, strict=True):
            indexer.fused_cuda_rotation = value


def decode(engine, logits, *, forced=None):
    hashes, tokens, spans = [], [], []
    for step in range(OUTPUT_TOKENS):
        observed = logits.cpu()
        if not bool(observed.isfinite().all()):
            raise AssertionError("context probe produced non-finite vocabulary logits")
        hashes.append(hashlib.sha256(observed.numpy().tobytes()).hexdigest())
        token = int(observed.argmax()) if forced is None else forced[step]
        tokens.append(token)
        if step < OUTPUT_TOKENS - 1:
            begin = time.perf_counter()
            logits = engine.forward([token])
            spans.append(time.perf_counter() - begin)
    return hashes, tokens, spans


def run(config, chat_request, reference_path, reference_sha256, *, complete_context=False):
    """Preserve math, compare all output vectors, then exercise the full cache."""
    from transformers import AutoTokenizer

    spec = json.loads(config.read_bytes())
    raw = reference_path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != reference_sha256 or not json.loads(raw).get("passed"):
        raise ValueError("indexer benchmark reference failed authentication or qualification")
    if spec.get("prompt_prefix_cache") or not cuda_rotation_enabled(spec):
        raise ValueError("matched indexer benchmark requires no snapshots and an admitted fused CUDA path")
    qualified = json.loads(raw)
    output = Path(spec["output"]) / "indexer-benchmark.json"
    report = dict(passed=False, measurements=[], complete_context=None, decode_excludes_logit_observers=True)
    engine = ExpertCoordinator(config)
    try:
        tokenizer = AutoTokenizer.from_pretrained(spec["checkpoint"], local_files_only=True, trust_remote_code=False)
        seed = tokenizer.encode("Inspect this module and explain its control flow. ")
        validate(
            engine,
            seed,
            output=engine.output / "prefill-parity.json",
            reference_path=Path(spec["prefill_reference"]),
            reference_sha256=spec["prefill_reference_sha256"],
        )
        full, _ = request_tokens(json.loads(chat_request.read_bytes()), tokenizer, engine.max_context, chat=True)
        for name, prompt in (("2K", exact_prompt(seed, 2048)), ("8K", exact_prompt(seed, 8192)), ("full-code", full)):
            historical = next(
                row
                for row in qualified["measurements"]
                if row["case"] == name and row["window_rows"] == engine.prefill_window_size
            )
            signature = hashlib.sha256(json.dumps(prompt, separators=(",", ":")).encode()).hexdigest()
            if signature != historical["prompt_ids_sha256"]:
                raise AssertionError("matched benchmark changed the complete prompt")
            forced = None
            for enabled in (False, True):
                with rotation_mode(engine, enabled):
                    engine.reset(context_limit=len(prompt) + OUTPUT_TOKENS)
                    begin = time.perf_counter()
                    logits = engine.prefill(prompt)
                    prefill_seconds = time.perf_counter() - begin
                    hashes, forced, spans = decode(engine, logits, forced=forced)
                    equal = hashes == historical["logits_sha256"]
                    if not equal:
                        raise AssertionError("fused indexer changed full-vocabulary model logits")
                    row = dict(
                        case=name,
                        fused=enabled,
                        prompt_tokens=len(prompt),
                        prompt_ids_sha256=signature,
                        cached_tokens=0,
                        prefill_seconds=prefill_seconds,
                        decode_tokens_per_second=len(spans) / sum(spans),
                        decode_forward_seconds=spans,
                        full_vocabulary_bitwise_equal=True,
                        logits_sha256=hashes,
                    )
                    report["measurements"].append(row)
                    output.write_text(json.dumps(report, indent=2) + "\n")
                    print(
                        json.dumps(
                            {
                                key: value
                                for key, value in row.items()
                                if key not in ("logits_sha256", "decode_forward_seconds")
                            }
                        ),
                        flush=True,
                    )
        if complete_context:
            prompt = exact_prompt(seed, engine.max_context - OUTPUT_TOKENS)
            engine.reset(context_limit=engine.max_context)
            begin = time.perf_counter()
            logits = engine.prefill(prompt)
            prefill_seconds = time.perf_counter() - begin
            snapshots = PromptPrefix(engine)
            snapshot = snapshots.snapshot(prompt, logits, True)
            hashes, tokens, spans = decode(engine, logits)
            engine.reset(context_limit=engine.max_context)
            restored = snapshots.restore(snapshot)
            repeated, _, _ = decode(engine, restored, forced=tokens)
            if repeated != hashes:
                raise AssertionError("full-context vocabulary repeat was not bitwise equal")
            report["complete_context"] = dict(
                passed=True,
                generation_completed=True,
                max_context=engine.max_context,
                prompt_tokens=len(prompt),
                output_tokens=OUTPUT_TOKENS,
                cached_tokens=0,
                prompt_ids_sha256=hashlib.sha256(json.dumps(prompt, separators=(",", ":")).encode()).hexdigest(),
                prefill_seconds=prefill_seconds,
                decode_tokens_per_second=len(spans) / sum(spans),
                vocabulary_repeat_bitwise_equal=True,
                logits_sha256=hashes,
                snapshot_bytes=snapshots.size(snapshot),
                gpu_free_bytes_with_snapshot=engine.torch.cuda.mem_get_info()[0],
                task_accuracy_evaluated=False,
            )
            print(
                json.dumps({key: value for key, value in report["complete_context"].items() if key != "logits_sha256"}),
                flush=True,
            )
        report["passed"] = True
    except Exception as error:
        report["error"] = dict(type=type(error).__name__, message=str(error))
        raise
    finally:
        engine.close()
        output.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--chat-request", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--reference-sha256", required=True)
    parser.add_argument("--complete-context", action="store_true")
    args = parser.parse_args()
    run(args.config, args.chat_request, args.reference, args.reference_sha256, complete_context=args.complete_context)


if __name__ == "__main__":
    main()
