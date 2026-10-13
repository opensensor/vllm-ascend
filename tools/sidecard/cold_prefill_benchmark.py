# SPDX-License-Identifier: Apache-2.0
"""Compare cold expert batching on identical prompts with fresh model state."""

import argparse
import hashlib
import json
import time
from pathlib import Path

from .expert_actor import ExpertCoordinator
from .layer_server import request_tokens
from .layer_weights import COLD_PREFILL_WINDOW_CHOICES, MAX_CHUNK, prefill_window_size
from .prefill_benchmark import exact_prompt
from .prefill_gate import validate

OUTPUT_TOKENS = 64


def run(engine, cases, *, output, windows=None):
    """Measure compute, bypassing the API queue and every prefix-cache snapshot.

    The caller owns the loaded engine and its shutdown. Decode is teacher forced
    for a complete vocabulary comparison, with observer costs in its timing.
    These timings are model compute measurements, not HTTP TTFT measurements.
    """
    original_window, original_fingerprints = engine.prefill_window_size, engine.fingerprints
    windows = tuple(windows) if windows is not None else (MAX_CHUNK, original_window)
    if (
        engine.prefill_chunk_size != MAX_CHUNK
        or original_window not in COLD_PREFILL_WINDOW_CHOICES
        or len(windows) < 2
        or any(type(value) is not int or value not in (MAX_CHUNK, *COLD_PREFILL_WINDOW_CHOICES) for value in windows)
        or len(set(windows)) != len(windows)
        or max(windows) > original_window
    ):
        raise ValueError("matched cold benchmark requires distinct admitted cold windows and 32-row CUDA chunks")
    report = dict(passed=False, cached_prompt_tokens=0, measurements=[], decode_includes_parity_observer=True)
    try:
        engine.fingerprints = None
        for name, tokens in cases:
            if not tokens or len(tokens) + OUTPUT_TOKENS > engine.max_context:
                raise ValueError("benchmark prompt and output must fit model capacity")
            token_hash = hashlib.sha256(json.dumps(tokens, separators=(",", ":")).encode()).hexdigest()
            forced, baseline = [], []
            for window in windows:
                engine.prefill_window_size = window
                engine.reset(context_limit=len(tokens) + OUTPUT_TOKENS)
                started = time.perf_counter()
                logits = engine.prefill(tokens)
                prefill_seconds = time.perf_counter() - started
                hashes, max_error = [], 0.0
                decode_started = time.perf_counter()
                for step in range(OUTPUT_TOKENS):
                    observed = logits.cpu()
                    hashes.append(hashlib.sha256(observed.numpy().tobytes()).hexdigest())
                    if window == windows[0]:
                        baseline.append(observed)
                        forced.append(int(observed.argmax()))
                    else:
                        max_error = max(max_error, float((baseline[step] - observed).abs().max()))
                    if step < OUTPUT_TOKENS - 1:
                        logits = engine.forward([forced[step]])
                decode_seconds = time.perf_counter() - decode_started
                equal = hashes == [hashlib.sha256(value.numpy().tobytes()).hexdigest() for value in baseline]
                row = dict(
                    case=name,
                    prompt_tokens=len(tokens),
                    prompt_ids_sha256=token_hash,
                    window_rows=window,
                    cuda_chunk_rows=MAX_CHUNK,
                    cached_prompt_tokens=0,
                    prefill_seconds=prefill_seconds,
                    prompt_tokens_per_second=len(tokens) / prefill_seconds,
                    decode_tokens_per_second=(OUTPUT_TOKENS - 1) / decode_seconds,
                    full_vocabulary_bitwise_equal=equal,
                    max_abs_logit_error=max_error,
                    logits_sha256=hashes,
                )
                report["measurements"].append(row)
                output.write_text(json.dumps(report, indent=2) + "\n")
                print(json.dumps({key: value for key, value in row.items() if key != "logits_sha256"}), flush=True)
                if not equal:
                    raise AssertionError("matched cold benchmark changed full-vocabulary logits")
        report["passed"] = True
        return report
    except BaseException as error:
        report["error"] = dict(type=type(error).__name__, message=str(error))
        raise
    finally:
        engine.prefill_window_size, engine.fingerprints = original_window, original_fingerprints
        output.write_text(json.dumps(report, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--chat-request", type=Path, required=True, help="Private captured full Code request JSON")
    parser.add_argument("--windows", nargs="+", type=int, choices=(MAX_CHUNK, *COLD_PREFILL_WINDOW_CHOICES))
    args = parser.parse_args()
    spec = json.loads(args.config.read_bytes())
    if prefill_window_size(spec) not in COLD_PREFILL_WINDOW_CHOICES:
        raise ValueError("benchmark runtime must allocate a supported cold window")
    # Load device libraries and tokenizer only in the process owning the runtime.
    from transformers import AutoTokenizer

    engine = ExpertCoordinator(args.config)
    try:
        tokenizer = AutoTokenizer.from_pretrained(spec["checkpoint"], trust_remote_code=False)
        seed = tokenizer.encode("Inspect this module and explain its control flow. ")
        validate(
            engine,
            seed,
            output=engine.output / "prefill-parity.json",
            reference_path=Path(spec["prefill_reference"]),
            reference_sha256=spec["prefill_reference_sha256"],
        )
        full, _ = request_tokens(json.loads(args.chat_request.read_bytes()), tokenizer, engine.max_context, chat=True)
        run(
            engine,
            [("2K", exact_prompt(seed, 2048)), ("8K", exact_prompt(seed, 8192)), ("full-code", full)],
            output=engine.output / "cold-benchmark.json",
            windows=args.windows,
        )
    finally:
        engine.close()


if __name__ == "__main__":
    main()
