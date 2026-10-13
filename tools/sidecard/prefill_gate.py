# SPDX-License-Identifier: Apache-2.0
"""Compare optimized prefill against the qualified 32-row arithmetic path."""

import hashlib
import json

from .layer_weights import MAX_CHUNK
from .prefill_benchmark import exact_prompt

PARITY_PROMPT_TOKENS = 2048
PARITY_OUTPUT_TOKENS = 64


def validate(engine, seed, *, output=None, reference_path=None, reference_sha256=None):
    """Restore runtime choices even when a real-weight numerical gate fails."""
    original_chunk = getattr(engine, "prefill_chunk_size", MAX_CHUNK)
    original_flags = {index: layer.vectorized_prefill_conv for index, layer in engine.layers.items()}
    original_fingerprints = engine.fingerprints
    length = min(PARITY_PROMPT_TOKENS, engine.max_context - PARITY_OUTPUT_TOKENS)
    if length < 1:
        raise ValueError("prefill parity requires room for prompt and decode")
    prompt = exact_prompt(seed, length)
    traces, logits_traces, forced = [], [], []
    try:
        for optimized in (False, True):
            engine.prefill_chunk_size = original_chunk if optimized else MAX_CHUNK
            for index, layer in engine.layers.items():
                layer.vectorized_prefill_conv = original_flags[index] if optimized else False
            engine.reset(context_limit=length + PARITY_OUTPUT_TOKENS)
            engine.fingerprints = []
            logits = engine.prefill(prompt)
            observed = [logits.cpu()]
            for step in range(PARITY_OUTPUT_TOKENS):
                if not optimized:
                    forced.append(int(logits.argmax().cpu()))
                if step < PARITY_OUTPUT_TOKENS - 1:
                    logits = engine.forward([forced[step]])
                    observed.append(logits.cpu())
            traces.append(engine.fingerprints)
            logits_traces.append(observed)
        all_boundaries = original_chunk == MAX_CHUNK
        compared = traces if all_boundaries else [trace[-PARITY_OUTPUT_TOKENS:] for trace in traces]
        hashes = [[row["logits_sha256"] for row in trace] for trace in compared]
        equal = hashes[0] == hashes[1]
        max_error = max(float((a - b).abs().max()) for a, b in zip(*logits_traces, strict=True))
        reference_equal = None
        if reference_path is not None:
            body = reference_path.read_bytes()
            if hashlib.sha256(body).hexdigest() != reference_sha256:
                raise ValueError("qualified prefill reference changed")
            reference = json.loads(body)
            reference_equal = (
                reference["passed"]
                and reference["teacher_forced_tokens"] == forced
                and reference["traces"][1] == traces[1]
            )
            equal = equal and reference_equal
        report = dict(
            passed=equal,
            full_vocabulary_bitwise_equal=equal,
            max_abs_logit_error=max_error,
            baseline_chunk_rows=MAX_CHUNK,
            candidate_chunk_rows=original_chunk,
            all_prefill_boundaries_compared=all_boundaries,
            vectorized_convolution=any(original_flags.values()),
            prompt_tokens=length,
            output_tokens=PARITY_OUTPUT_TOKENS,
            teacher_forced_tokens=forced,
            traces=traces,
            reference_bitwise_equal=reference_equal,
        )
        if output is not None:
            output.write_text(json.dumps(report, indent=2) + "\n")
        if not equal:
            raise AssertionError("optimized prefill changed qualified full-vocabulary logits")
        return report
    finally:
        engine.prefill_chunk_size = original_chunk
        engine.fingerprints = original_fingerprints
        for index, layer in engine.layers.items():
            layer.vectorized_prefill_conv = original_flags[index]
