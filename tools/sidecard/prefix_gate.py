# SPDX-License-Identifier: Apache-2.0
"""Qualify recurrent prefix restoration against uncached full-vocabulary logits."""

import hashlib
import json
import time

from .layer_prefix import PromptPrefix

PROMPT_TOKENS = 2048
DECODE_TOKENS = 64
SUFFIX_TOKENS = 17
MUTATION_TOKENS = 20
INTERLEAVED_TOKENS = 64
SECOND_INTERLEAVED_TOKENS = 32


def validate(engine, seed, *, output):
    prompt = (seed * (PROMPT_TOKENS // len(seed) + 1))[:PROMPT_TOKENS]
    extended = prompt + (seed * (SUFFIX_TOKENS // len(seed) + 1))[:SUFFIX_TOKENS]
    cache = PromptPrefix(engine)
    traces, times, forced = [], [], []
    with engine.torch.inference_mode():
        cache.prefill(prompt, engine.max_context)
        for token in (seed * (MUTATION_TOKENS // len(seed) + 1))[:MUTATION_TOKENS]:
            engine.forward([token])
        # Overwrite the live KV/pool prefix with an unrelated short request.
        unrelated = (seed[::-1] * (INTERLEAVED_TOKENS // len(seed) + 1))[:INTERLEAVED_TOKENS]
        cache.prefill(unrelated, engine.max_context)
        cache.prefill(unrelated[:SECOND_INTERLEAVED_TOKENS], engine.max_context)
        for trial in range(2):
            if trial:
                cache = PromptPrefix(engine)
            begin = time.perf_counter()
            logits, cached = cache.prefill(extended, engine.max_context)
            engine.torch.cuda.synchronize()
            times.append({"prefill_seconds": time.perf_counter() - begin, "cached_tokens": cached})
            trace = []
            for step in range(DECODE_TOKENS):
                trace.append(hashlib.sha256(logits.cpu().numpy().tobytes()).hexdigest())
                token = int(logits.argmax().cpu()) if trial == 0 else forced[step]
                if trial == 0:
                    forced.append(token)
                if step < DECODE_TOKENS - 1:
                    logits = engine.forward([token])
            traces.append(trace)
    receipt = {
        "passed": traces[0] == traces[1] and times[0]["cached_tokens"] == PROMPT_TOKENS,
        "bitwise_equal": traces[0] == traces[1],
        "prompt_tokens": len(extended),
        "teacher_forced_steps": DECODE_TOKENS,
        "interleaved_prompt_tokens": [INTERLEAVED_TOKENS, SECOND_INTERLEAVED_TOKENS],
        "reused": times[0],
        "cold": times[1],
        "logit_sha256": traces,
    }
    output.write_text(json.dumps(receipt, indent=2) + "\n")
    if not receipt["passed"]:
        raise RuntimeError("prompt prefix restoration changed full-vocabulary logits")
    return receipt
