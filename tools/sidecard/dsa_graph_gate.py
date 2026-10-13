# SPDX-License-Identifier: Apache-2.0
"""Compare captured DSA math with eager execution across sparse admission."""

import json

from .prefill_benchmark import exact_prompt

PARITY_PROMPT_TOKENS = 2048
PARITY_OUTPUT_TOKENS = 8


def validate(engine, seed, *, output=None):
    stages = (engine.prefix_graphs, engine.middle_graphs, engine.suffix_graphs)
    saved = tuple(stage.dsa_graphs for stage in stages)
    fingerprints = engine.fingerprints
    length = min(PARITY_PROMPT_TOKENS, engine.max_context - PARITY_OUTPUT_TOKENS)
    if length < 1 or not all(saved):
        raise ValueError("DSA parity requires captured stages and room for prompt and decode")
    prompt = exact_prompt(seed, length)
    traces, forced = [], []
    try:
        for captured in (False, True):
            for stage, graphs in zip(stages, saved, strict=True):
                stage.dsa_graphs = graphs if captured else {}
            engine.reset(context_limit=length + PARITY_OUTPUT_TOKENS)
            engine.fingerprints = []
            logits = engine.prefill(prompt)
            for step in range(PARITY_OUTPUT_TOKENS):
                if not captured:
                    forced.append(int(logits.argmax().cpu()))
                if step < PARITY_OUTPUT_TOKENS - 1:
                    logits = engine.forward([forced[step]])
            traces.append(engine.fingerprints)
        report = dict(
            passed=traces[0] == traces[1],
            full_vocabulary_bitwise_equal=traces[0] == traces[1],
            prompt_tokens=length,
            output_tokens=PARITY_OUTPUT_TOKENS,
            sparse_indexer_exercised=length + PARITY_OUTPUT_TOKENS - 1 > engine.config["index_topk"],
            captured_layers=sorted(index for graphs in saved for index in graphs),
            teacher_forced_tokens=forced,
            traces=traces,
        )
        if output is not None:
            output.write_text(json.dumps(report, indent=2) + "\n")
        if not report["passed"]:
            raise AssertionError("DSA graph changed eager full-vocabulary logits")
        return report
    finally:
        for stage, graphs in zip(stages, saved, strict=True):
            stage.dsa_graphs = graphs
        engine.fingerprints = fingerprints
