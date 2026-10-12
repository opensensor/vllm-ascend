# SPDX-License-Identifier: Apache-2.0
"""Run bounded real-model stage attribution, retaining request boundaries.

Use a private diagnostic snapshot with stage_profile enabled. Never interpret
the synchronized run's token rate as production performance. The caller owns
service restoration; this driver closes only its own accelerator actors.
"""

import argparse
import json
from pathlib import Path

from .layer_actor import Coordinator
from .layer_server import Generation
from .prefill_benchmark import exact_prompt

WORKLOADS = (
    ("matched2048", "Inspect this module for profiler test 20261011 and explain the control flow. ", 2048),
    ("python", "Write a Python function that validates a JSON configuration and reports all missing fields.", None),
    ("reasoning", "Explain why a distributed database needs consensus and compare Raft with Paxos.", None),
    ("arithmetic", "Work through 37 times 24, then explain a way to check the answer.", None),
)
OUTPUT_TOKENS = 64


def run(config_path, reference):
    from transformers import AutoTokenizer

    spec = json.loads(Path(config_path).read_bytes())
    if not spec.get("stage_profile") or spec.get("npu_decode_graphs"):
        raise ValueError("diagnostic driver requires eager Ascend stage attribution")
    output = Path(spec["output"])
    engine = None
    report = dict(passed=False, production_speed_validated=False, workloads=[])
    try:
        engine = Coordinator(config_path)
        tokenizer = AutoTokenizer.from_pretrained(spec["checkpoint"], trust_remote_code=False)
        expected = json.loads(Path(reference).read_bytes())
        prompt = tokenizer.encode('def factorial(n):\n    """Compute factorial recursively."""\n')
        engine.reset(context_limit=len(prompt) + OUTPUT_TOKENS)
        engine.fingerprints = []
        engine.prefill(prompt)
        forced = expected["teacher_forced_tokens"]
        for token in forced[: OUTPUT_TOKENS - 1]:
            engine.forward([token])
        report["unprofiled_reference_bitwise_equal"] = engine.fingerprints == expected["traces"][0]
        engine.fingerprints = None
        if not report["unprofiled_reference_bitwise_equal"]:
            raise AssertionError("synchronized profiling changed complete handoff or vocabulary bytes")
        generator = Generation(engine, tokenizer)
        for name, text, length in WORKLOADS:
            tokens = tokenizer.encode(text, add_special_tokens=False)
            if length is not None:
                tokens = exact_prompt(tokens, length)
            first_sequence = engine.sequence
            result = {}
            list(generator.generate(tokens, OUTPUT_TOKENS, result, ignore_eos=True))
            report["workloads"].append(
                dict(
                    name=name,
                    first_sequence=first_sequence,
                    end_sequence=engine.sequence,
                    prompt_tokens=len(tokens),
                    output_ids=result["output_ids"],
                    text=result["text"],
                )
            )
            (output / "profile-run.json").write_text(json.dumps(report, indent=2) + "\n")
        report["passed"] = True
    except Exception as error:
        report["error"] = dict(type=type(error).__name__, message=str(error))
        raise
    finally:
        (output / "profile-run.json").write_text(json.dumps(report, indent=2) + "\n")
        if engine is not None:
            engine.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    args = parser.parse_args()
    run(args.config, args.reference)


if __name__ == "__main__":
    main()
