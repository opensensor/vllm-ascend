# SPDX-License-Identifier: Apache-2.0
"""Matched complete-layer socket/DMA experiment with full vocabulary checks.

Run each specification in its own process so NPU graph and collective owners
have an unambiguous lifetime. The reference is another complete-layer run;
expert-only and TP2 backbone arithmetic are not assumed bitwise equivalent.
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

from .indexer_benchmark import decode
from .layer_actor import Coordinator
from .prefill_benchmark import exact_prompt


def run(config, reference=None):
    from transformers import AutoTokenizer

    spec = json.loads(config.read_bytes())
    report = dict(
        passed=False,
        architecture="cuda-prefix-ascend-middle-cuda-suffix",
        measurements=[],
        configured_max_context=spec["max_context"],
        full_context_boundary_tested=False,
        task_accuracy_evaluated=False,
        transport=spec.get("stage_transport", "socket"),
        checkpoint_sha256=spec["checkpoint_sha256"],
        prefill_overlap_used=spec.get("stage_transport", "socket") == "socket",
        npu_native_bf16_round=spec.get("npu_native_bf16_round", False),
        npu_decode_graphs=spec.get("npu_decode_graphs", False),
    )
    golden = json.loads(reference.read_bytes()) if reference is not None else None
    if golden is not None and not golden.get("passed"):
        raise ValueError("complete-stage reference did not pass")
    if reference is not None:
        report["reference_sha256"] = hashlib.sha256(reference.read_bytes()).hexdigest()
    output = Path(spec["output"]) / "stage-benchmark.json"
    engine = Coordinator(config)
    try:
        tokenizer = AutoTokenizer.from_pretrained(spec["checkpoint"], local_files_only=True, trust_remote_code=False)
        seed = tokenizer.encode("Inspect this module and explain its control flow. ")
        for length, repeats in ((2048, 2), (8192, 1)):
            prompt = exact_prompt(seed, length)
            expected = next((r for r in (golden or {}).get("measurements", []) if r["prompt_tokens"] == length), None)
            forced, reference_hashes = None, None
            if expected is not None:
                forced, reference_hashes = expected["generated_ids"], expected["logits_sha256"]
            for repeat in range(repeats):
                engine.reset(context_limit=length + 64)
                engine.metrics = {name: [] for name in engine.metrics}
                begin = time.perf_counter()
                logits = engine.prefill(prompt)
                prefill_seconds = time.perf_counter() - begin
                hashes, ids, spans = decode(engine, logits, forced=forced)
                compared = reference_hashes is not None
                if compared and hashes != reference_hashes:
                    raise AssertionError("complete-stage vocabulary differs from its matched reference")
                forced, reference_hashes = ids, hashes
                row = dict(
                    prompt_tokens=length,
                    output_tokens=64,
                    repeat=repeat,
                    cached_tokens=0,
                    prompt_ids_sha256=hashlib.sha256(json.dumps(prompt, separators=(",", ":")).encode()).hexdigest(),
                    prefill_seconds=prefill_seconds,
                    decode_tokens_per_second=len(spans) / sum(spans),
                    decode_forward_seconds=spans,
                    generated_ids=ids,
                    text=tokenizer.decode(ids),
                    logits_sha256=hashes,
                    repeated_or_reference_bitwise_equal=True if compared else None,
                    decode_components_ms={name: values[-63:] for name, values in engine.metrics.items()},
                )
                report["measurements"].append(row)
                output.write_text(json.dumps(report, indent=2) + "\n")
                print(
                    json.dumps(
                        {
                            key: row[key]
                            for key in ("prompt_tokens", "repeat", "prefill_seconds", "decode_tokens_per_second")
                        }
                    ),
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
    parser.add_argument("--reference", type=Path)
    args = parser.parse_args()
    run(args.config, args.reference)


if __name__ == "__main__":
    main()
