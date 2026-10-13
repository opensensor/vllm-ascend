# SPDX-License-Identifier: Apache-2.0
"""Qualify unpadded decode against authenticated, complete GLM vocabularies."""

import argparse
import hashlib
import json
import time
from pathlib import Path

from .expert_actor import ExpertCoordinator
from .indexer_benchmark import OUTPUT_TOKENS, decode
from .layer_server import request_tokens
from .prefill_benchmark import exact_prompt
from .single_route_graph import single_route_enabled


def run(config, chat_request, reference_path, reference_sha256):
    from transformers import AutoTokenizer

    spec = json.loads(config.read_bytes())
    raw = reference_path.read_bytes()
    qualified = json.loads(raw)
    if hashlib.sha256(raw).hexdigest() != reference_sha256 or not qualified.get("passed"):
        raise ValueError("single-route reference failed authentication or qualification")
    if spec.get("prompt_prefix_cache") or not spec.get("cuda_indexer_hadamard") or not single_route_enabled(spec):
        raise ValueError("single-route benchmark requires cold prompts and qualified CUDA and NPU graphs")
    report = dict(passed=False, reference_sha256=reference_sha256, measurements=[], task_accuracy_evaluated=False)
    output = Path(spec["output"]) / "single-route-benchmark.json"
    engine = ExpertCoordinator(config)
    try:
        tokenizer = AutoTokenizer.from_pretrained(spec["checkpoint"], local_files_only=True, trust_remote_code=False)
        seed = tokenizer.encode("Inspect this module and explain its control flow. ")
        full, _ = request_tokens(json.loads(chat_request.read_bytes()), tokenizer, engine.max_context, chat=True)
        for name, prompt in (("2K", exact_prompt(seed, 2048)), ("8K", exact_prompt(seed, 8192)), ("full-code", full)):
            historical = next(row for row in qualified["measurements"] if row["case"] == name and row["fused"])
            signature = hashlib.sha256(json.dumps(prompt, separators=(",", ":")).encode()).hexdigest()
            if signature != historical["prompt_ids_sha256"]:
                raise AssertionError("single-route benchmark changed the complete prompt")
            engine.reset(context_limit=len(prompt) + OUTPUT_TOKENS)
            begin = time.perf_counter()
            logits = engine.prefill(prompt)
            prefill_seconds = time.perf_counter() - begin
            hashes, _, spans = decode(engine, logits)
            if hashes != historical["logits_sha256"]:
                raise AssertionError("single-route replay changed full-vocabulary model logits")
            row = dict(
                case=name,
                prompt_tokens=len(prompt),
                prompt_ids_sha256=signature,
                cached_tokens=0,
                prefill_seconds=prefill_seconds,
                decode_tokens_per_second=len(spans) / sum(spans),
                decode_forward_seconds=spans,
                historical_decode_tokens_per_second=historical["decode_tokens_per_second"],
                full_vocabulary_bitwise_equal=True,
                logits_sha256=hashes,
            )
            report["measurements"].append(row)
            output.write_text(json.dumps(report, indent=2) + "\n")
            print(
                json.dumps({k: v for k, v in row.items() if k not in ("logits_sha256", "decode_forward_seconds")}),
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
    args = parser.parse_args()
    run(args.config, args.chat_request, args.reference, args.reference_sha256)


if __name__ == "__main__":
    main()
