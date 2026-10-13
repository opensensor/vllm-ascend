# SPDX-License-Identifier: Apache-2.0
"""Alternate eager and captured indexer decode from identical prompt state."""

import argparse
import hashlib
import json
import time
from contextlib import contextmanager
from pathlib import Path

from .expert_actor import ExpertCoordinator
from .indexer_benchmark import OUTPUT_TOKENS, decode
from .indexer_projection_graph import enabled
from .layer_prefix import PromptPrefix
from .layer_server import request_tokens
from .prefill_benchmark import exact_prompt


@contextmanager
def graph_mode(engine, selected):
    if type(selected) is not bool:
        raise ValueError("indexer projection mode requires a boolean")
    indexers = [layer.indexer for layer in engine.layers.values() if not layer.linear]
    if any(indexer.projection_graph is None for indexer in indexers):
        raise ValueError("projection benchmark requires every CUDA graph to be captured")
    previous = [indexer.projection_graph_enabled for indexer in indexers]
    try:
        for indexer in indexers:
            indexer.projection_graph_enabled = selected
        yield
    finally:
        for indexer, value in zip(indexers, previous, strict=True):
            indexer.projection_graph_enabled = value


def run(config, chat_request, reference_path, reference_sha256):
    """Record cold prefill once; alternate decode from byte-identical snapshots."""
    from transformers import AutoTokenizer

    spec = json.loads(config.read_bytes())
    raw = reference_path.read_bytes()
    qualified = json.loads(raw)
    if hashlib.sha256(raw).hexdigest() != reference_sha256 or not qualified.get("passed"):
        raise ValueError("projection benchmark reference failed authentication or qualification")
    if not enabled(spec) or spec.get("direct_dma", {}).get("npu_single_route_graphs", False):
        raise ValueError("projection benchmark requires captured CUDA projections and unchanged Ascend math")
    output = Path(spec["output"]) / "projection-benchmark.json"
    report = dict(passed=False, cold_prefills=[], measurements=[], task_accuracy_evaluated=False)
    engine = ExpertCoordinator(config)
    try:
        tokenizer = AutoTokenizer.from_pretrained(spec["checkpoint"], local_files_only=True, trust_remote_code=False)
        seed = tokenizer.encode("Inspect this module and explain its control flow. ")
        full, _ = request_tokens(json.loads(chat_request.read_bytes()), tokenizer, engine.max_context, chat=True)
        for name, prompt in (("2K", exact_prompt(seed, 2048)), ("8K", exact_prompt(seed, 8192)), ("full-code", full)):
            historical = next(row for row in qualified["measurements"] if row["case"] == name and row["fused"])
            signature = hashlib.sha256(json.dumps(prompt, separators=(",", ":")).encode()).hexdigest()
            if signature != historical["prompt_ids_sha256"]:
                raise AssertionError("projection benchmark changed the complete prompt")
            with graph_mode(engine, False):
                engine.reset(context_limit=len(prompt) + OUTPUT_TOKENS)
                begin = time.perf_counter()
                logits = engine.prefill(prompt)
                cold_seconds = time.perf_counter() - begin
            snapshots = PromptPrefix(engine)
            snapshot = snapshots.snapshot(prompt, logits, True)
            report["cold_prefills"].append(dict(case=name, cached_tokens=0, prefill_seconds=cold_seconds))
            forced = None
            for repeat in range(2):
                for selected in (False, True):
                    with graph_mode(engine, selected):
                        engine.reset(context_limit=len(prompt) + OUTPUT_TOKENS)
                        restored = snapshots.restore(snapshot)
                        hashes, forced, spans = decode(engine, restored, forced=forced)
                        if hashes != historical["logits_sha256"]:
                            raise AssertionError("captured projections changed full-vocabulary model logits")
                        row = dict(
                            case=name,
                            repeat=repeat,
                            captured=selected,
                            prompt_tokens=len(prompt),
                            prompt_ids_sha256=signature,
                            decode_tokens_per_second=len(spans) / sum(spans),
                            decode_forward_seconds=spans,
                            full_vocabulary_bitwise_equal=True,
                            logits_sha256=hashes,
                            decode_state_source="identical GPU prompt snapshot",
                        )
                        report["measurements"].append(row)
                        output.write_text(json.dumps(report, indent=2) + "\n")
                        print(
                            json.dumps(
                                {k: v for k, v in row.items() if k not in ("logits_sha256", "decode_forward_seconds")}
                            ),
                            flush=True,
                        )
            del snapshot, snapshots
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
