"""Record the live QFN API and verify a draft reload without replacing workers."""

import argparse
import json
import statistics
import time
import uuid
from pathlib import Path

from tools.glm_perf.resident_control import Control
from tools.sidecard.qfn_resident_server import QfnResidentClient


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("status", "benchmark", "draft"))
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    client = QfnResidentClient("http://127.0.0.1:8001", args.run_root / "control.token")
    result = {"action": args.action, "before": client.rpc("resident_status")}
    args.receipt.write_text(json.dumps(result, indent=2) + "\n")
    if args.action == "draft":
        control = Control(
            generation=uuid.uuid4().hex,
            candidate="draft_partition",
            source=(
                "from tools.qwen4exp.resident_candidates.draft_graph_experts import replacements as draft\n"
                "from tools.qwen4exp.resident_candidates.qfn_reset_barrier import replacements as barrier\n"
                "from tools.qwen4exp.resident_candidates.query_lens_capacity import replacements as capacity\n"
                "def replacements():\n"
                "    return {**draft(), **barrier(), **capacity()}\n"
            ),
            recapture=True,
        )
        started = time.monotonic()
        result["after"] = client.switch(control)
        result["switch_seconds"] = time.monotonic() - started
    elif args.action == "benchmark":
        result["trials"] = []
        for length in (64, 512, 512, 512):
            payload = {
                "model": "qwen38-flash-next",
                "messages": [
                    {
                        "role": "user",
                        "content": "Explain how a hash table works and discuss collision resolution strategies.",
                    }
                ],
                "chat_template_kwargs": {"enable_thinking": False},
                "temperature": 0,
                "max_tokens": length,
                "min_tokens": length,
                "ignore_eos": True,
            }
            started = time.monotonic()
            response = client.request("/v1/chat/completions", payload)
            elapsed = time.monotonic() - started
            tokens = response["usage"]["completion_tokens"]
            if tokens != length:
                raise RuntimeError(f"expected {length} completion tokens, received {tokens}")
            trial = {
                "seconds": elapsed,
                "output_tokens": tokens,
                "tokens_per_second": tokens / elapsed,
                "response": response,
            }
            result["trials"].append(trial)
            args.receipt.write_text(json.dumps(result, indent=2) + "\n")
            print(json.dumps({key: value for key, value in trial.items() if key != "response"}), flush=True)
        result["median_tokens_per_second"] = statistics.median(
            trial["tokens_per_second"] for trial in result["trials"][1:]
        )
        result["after"] = client.rpc("resident_status")
    args.receipt.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "receipt": str(args.receipt),
                "workers": [
                    {key: worker.get(key) for key in ("rank", "pid", "candidate", "graphs_dirty", "native_failed")}
                    for worker in result.get("after", result["before"])
                ],
                "switch_seconds": result.get("switch_seconds"),
                "median_tokens_per_second": result.get("median_tokens_per_second"),
            }
        )
    )


if __name__ == "__main__":
    main()
