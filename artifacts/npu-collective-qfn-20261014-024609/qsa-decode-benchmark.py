"""Measure fixed-budget single-request output throughput on the resident server."""

import argparse
import json
import statistics
import time
from pathlib import Path

from tools.sidecard.qfn_resident_server import QfnResidentClient

ROOT = Path("/srv/ai/storage/artifacts/npu-collective-qfn-20261014-024609")
PROMPT = "Explain how a hash table works and discuss collision resolution strategies."


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    parser.add_argument("--run-root", type=Path, default=ROOT / "resident-live-v3")
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--tokens", type=int, default=256)
    args = parser.parse_args()
    output = ROOT / "qsa-resident-v1" / f"decode-benchmark-{args.label}.json"
    if output.exists() or not 1 <= args.trials <= 5 or not 64 <= args.tokens <= 512:
        raise ValueError("use a new label, 1..5 trials, and 64..512 output tokens")
    client = QfnResidentClient("http://127.0.0.1:8001", args.run_root / "control.token")
    request = dict(
        model="qwen38-flash-next",
        messages=[dict(role="user", content=PROMPT)],
        chat_template_kwargs={"enable_thinking": False},
        temperature=0,
        max_tokens=args.tokens,
        min_tokens=args.tokens,
        ignore_eos=True,
    )
    result = dict(
        before=client.rpc("resident_status"),
        request=request,
        scope="C1 HTTP output tokens/s includes short prefill and API overhead; not pure device TPOT",
        trials=[],
    )
    for trial in range(args.trials):
        start = time.monotonic()
        response = client.request("/v1/chat/completions", request)
        seconds = time.monotonic() - start
        assert response["usage"]["completion_tokens"] == args.tokens
        row = dict(trial=trial, seconds=seconds, output_tokens_per_second=args.tokens / seconds, response=response)
        result["trials"].append(row)
        output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps({k: v for k, v in row.items() if k != "response"}), flush=True)
    result["after"] = client.rpc("resident_status")
    identity = lambda rows: sorted((r["rank"], r["pid"], r["weight_storage_digest"]) for r in rows)
    assert identity(result["before"]) == identity(result["after"])
    assert all(not r["graphs_dirty"] and not r["native_failed"] for r in result["after"])
    result["median_output_tokens_per_second"] = statistics.median(
        r["output_tokens_per_second"] for r in result["trials"]
    )
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(dict(label=args.label, median_output_tokens_per_second=result["median_output_tokens_per_second"])),
        flush=True,
    )


if __name__ == "__main__":
    main()
