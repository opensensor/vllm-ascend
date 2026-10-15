"""Measure exact-length uncached requests against the persistent endpoint."""

import argparse
import json
import statistics
import time
import uuid
from pathlib import Path

from tools.sidecard.qfn_resident_server import QfnResidentClient

ROOT = Path("/srv/ai/storage/artifacts/npu-collective-qfn-20261014-024609")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--length", type=int, default=8192)
    args = parser.parse_args()
    client = QfnResidentClient("http://127.0.0.1:8001", ROOT / "resident-live-v3/control.token")
    output = ROOT / "qsa-resident-v1" / f"cold-benchmark-{args.label}.json"
    result = {"label": args.label, "before": client.rpc("resident_status"), "trials": []}
    filler = client.request(
        "/tokenize",
        {
            "model": "qwen38-flash-next",
            "prompt": "A hash table stores keys and values in buckets. ",
            "add_special_tokens": False,
        },
    )["tokens"]
    for index in range(args.trials):
        base = client.request(
            "/tokenize",
            {
                "model": "qwen38-flash-next",
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            f"Unique benchmark {uuid.uuid4().hex}. Explain why a hash table needs collision resolution."
                        ),
                    }
                ],
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )["tokens"]
        # Distinct early nonce prevents reused prefix blocks; leave template end intact.
        padding = args.length - len(base)
        prompt = base[:-8] + (filler * ((padding + len(filler) - 1) // len(filler)))[:padding] + base[-8:]
        started = time.monotonic()
        response = client.request(
            "/v1/completions",
            dict(model="qwen38-flash-next", prompt=prompt, temperature=0, max_tokens=1, min_tokens=1, ignore_eos=True),
        )
        seconds = time.monotonic() - started
        usage = response["usage"]
        assert usage["prompt_tokens"] == args.length and usage["completion_tokens"] == 1, usage
        assert usage["prompt_tokens_details"]["cached_tokens"] == 0, usage
        row = dict(trial=index, seconds=seconds, prompt_tokens_per_second=args.length / seconds, response=response)
        result["trials"].append(row)
        output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps({key: value for key, value in row.items() if key != "response"}), flush=True)
    result["median_seconds"] = statistics.median(row["seconds"] for row in result["trials"])
    result["median_prompt_tokens_per_second"] = args.length / result["median_seconds"]
    result["after"] = client.rpc("resident_status")
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            dict(
                label=args.label,
                median_seconds=result["median_seconds"],
                median_prompt_tokens_per_second=result["median_prompt_tokens_per_second"],
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
