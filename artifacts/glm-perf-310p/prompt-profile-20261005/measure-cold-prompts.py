# SPDX-License-Identifier: Apache-2.0
"""Measure uncached prompt latency through the live completion API."""

import argparse
import hashlib
import json
import time
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path


def request(base, path, payload=None):
    call = urllib.request.Request(
        base + path,
        data=json.dumps(payload).encode() if payload is not None else None,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(call, timeout=900) as response:
        content = response.read().decode()
    return json.loads(content) if payload is not None else content


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("label")
    parser.add_argument("--lengths", type=int, nargs="+", default=[8192, 20643])
    parser.add_argument("--paired-with", help="reuse nonces and text from this saved baseline label")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    base = "http://127.0.0.1:8001"
    model = "glm53-flash-selective-w3"
    status = request(base, "/collective_rpc", {"method": "resident_status", "args": [], "timeout": 120})["results"]
    assert len(status) == 4 and all(
        row["candidate"] == "completed_pools_sinkhorn" and row["graphs_dirty"] is False and not row.get("native_failed")
        for row in status
    ), status
    (root / f"cold-{args.label}-worker-status.json").write_text(json.dumps(status, indent=2) + "\n")
    records = []
    paired = (
        {row["input_tokens"]: row for row in json.loads((root / f"cold-{args.paired_with}.json").read_text())}
        if args.paired_with
        else {}
    )
    for length in args.lengths:
        nonce = paired[length]["nonce"] if paired else uuid.uuid4().hex
        text = f"# Cold coding benchmark {nonce}\nReview this Python module and suggest a correction.\n"
        text += "\n".join(
            f"def update_record_{index}(records, key, value):\n"
            "    previous = records.get(key)\n"
            "    if previous is None:\n"
            "        records[key] = {'value': value, 'revision': 1}\n"
            "    else:\n"
            "        previous['value'] = value\n"
            "        previous['revision'] += 1\n"
            "    return records[key]\n"
            for index in range(1200)
        )
        tokens = request(base, "/tokenize", {"model": model, "prompt": text})["tokens"][:length]
        assert len(tokens) == length
        token_digest = hashlib.sha256(json.dumps(tokens).encode()).hexdigest()
        before = request(base, "/metrics")
        (root / f"cold-{args.label}-{length}-before.txt").write_text(before)
        payload = {
            "model": model,
            "prompt": tokens,
            "max_tokens": 1,
            "temperature": 0,
            "ignore_eos": True,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        call = urllib.request.Request(
            base + "/v1/completions", data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}
        )
        record = {
            "label": args.label,
            "input_tokens": length,
            "nonce": nonce,
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "paired_with": args.paired_with,
            "token_ids_sha256": token_digest,
        }
        print(json.dumps(record), flush=True)
        start = time.perf_counter()
        first = None
        chunks = []
        with urllib.request.urlopen(call, timeout=900) as response:
            for line in response:
                if not line.startswith(b"data: ") or line.strip() == b"data: [DONE]":
                    continue
                chunk = json.loads(line[6:])
                chunks.append(chunk)
                if first is None and any(choice.get("text") for choice in chunk.get("choices", [])):
                    first = time.perf_counter() - start
        elapsed = time.perf_counter() - start
        assert first is not None, chunks
        usage = next((chunk["usage"] for chunk in reversed(chunks) if chunk.get("usage")), None)
        assert usage is not None and usage["prompt_tokens"] == length, usage
        record.update(
            {
                "ttft_s": first,
                "elapsed_s": elapsed,
                "input_tokens_per_ttft_second": length / first,
                "usage": usage,
                "chunks": chunks,
            }
        )
        records.append(record)
        (root / f"cold-{args.label}-{length}-after.txt").write_text(request(base, "/metrics"))
        (root / f"cold-{args.label}.json").write_text(json.dumps(records, indent=2) + "\n")
        print(json.dumps({key: value for key, value in record.items() if key != "chunks"}), flush=True)


if __name__ == "__main__":
    main()
