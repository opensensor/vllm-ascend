# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real-weight long-context retrieval and cached-prefix gates for YaRN.

Use only during an authorized serving maintenance window. Background is
public synthetic text; receipts contain no private conversation payloads.
"""

import argparse
import json
import time
import urllib.request
from pathlib import Path

from tokenizers import Tokenizer

CONTEXT_STAGES = (280000, 550000, 1000000)
ARCHIVE_KEYS = ((100000, "ALPHA", "ORCHID-2718"), (500000, "BETA", "COPPER-3141"), (900000, "GAMMA", "LANTERN-1618"))
BACKGROUND = "Archive background record. This entry describes routine maintenance of a garden and its storage shed.\n"


def request(base, path, body=None, timeout=300):
    req = urllib.request.Request(
        base + path,
        data=None if body is None else json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.load(response)


def probe(base, model, content, max_tokens=128):
    body = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "temperature": 0,
        "seed": 42,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    req = urllib.request.Request(
        base + "/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
    )
    start = time.monotonic()
    first = None
    text = ""
    usage = None
    finish = None
    done = False
    with urllib.request.urlopen(req, timeout=14400) as response:
        for line in response:
            if not line.startswith(b"data: "):
                continue
            payload = line[6:].strip()
            if payload == b"[DONE]":
                done = True
                break
            event = json.loads(payload)
            if event.get("error"):
                raise RuntimeError(event["error"])
            usage = event.get("usage") or usage
            for choice in event.get("choices", []):
                delta = choice.get("delta", {})
                if delta.get("content"):
                    if first is None:
                        first = time.monotonic() - start
                    text += delta["content"]
                finish = choice.get("finish_reason") or finish
    if not done or not usage or not finish:
        raise RuntimeError("incomplete completion stream")
    return {
        "elapsed_s": time.monotonic() - start,
        "first_token_s": first,
        "usage": usage,
        "text": text,
        "finish_reason": finish,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8001")
    parser.add_argument("--model", default="qwen38-flash-next")
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    unit = tokenizer.encode(BACKGROUND, add_special_tokens=False).ids
    tokens = (unit * (CONTEXT_STAGES[-1] // len(unit) + 1))[: CONTEXT_STAGES[-1]]
    for position, name, value in ARCHIVE_KEYS:
        key = tokenizer.encode(
            f"\nIMPORTANT: The archive key for section {name} is {value}.\n", add_special_tokens=False
        ).ids
        tokens[position : position + len(key)] = key
    results = []
    request(args.base_url, "/resident/maintenance", {})
    for length in CONTEXT_STAGES:
        background = tokenizer.decode(tokens[:length])
        question = "\nList the exact archive keys explicitly recorded for ALPHA, BETA, and GAMMA. Omit absent sections."
        print(json.dumps({"starting_background_tokens": length}), flush=True)
        result = probe(args.base_url, args.model, background + question)
        expected = [value for position, _, value in ARCHIVE_KEYS if position < length]
        result.update(
            background_tokens=length,
            phase="extension",
            expected=expected,
            passed=all(value in result["text"] for value in expected),
        )
        results.append(result)
        args.output.write_text(json.dumps({"passed": False, "requests": results}, indent=2) + "\n")
        print(json.dumps(result), flush=True)
        if not result["passed"]:
            raise AssertionError(f"retrieval failed at {length} tokens")
    repeat = probe(args.base_url, args.model, background + question)
    repeat.update(phase="cached-repeat", passed=all(value in repeat["text"] for value in expected))
    cached = repeat["usage"].get("prompt_tokens_details", {}).get("cached_tokens", 0)
    if cached < CONTEXT_STAGES[-1] - 1024 or not repeat["passed"]:
        raise AssertionError(f"million-token repeat lost cache or retrieval: {repeat}")
    results.append(repeat)
    args.output.write_text(json.dumps({"passed": True, "requests": results}, indent=2) + "\n")
    print(json.dumps(repeat), flush=True)


if __name__ == "__main__":
    main()
