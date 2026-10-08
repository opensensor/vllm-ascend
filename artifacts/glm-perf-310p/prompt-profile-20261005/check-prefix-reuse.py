# SPDX-License-Identifier: Apache-2.0
"""Repeat the completed 20K request and verify measured prefix reuse."""

import hashlib
import importlib.util
import json
import re
import time
import urllib.request
from pathlib import Path


def counter(metrics, name):
    values = [float(line.rsplit(" ", 1)[1]) for line in metrics.splitlines() if re.match(re.escape(name) + r"\{", line)]
    assert len(values) == 1, name
    return values[0]


def main():
    root = Path(__file__).resolve().parent
    spec = importlib.util.spec_from_file_location("glm_cold_prompt_benchmark", root / "measure-cold-prompts.py")
    benchmark = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(benchmark)
    request = benchmark.request
    base = "http://127.0.0.1:8001"
    row = next(row for row in json.loads((root / "cold-score-cache.json").read_text()) if row["input_tokens"] == 20643)
    # Reconstruct exactly the same text used by the cold measurements.
    text = f"# Cold coding benchmark {row['nonce']}\nReview this Python module and suggest a correction.\n"
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
    tokens = request(base, "/tokenize", {"model": "glm53-flash-selective-w3", "prompt": text})["tokens"][:20643]
    assert hashlib.sha256(json.dumps(tokens).encode()).hexdigest() == row["token_ids_sha256"]
    before = request(base, "/metrics")
    call = urllib.request.Request(
        base + "/v1/completions",
        data=json.dumps(
            {
                "model": "glm53-flash-selective-w3",
                "prompt": tokens,
                "max_tokens": 1,
                "temperature": 0,
                "ignore_eos": True,
            }
        ).encode(),
        headers={"Content-Type": "application/json"},
    )
    start = time.perf_counter()
    with urllib.request.urlopen(call, timeout=120) as response:
        completion = json.loads(response.read())
    elapsed = time.perf_counter() - start
    after = request(base, "/metrics")
    hits = counter(after, "vllm:prefix_cache_hits_total") - counter(before, "vllm:prefix_cache_hits_total")
    result = {
        "elapsed_s": elapsed,
        "input_tokens": 20643,
        "cached_tokens": hits,
        "computed_tokens": 20643 - hits,
        "completion": completion,
    }
    (root / "prefix-reuse.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)
    assert hits >= 20480 and completion["usage"]["completion_tokens"] == 1, result


if __name__ == "__main__":
    main()
