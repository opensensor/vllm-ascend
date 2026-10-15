# SPDX-License-Identifier: Apache-2.0
"""Compare paired requests using the server's token-gap timings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median

import regex as re

from tools.qwen38_decode_study.speculation import round_estimate

TIMING = re.compile(
    r"request (?P<id>\S+) \([^)]*\):.*?decode (?P<tokens>\d+) tokens, "
    r"(?P<gaps>\d+) gaps in (?P<ms>[\d.]+) ms"
)


def server_timings(text: str) -> dict[str, dict]:
    result = {}
    for match in TIMING.finditer(text):
        elapsed = float(match["ms"])
        if elapsed <= 0:
            continue
        result[match["id"]] = {
            "tokens": int(match["tokens"]),
            "decode_tok_s": 1000 * int(match["gaps"]) / elapsed,
        }
    return result


def read_arm(directory: Path) -> dict[tuple[str, int, int], dict]:
    timings = server_timings((directory / "serve.log").read_text(errors="replace"))
    results = {}
    for context, filename in (("short", "benchmark.jsonl"), ("long", "long-benchmark.jsonl")):
        path = directory / filename
        if not path.exists():
            continue
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if row["warmup"]:
                continue
            timing = timings.get(row["request_id"])
            if timing is None:
                raise ValueError(f"Missing server timing for {row['request_id']}")
            if timing["tokens"] != row["usage"]["completion_tokens"]:
                raise ValueError(f"Server/client token counts differ for {row['request_id']}")
            row["server_decode_tok_s"] = timing["decode_tok_s"]
            results[context, row["length"], row["repeat"]] = row
    return results


def compare(control, candidate) -> dict:
    pairs = []
    for key in sorted(control.keys() & candidate.keys()):
        before, after = control[key], candidate[key]
        if before["request"] != after["request"]:
            raise ValueError(f"Request configuration differs for {key}")
        pairs.append(
            {
                "context": key[0],
                "length": key[1],
                "repeat": key[2],
                "prompt_tokens": after["usage"]["prompt_tokens"],
                "baseline_tok_s": before["server_decode_tok_s"],
                "candidate_tok_s": after["server_decode_tok_s"],
                "speedup": after["server_decode_tok_s"] / before["server_decode_tok_s"],
                "same_text": before["text_sha256"] == after["text_sha256"],
                "baseline_acceptance": before["acceptance"],
                "candidate_acceptance": after["acceptance"],
                "baseline_estimated_step_ms": round_estimate(before, before["num_speculative_tokens"])[
                    "estimated_round_ms"
                ]
                if before.get("num_speculative_tokens") is not None and before.get("drafted_delta", 0) > 0
                else None,
                "candidate_estimated_step_ms": round_estimate(after, after["num_speculative_tokens"])[
                    "estimated_round_ms"
                ]
                if after.get("num_speculative_tokens") is not None and after.get("drafted_delta", 0) > 0
                else None,
            }
        )
    summary = []
    for context, length in sorted({(pair["context"], pair["length"]) for pair in pairs}):
        group = [pair for pair in pairs if (pair["context"], pair["length"]) == (context, length)]
        summary.append(
            {
                "context": context,
                "length": length,
                "n": len(group),
                "baseline_median_tok_s": median(pair["baseline_tok_s"] for pair in group),
                "candidate_median_tok_s": median(pair["candidate_tok_s"] for pair in group),
                "median_paired_speedup": median(pair["speedup"] for pair in group),
                "identical_outputs": sum(pair["same_text"] for pair in group),
            }
        )
    return {
        "matched": len(pairs),
        "missing": len(control.keys() - candidate.keys()),
        "summary": summary,
        "pairs": pairs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    control = read_arm(args.root / "control")
    comparisons = {}
    for arm in ("affinity", "grouped", "combined", "combined-eager"):
        if (args.root / arm / "benchmark.jsonl").exists():
            comparisons[arm] = compare(control, read_arm(args.root / arm))
    print(json.dumps(comparisons, indent=2))


if __name__ == "__main__":
    main()
