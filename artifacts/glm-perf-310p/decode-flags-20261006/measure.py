# SPDX-License-Identifier: Apache-2.0
"""Warm up, then retain three consecutive c1/c4 runs with metrics."""

import argparse
import dataclasses
import json
import time
import urllib.request
from pathlib import Path

from qualified_harness import ResidentClient

from tools.glm_perf.suite import make_groups, run_groups, summarize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("label")
    parser.add_argument("--quality", action="store_true")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup-tokens", type=int, default=128)
    args = parser.parse_args()
    if args.repeats < 1 or args.warmup_tokens < 1:
        parser.error("repeats and warmup tokens must be positive")
    root = Path(__file__).resolve().parent
    client = ResidentClient("http://127.0.0.1:8001", timeout=900)
    model = "glm53-flash-selective-w3"
    short = make_groups(["short"], [])
    status = client._acknowledged_rpc("resident_status", matches=client._status_matches)
    assert all(not row["graphs_dirty"] and not row.get("native_failed") for row in status)
    (root / f"{args.label}-start.json").write_text(json.dumps(status, indent=2) + "\n")

    def run(name, groups):
        output_path = root / f"{args.label}-{name}-results.jsonl"
        assert not output_path.exists()
        with output_path.open("w") as output:

            def record(row):
                output.write(json.dumps(row) + "\n")
                output.flush()
                print(
                    json.dumps(
                        {
                            "label": args.label,
                            "run": name,
                            "case": row["case_id"],
                            "valid": row["valid"],
                            "rate": row.get("decode_tokens_per_s"),
                        }
                    ),
                    flush=True,
                )

            started = time.time()
            rows = run_groups(groups, client.base_url, model, 42, on_result=record, timeout_s=900)
        result = summarize(rows)
        result["started_unix"] = started
        result["finished_unix"] = time.time()
        (root / f"{args.label}-{name}-summary.json").write_text(json.dumps(result, indent=2) + "\n")
        assert result["valid"], result
        return result

    run(
        "warmup",
        [(name, [dataclasses.replace(case, max_tokens=args.warmup_tokens) for case in cases]) for name, cases in short],
    )
    if args.quality:
        quality = [case for _, cases in make_groups(["quality"], []) for case in cases]
        groups = [(f"quality_{offset}", quality[offset : offset + 4]) for offset in range(0, len(quality), 4)]
        quality_result = run("quality", groups + make_groups(["tool"], []))
        print(json.dumps({"label": args.label, "strict_quality_passed": quality_result["passed"]}), flush=True)
    results = []
    for repeat in range(args.repeats):
        # No switching, cache-clearing, profiling, or operator builds during these runs.
        with urllib.request.urlopen(client.base_url + "/metrics") as response:
            (root / f"{args.label}-{repeat}-metrics-before.txt").write_bytes(response.read())
        result = run(f"repeat{repeat}", short)
        assert result["passed"], "throughput comparisons require full token caps"
        results.append(result)
        with urllib.request.urlopen(client.base_url + "/metrics") as response:
            (root / f"{args.label}-{repeat}-metrics-after.txt").write_bytes(response.read())
        print(
            json.dumps(
                {
                    "label": args.label,
                    "repeat": repeat,
                    "c1": result["groups"]["short_1"]["aggregate_decode_tokens_per_s"],
                    "c4": result["groups"]["short_4"]["aggregate_decode_tokens_per_s"],
                    "c4_per_request": result["groups"]["short_4"]["per_request_decode_p50"],
                }
            ),
            flush=True,
        )
    (root / f"{args.label}-repeats.json").write_text(json.dumps(results, indent=2) + "\n")
    if args.quality:
        # Uneven completion lengths force requests to drain and repack draft rows.
        cases = short[1][1]
        run(
            "drain",
            [("uneven_c4", [dataclasses.replace(case, max_tokens=cap) for case, cap in zip(cases, (17, 33, 65, 129))])],
        )
    (root / f"{args.label}-end.json").write_text(json.dumps(client.rpc("resident_status"), indent=2) + "\n")


if __name__ == "__main__":
    main()
