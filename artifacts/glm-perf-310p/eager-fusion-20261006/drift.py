# SPDX-License-Identifier: Apache-2.0
"""Measure short-workload decode stability across back-to-back runs (no switch)."""

import dataclasses
import json
from pathlib import Path

from tools.glm_perf.resident_harness import ResidentClient
from tools.glm_perf.suite import make_groups, run_groups, summarize

MODEL = "glm53-flash-selective-w3"


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=128)
    args = parser.parse_args()

    client = ResidentClient("http://127.0.0.1:8001", timeout=900)
    status = client.rpc("resident_status")
    print(json.dumps({"state": [{"rank": w["rank"], "candidate": w["candidate"], "graphs_dirty": w["graphs_dirty"]} for w in status]}), flush=True)

    short = [
        (name, [dataclasses.replace(case, max_tokens=args.max_tokens) for case in cases])
        for name, cases in make_groups(["short"], [])
    ]
    out = []
    for repeat in range(args.repeats):
        rows = run_groups(short, client.base_url, MODEL, 42, timeout_s=900)
        summary = summarize(rows)
        out.append(summary)
        print(
            json.dumps(
                {
                    "repeat": repeat,
                    "c1": summary["groups"]["short_1"]["aggregate_decode_tokens_per_s"],
                    "c4": summary["groups"]["short_4"]["aggregate_decode_tokens_per_s"],
                    "c4_p50": summary["groups"]["short_4"]["per_request_decode_p50"],
                }
            ),
            flush=True,
        )
    Path(__file__).resolve().with_name("drift-summary.json").write_text(json.dumps(out, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
