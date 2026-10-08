# SPDX-License-Identifier: Apache-2.0
"""Run the existing c1/c4 and tool-call serving gates after prompt benchmarks."""

import json
from pathlib import Path

from tools.glm_perf.resident_harness import ResidentClient
from tools.glm_perf.suite import make_groups, run_groups, summarize


def main():
    root = Path(__file__).resolve().parent
    client = ResidentClient("http://127.0.0.1:8001")
    with (root / "score-cache-serving-results.jsonl").open("w") as output:

        def record(row):
            output.write(json.dumps(row) + "\n")
            output.flush()
            print(
                json.dumps(
                    {key: row.get(key) for key in ("case_id", "valid", "ttft_s", "decode_tokens_per_s", "error")}
                ),
                flush=True,
            )

        rows = run_groups(
            make_groups(["short", "tool"], []),
            client.base_url,
            "glm53-flash-selective-w3",
            42,
            on_result=record,
            timeout_s=900,
        )
    result = summarize(rows)
    (root / "score-cache-serving-summary.json").write_text(json.dumps(result, indent=2) + "\n")
    status = client.rpc("resident_status")
    assert all(
        row["candidate"] == "completed_pools_sinkhorn" and not row["graphs_dirty"] and not row.get("native_failed")
        for row in status
    ), status
    (root / "final-score-cache-status.json").write_text(
        json.dumps(
            {
                "workers": status,
                "pause": client.request("/is_paused", method="GET"),
                "models": client.request("/v1/models", method="GET"),
            },
            indent=2,
        )
        + "\n"
    )
    assert len(rows) == 6 and all(row["valid"] for row in rows), result
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
