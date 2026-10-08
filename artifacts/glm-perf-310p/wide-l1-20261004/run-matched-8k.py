"""Repeat the saved cold 7,269-token retrieval against an experimental server."""

import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

from tools.glm_perf.suite import Case, run_request


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-url", default="http://192.168.53.187:8001")
    parser.add_argument("--group-id", required=True)
    args = parser.parse_args()

    saved = json.loads(args.source.read_text().splitlines()[0])
    case = Case(
        case_id=saved["case_id"],
        category=saved["category"],
        prompt=saved["prompt"],
        expected=saved["expected"],
        max_tokens=saved["request_settings"]["max_tokens"],
    )
    for _ in range(300):
        try:
            with urllib.request.urlopen(args.base_url + "/health", timeout=2):
                break
        except (OSError, urllib.error.URLError):
            time.sleep(2)
    else:
        raise TimeoutError("server did not become healthy within 600 seconds")

    row = run_request(case, args.base_url, saved["model"], 42, args.group_id, timeout_s=900)
    args.output.write_text(json.dumps(row) + "\n")
    print(
        json.dumps(
            {key: row.get(key) for key in ("valid", "passed", "ttft_s", "elapsed_s", "content", "error", "usage")}
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
