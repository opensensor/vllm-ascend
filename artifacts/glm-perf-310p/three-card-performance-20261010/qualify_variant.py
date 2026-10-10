"""Run real requests and verify successful decode replays on every PP/TP worker."""

import argparse
import json
import subprocess
import urllib.request
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:18701")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    def rpc(method):
        request = urllib.request.Request(
            args.url + "/collective_rpc",
            data=json.dumps({"method": method, "timeout": 300}).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=310) as response:
            return json.loads(response.read())

    before = rpc("card_start_replay_count")
    (args.output / "replay-before.json").write_text(json.dumps(before, indent=2) + "\n")
    try:
        result = subprocess.run(
            [
                "pytest",
                "--confcutdir=tests/e2e/nightly/310p/single_node/card_profiles",
                "-q",
                "tests/e2e/nightly/310p/single_node/card_profiles",
                "--glm-profile-url=" + args.url,
                "--glm-profile-cards=3",
                "--junitxml=" + str(args.output / "e2e.xml"),
            ],
            check=False,
            text=True,
            capture_output=True,
        )
        (args.output / "e2e.txt").write_text(result.stdout + result.stderr)
        after = rpc("card_replay_status")
        (args.output / "replay-after.json").write_text(json.dumps(after, indent=2) + "\n")
    finally:
        rpc("card_stop_replay_count")
    print(result.stdout + result.stderr, flush=True)
    if result.returncode:
        raise SystemExit(result.returncode)
    rows = after["results"]
    assert len(rows) == 6 and all(row["replays"] > 0 for row in rows), after
    print("ALL_SIX_WORKERS_REPLAYED_REAL_REQUESTS", flush=True)


if __name__ == "__main__":
    main()
