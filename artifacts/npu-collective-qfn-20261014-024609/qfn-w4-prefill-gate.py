"""Run one bounded private W4 gate in the existing drained workers."""

import argparse
import json
import subprocess
import time
from pathlib import Path

from tools.sidecard.qfn_resident_server import QfnResidentClient

ROOT = Path("/srv/ai/storage/artifacts/npu-collective-qfn-20261014-024609")
BUILD = "/srv/ai/storage/artifacts/qfn-route-finalize-compile-20261014-v1/build2"


def identity(rows):
    return sorted((row["rank"], row["pid"], row["weight_storage_digest"]) for row in rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("load", "schedule", "validate", "timing"))
    parser.add_argument("--version", type=int, choices=(1, 2, 3), default=1)
    parser.add_argument("--build", default=BUILD)
    parser.add_argument("--rows", type=int, default=650)
    parser.add_argument("--projection", choices=("gate_up_proj", "down_proj"), default="down_proj")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--label", required=True)
    args = parser.parse_args()
    output = ROOT / "qsa-resident-v1" / f"w4-{args.label}.json"
    if output.exists():
        raise RuntimeError("preserve existing receipts; choose a new label")
    client = QfnResidentClient("http://127.0.0.1:8001", ROOT / "resident-live-v3/control.token")
    if client.request("/is_paused", method="GET")["is_paused"]:
        raise RuntimeError("do not override an existing pause")
    result = dict(
        before=client.rpc("resident_status"),
        sensors_before=subprocess.check_output(["npu-smi", "info"], text=True, timeout=15),
    )
    request = dict(
        action="preflight",
        build=args.build,
        version=args.version,
        rows=[args.rows],
        repeats=args.repeats,
        projection=args.projection,
    )
    result["request"] = request.copy()
    assert client.request("/pause?mode=wait&clear_cache=false")["status"] == "paused"
    output.write_text(json.dumps(result, indent=2) + "\n")
    started = time.monotonic()
    try:
        result["preflight"] = client.rpc("qfn_cpu_profile_stop", json.dumps(request))
        if not all(row.get("passed") is True for row in result["preflight"]):
            # Preflight verifies files/free space only; no native submissions.
            result["resume"] = client.resume()
            result["skipped"] = "preflight failed; no native load or submission"
        else:
            request["action"] = args.action
            result["ranks"] = client.rpc("qfn_cpu_profile_stop", json.dumps(request))
            output.write_text(json.dumps(result, indent=2) + "\n")
            if not all(row.get("passed") is True for row in result["ranks"]):
                raise RuntimeError("private gate did not pass on all ranks; leave paused for inspection")
            result["after"] = client.rpc("resident_status")
            assert identity(result["before"]) == identity(result["after"])
            assert all(not row["graphs_dirty"] and not row["native_failed"] for row in result["after"])
            result["resume"] = client.resume()
    except BaseException as error:
        result["error"] = f"{type(error).__name__}: {error}"
        result["inference_left_paused"] = True
        output.write_text(json.dumps(result, indent=2) + "\n")
        raise
    result["seconds"] = time.monotonic() - started
    result["sensors_after"] = subprocess.check_output(["npu-smi", "info"], text=True, timeout=15)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: result.get(key) for key in ("seconds", "resume", "skipped")}), flush=True)
    print(
        json.dumps(
            [
                dict(rank=row.get("rank"), passed=row.get("passed"), error=row.get("error"))
                for row in result.get("ranks", [])
            ]
        )
    )


if __name__ == "__main__":
    main()
