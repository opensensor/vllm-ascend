"""Run a drained private collective gate, resuming only after clean receipts."""

import json
import statistics
import subprocess
import time
from pathlib import Path

from tools.sidecard.qfn_resident_server import QfnResidentClient

ROOT = Path("/srv/ai/storage/artifacts/npu-collective-qfn-20261014-024609")
OUTPUT = ROOT / "qsa-resident-v1/prefill-collective-gate.json"


def main():
    client = QfnResidentClient("http://127.0.0.1:8001", ROOT / "resident-live-v3/control.token")
    if client.request("/is_paused", method="GET")["is_paused"]:
        raise RuntimeError("do not override an existing pause")
    result = dict(
        before=client.rpc("resident_status"),
        sensors_before=subprocess.check_output(["npu-smi", "info"], text=True, timeout=15),
    )
    assert client.request("/pause?mode=wait&clear_cache=false")["status"] == "paused"
    result["pause_mode"] = "wait"
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n")
    started = time.monotonic()
    try:
        result["ranks"] = client.rpc("qfn_cpu_profile_stop")
        result["seconds"] = time.monotonic() - started
        OUTPUT.write_text(json.dumps(result, indent=2) + "\n")
        if not all(row["prefill_collectives"].get("passed") for row in result["ranks"]):
            raise RuntimeError("private probe did not pass on all ranks; retain pause and inspect owners")
        if not all(
            record["private_owner_closed"]
            for row in result["ranks"]
            for record in row["prefill_collectives"]["records"]
        ):
            raise RuntimeError("not every private owner closed; retain pause")
        result["after"] = client.rpc("resident_status")
        identity = lambda rows: sorted((row["rank"], row["pid"], row["weight_storage_digest"]) for row in rows)
        assert identity(result["before"]) == identity(result["after"])
        assert all(not row["graphs_dirty"] and not row["native_failed"] for row in result["after"])
    except BaseException as error:
        result["error"] = f"{type(error).__name__}: {error}"
        result["inference_left_paused"] = True
        OUTPUT.write_text(json.dumps(result, indent=2) + "\n")
        raise
    result["resume"] = client.resume()
    result["sensors_after"] = subprocess.check_output(["npu-smi", "info"], text=True, timeout=15)
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n")
    # Each rank carries the same all-rank timings; summarize exactly one copy.
    for record in result["ranks"][0]["prefill_collectives"]["records"]:
        modes = sorted({sample["mode"] for sample in record["timings"]})
        medians = {
            mode: statistics.median(
                sample["completed_us_per_call"] for sample in record["timings"] if sample["mode"] == mode
            )
            for mode in modes
        }
        print(json.dumps(dict(geometry=record["geometry"], median_us=medians)), flush=True)
    print(json.dumps(dict(seconds=result["seconds"], resumed=result["resume"])), flush=True)


if __name__ == "__main__":
    main()
