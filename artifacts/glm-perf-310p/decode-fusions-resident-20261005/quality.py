# SPDX-License-Identifier: Apache-2.0
import dataclasses
import json
import uuid
from pathlib import Path

from tools.glm_perf.resident_control import Control
from tools.glm_perf.resident_harness import ResidentClient
from tools.glm_perf.suite import make_groups, run_groups, summarize

root = Path(__file__).resolve().parent
client = ResidentClient("http://127.0.0.1:8001")
try:
    client.switch(Control(uuid.uuid4().hex, candidate="rotation_shadow", source=(root / "shadow.py").read_text()))
    with (root / "quality.jsonl").open("w") as out:

        def record(row):
            out.write(json.dumps(row) + "\n")
            out.flush()
            print(json.dumps({k: row.get(k) for k in ("case_id", "passed", "content", "error")}), flush=True)

        rows = run_groups(
            make_groups(["quality"], []), "http://127.0.0.1:8001", "glm53-flash-selective-w3", 42, on_result=record
        )
        (root / "quality-summary.json").write_text(json.dumps(summarize(rows), indent=2))
    # Cover the eight-row graph as well; this is a parity check, not timing.
    cases = make_groups(["short"], [])[1][1]
    run_groups(
        [("shadow_c4", [dataclasses.replace(case, max_tokens=16) for case in cases])],
        "http://127.0.0.1:8001",
        "glm53-flash-selective-w3",
        42,
    )
    client.request("/pause?mode=wait&clear_cache=true")
    receipts = client.rpc("resident_status")
    (root / "live-rotation-parity.json").write_text(json.dumps(receipts, indent=2))
    print(json.dumps([{k: r[k] for k in ("rank", "rotation_parity")} for r in receipts]), flush=True)
    assert all(all(pair[0] == 0 and pair[1] > 0 for pair in r["rotation_parity"].values()) for r in receipts)
finally:
    restored = client.switch(Control(uuid.uuid4().hex))
    if client.request("/is_paused", method="GET")["is_paused"]:
        client.resume()
    (root / "quality-restored.json").write_text(json.dumps(restored, indent=2))
