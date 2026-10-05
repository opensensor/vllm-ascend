# SPDX-License-Identifier: Apache-2.0
import json
import uuid
from pathlib import Path

from tools.glm_perf.resident_control import Control
from tools.glm_perf.resident_harness import ResidentClient
from tools.glm_perf.suite import make_groups, run_groups, summarize

root = Path(__file__).resolve().parent
client = ResidentClient("http://127.0.0.1:8001")
identities = None
summaries = {}
try:
    with (root / "serving-ab.jsonl").open("w") as out:
        for name in ("baseline", "rotation"):
            source = (root / "kpool_rotation_direct.py").read_text() if name != "baseline" else ""
            receipts = client.switch(Control(uuid.uuid4().hex, candidate=name, source=source))
            current = {(r["rank"], r["pid"], r["weight_storage_digest"]) for r in receipts}
            assert identities is None or current == identities
            identities = current
            (root / f"{name}-status.json").write_text(json.dumps(receipts, indent=2))
            print("ROUND", name, flush=True)

            def record(row, name=name):
                row["candidate"] = name
                out.write(json.dumps(row) + "\n")
                out.flush()
                print(
                    json.dumps(
                        {k: row.get(k) for k in ("candidate", "case_id", "passed", "decode_tokens_per_s", "error")}
                    ),
                    flush=True,
                )

            results = run_groups(
                make_groups(["short"], []), "http://127.0.0.1:8001", "glm53-flash-selective-w3", 42, on_result=record
            )
            summaries[name] = summarize(results)
            (root / "serving-summary.json").write_text(json.dumps(summaries, indent=2))
            print(json.dumps(summaries[name]), flush=True)
            if not summaries[name]["valid"]:
                raise RuntimeError("serving benchmark failed")
finally:
    restored = client.switch(Control(uuid.uuid4().hex))
    if client.request("/is_paused", method="GET")["is_paused"]:
        client.resume()
    (root / "comparison-restored.json").write_text(json.dumps(restored, indent=2))
