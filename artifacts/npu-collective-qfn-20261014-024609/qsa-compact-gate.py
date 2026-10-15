"""Run the isolated selector gate while resident requests are drained."""

import json
from pathlib import Path

from tools.sidecard.qfn_resident_server import QfnResidentClient

ROOT = Path("/srv/ai/storage/artifacts/npu-collective-qfn-20261014-024609")
client = QfnResidentClient("http://127.0.0.1:8001", ROOT / "resident-live-v3/control.token")
client.request("/pause?mode=wait&clear_cache=false")
try:
    result = client.rpc("qfn_cpu_profile_stop")
    (ROOT / "qsa-resident-v1/compact-component-gate.json").write_text(json.dumps(result, indent=2) + "\n")
    assert all(row["qsa_compact_topk"]["passed"] for row in result), result
    for row in result:
        print(
            json.dumps(
                {
                    "rank": row["rank"],
                    "cases": len(row["qsa_compact_topk"]["records"]),
                    "timings": [item for item in row["qsa_compact_topk"]["records"] if "timing" in item],
                }
            ),
            flush=True,
        )
finally:
    client.request("/resume")
