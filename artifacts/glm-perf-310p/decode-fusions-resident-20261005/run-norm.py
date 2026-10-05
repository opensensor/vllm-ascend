# SPDX-License-Identifier: Apache-2.0
import json
import uuid
from pathlib import Path

from tools.glm_perf.resident_control import Control
from tools.glm_perf.resident_harness import ResidentClient

root = Path(__file__).resolve().parent
client = ResidentClient("http://127.0.0.1:8001")
try:
    client.switch(Control(uuid.uuid4().hex, candidate="mhc_norm_probe", source=(root / "norm-probe.py").read_text()))
    # The switch status is collected by resident_reset, not the replaced RPC.
    client.request("/pause?mode=wait&clear_cache=true")
    results = client.rpc("resident_status")
    (root / "norm-results.json").write_text(json.dumps(results, indent=2))
    print(json.dumps([{k: r[k] for k in ("rank", "mhc_norm_audit")} for r in results]), flush=True)
finally:
    restored = client.switch(Control(uuid.uuid4().hex))
    if client.request("/is_paused", method="GET")["is_paused"]:
        client.resume()
    (root / "norm-restored.json").write_text(json.dumps(restored, indent=2))
