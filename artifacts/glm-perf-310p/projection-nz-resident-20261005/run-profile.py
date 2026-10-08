# SPDX-License-Identifier: Apache-2.0
import json
import uuid
from pathlib import Path

from tools.glm_perf.resident_control import Control
from tools.glm_perf.resident_harness import ResidentClient
from tools.glm_perf.suite import make_groups, run_groups

root = Path(__file__).resolve().parent
previous = Path("/home/matteius/experiments/glm-completed-pools-20261005/applied-candidate.py").read_text()
source = (
    previous.replace("def replacements(", "def serving_replacements(") + "\n" + (root / "profile-live.py").read_text()
)
client = ResidentClient("http://127.0.0.1:8001")
active = False
try:
    client.switch(Control(uuid.uuid4().hex, candidate="decode_trace", source=source))
    client.request("/pause?mode=wait&clear_cache=true")
    client.rpc("profile", True)
    active = True
    client.resume()
    rows = run_groups(make_groups(["fault"], []), client.base_url, "glm53-flash-selective-w3", 42)
    (root / "profile-request.json").write_text(json.dumps(rows, indent=2))
    client.request("/pause?mode=wait&clear_cache=false")
    receipts = client.rpc("profile", False)
    active = False
    (root / "profile-stop.json").write_text(json.dumps(receipts, indent=2))
    print(json.dumps(receipts), flush=True)
finally:
    if active:
        client.request("/pause?mode=wait&clear_cache=false")
        client.rpc("profile", False)
    result = client.switch(Control(uuid.uuid4().hex, candidate="completed_pools", source=previous))
    client.resume()
    (root / "profile-restored.json").write_text(json.dumps(result, indent=2))
