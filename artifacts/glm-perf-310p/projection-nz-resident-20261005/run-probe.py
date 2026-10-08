# SPDX-License-Identifier: Apache-2.0
import json
import uuid
from pathlib import Path

from tools.glm_perf.resident_control import Control
from tools.glm_perf.resident_harness import ResidentClient

root = Path(__file__).resolve().parent
previous = Path("/home/matteius/experiments/glm-completed-pools-20261005/applied-candidate.py").read_text()
source = previous.replace("def replacements(", "def serving_replacements(") + "\n" + (root / "probe.py").read_text()
client = ResidentClient("http://127.0.0.1:8001")
try:
    result = client.switch(Control(uuid.uuid4().hex, candidate="projection_probe", source=source))
    (root / "probe-results.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)
finally:
    restored = client.switch(Control(uuid.uuid4().hex, candidate="completed_pools", source=previous))
    client.resume()
    (root / "restored.json").write_text(json.dumps(restored, indent=2))
