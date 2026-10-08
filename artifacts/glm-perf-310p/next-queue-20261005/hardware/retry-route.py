# SPDX-License-Identifier: Apache-2.0
"""Record the graph failure's full exception chain, then restore baseline."""

import traceback
import uuid
from pathlib import Path

from tools.glm_perf.resident_control import Control
from tools.glm_perf.resident_harness import ResidentClient

client = ResidentClient("http://127.0.0.1:8001")
source = Path(
    "/srv/ai/src/glm-selective-w3-nz-test-20261004/tools/glm_perf/resident_candidates/direct_route_tokens.py"
).read_text()
try:
    client.switch(Control(uuid.uuid4().hex, candidate="direct_route_diagnostic", source=source))
    print("CAPTURE PASSED", flush=True)
except Exception:
    traceback.print_exc()
finally:
    client.switch(Control(uuid.uuid4().hex))
    if client.request("/is_paused", method="GET")["is_paused"]:
        client.resume()
