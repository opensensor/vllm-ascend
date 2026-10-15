"""Verify the complete retained stack after mixed-prefill optimization."""

import json
from pathlib import Path

from tools.qwen4exp.apply_ple_serving import apply
from tools.sidecard.qfn_resident_server import QfnResidentClient

ROOT = Path("/srv/ai/storage/artifacts/qfn-mixed-opt-20261015")
RUN = Path("/srv/ai/storage/artifacts/npu-collective-qfn-20261014-024609/resident-live-v14")
client = QfnResidentClient("http://127.0.0.1:8001", RUN / "control.token", timeout=900)
result = apply(client, (ROOT / "final-audit-source.py").read_text(), ROOT / "v14-final-audit-v2")
print(json.dumps(result["applied"], indent=2))
assert not client.request("/is_paused", method="GET")["is_paused"]
assert client.request("/health", method="GET") is not False
