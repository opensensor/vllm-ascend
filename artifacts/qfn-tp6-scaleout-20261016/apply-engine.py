"""Apply a recorded, drained engine update and retain before/after ownership receipts."""

import sys
from pathlib import Path

from tools.qwen4exp.apply_ple_serving import apply
from tools.sidecard.qfn_resident_server import QfnResidentClient

ROOT = Path("/srv/ai/storage/artifacts/qfn-tp6-scaleout-20261016")
client = QfnResidentClient(
    "http://127.0.0.1:8001", ROOT / "resident-v3/control.token", expected_workers=6, timeout=1800
)
result = apply(client, Path(sys.argv[1]).read_text(), ROOT / sys.argv[2])
print(result["applied"])
