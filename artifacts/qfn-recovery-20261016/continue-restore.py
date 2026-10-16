"""Start later restoration only after the retained restoration gate succeeds."""

import subprocess
import sys
import time
from pathlib import Path

BASE = Path("/srv/ai/storage/artifacts/npu-collective-qfn-20261014-024609")
ROOT = Path("/srv/ai/storage/artifacts/qfn-recovery-20261016")
for attempt in range(240):
    text = (BASE / "restore-retained-v16.log").read_text()
    if "Traceback (most recent call last)" in text:
        raise RuntimeError("retained restoration failed; inspect its log")
    if "ALL RETAINED OPTIMIZATIONS RESTORED AND VERIFIED" in text:
        break
    time.sleep(5)
else:
    raise TimeoutError("retained restoration did not finish")
subprocess.run([sys.executable, str(ROOT / "restore-later.py")], check=True)
