import json
import subprocess
import time
from pathlib import Path

root = Path(__file__).resolve().parent
stop = root / "prefill-reduce-monitor.stop"
with (root / "prefill-reduce-1280-memory.jsonl").open("x") as out:
    while not stop.exists():
        result = subprocess.run(["ssh", "threadripper", "npu-smi info"], capture_output=True, text=True, timeout=15)
        out.write(json.dumps({"time": time.time(), "returncode": result.returncode, "output": result.stdout}) + "\n")
        out.flush()
        time.sleep(5)
