"""Throttle only the restoration supervisor; do not pause serving workers."""
import os
import re
import signal
import subprocess
import time
from pathlib import Path

SUPERVISOR = 26070
EXPECTED = b"restore-retained-v10.py"
STOP_AT_C = 84
RESUME_AT_C = 79
paused = True  # Supervisor was explicitly SIGSTOPed before starting this guard.
while True:
    try:
        command = Path(f"/proc/{SUPERVISOR}/cmdline").read_bytes()
    except FileNotFoundError:
        break
    if EXPECTED not in command:
        break
    output = subprocess.check_output(["npu-smi", "info"], text=True, timeout=15)
    temperatures = [int(x) for x in re.findall(r"310P3\s*\|\s*OK\s*\|\s*\S+\s+(\d+)", output)]
    if len(temperatures) != 4:
        raise RuntimeError("temperature query did not return four healthy chips; supervisor left in current state")
    peak = max(temperatures)
    if not paused and peak >= STOP_AT_C:
        os.kill(SUPERVISOR, signal.SIGSTOP)
        paused = True
    elif paused and peak <= RESUME_AT_C:
        os.kill(SUPERVISOR, signal.SIGCONT)
        paused = False
    print(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), temperatures, "tests_waiting" if paused else "tests_running", flush=True)
    time.sleep(10)
print("Restoration supervisor exited", flush=True)
