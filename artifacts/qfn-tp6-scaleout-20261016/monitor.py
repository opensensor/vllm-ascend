"""Authenticated thermal holds for the explicitly selected six-NPU service."""

import os
import sys
import time
from pathlib import Path

from tools.qwen4exp.thermal_controller import EngineControl, ThermalController, run
from tools.sidecard.qfn_resident_server import read_control_token

root = Path(sys.argv[1])
pid = int(sys.argv[2])
while not (root / "control.token").exists():
    os.kill(pid, 0)
    time.sleep(1)
engine = EngineControl("http://127.0.0.1:8001")
engine.opener.addheaders = [("x-qfn-resident-token", read_control_token(root / "control.token"))]
controller = ThermalController(engine, high_c=94, low_c=85, expected_devices=6)
run(controller, pid, poll_s=1, sample_timeout_s=8)
