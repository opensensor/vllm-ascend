"""Run the shipped resident gate and retain control and inference receipts."""
import importlib.util
import json
import sys
import time
from pathlib import Path

from tools.glm_perf.resident_harness import ResidentClient

root = Path.cwd()
output = Path(sys.argv[1])
module_path = root / "tests/e2e/nightly/310p/single_node/resident/test_glm_resident_harness.py"
spec = importlib.util.spec_from_file_location("resident_gate", module_path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

with output.open("w") as log:
    def record(kind, **fields):
        row = dict(kind=kind, timestamp=time.time(), **fields)
        log.write(json.dumps(row) + "\n")
        log.flush()
        print(json.dumps(row), flush=True)

    class LoggedClient(ResidentClient):
        def request(self, path, payload=None, method="POST"):
            started = time.monotonic()
            try:
                result = super().request(path, payload, method)
            except Exception as error:
                record("control_error", path=path, payload=payload, error=repr(error))
                raise
            record("control", path=path, payload=payload, result=result, elapsed_s=time.monotonic()-started)
            return result

    original_request = module.run_request
    def run_request(*args, **kwargs):
        row = original_request(*args, **kwargs)
        record("inference", result=row)
        return row
    module.run_request = run_request
    client = LoggedClient("http://127.0.0.1:8001")
    assert client.request("/is_paused", method="GET")["is_paused"] is False
    try:
        module.test_glm_weights_stay_resident_through_modes_and_python_recapture(client)
    except Exception as error:
        record("gate", passed=False, error=repr(error))
        raise
    record("gate", passed=True)
