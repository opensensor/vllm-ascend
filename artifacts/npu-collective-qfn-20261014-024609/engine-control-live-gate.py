"""Qualify engine-side prepare/apply routing without changing scheduling."""

import hashlib
import json
import uuid
from pathlib import Path

from tools.sidecard.qfn_resident_server import QfnResidentClient

ROOT = Path("/srv/ai/storage/artifacts/npu-collective-qfn-20261014-024609")
SOURCE = """
import os
class Candidate:
    def validate(self, engine):
        return dict(passed=engine.is_scheduler_paused()
                    and not engine.batch_queue
                    and engine.scheduler.get_num_unfinished_requests() == 0)
    def apply(self, engine):
        return dict(passed=True, engine_pid=os.getpid(),
                    scheduler_type=type(engine.scheduler).__name__,
                    async_scheduling=engine.async_scheduling,
                    batch_queue_size=engine.batch_queue_size,
                    scheduler_changed=False)
def prepare(engine):
    return Candidate()
"""


def main():
    client = QfnResidentClient("http://127.0.0.1:8001", ROOT / "resident-live-v6/control.token")
    output = ROOT / "resident-live-v6/engine-control-live-gate.json"
    if output.exists():
        raise FileExistsError(output)
    result = dict(before=client.request("/qfn_engine_control", dict(operation="status")))
    assert not result["before"]["paused"] and not result["before"]["failed"]
    request = dict(
        operation="prepare",
        revision="gate_" + uuid.uuid4().hex,
        source=SOURCE,
        sha256=hashlib.sha256(SOURCE.encode()).hexdigest(),
    )
    result["unpaused_rejected"] = client.request("/qfn_engine_control", request)
    assert not result["unpaused_rejected"]["passed"]
    client.request("/pause?mode=wait&clear_cache=false")
    try:
        result["wrong_digest_rejected"] = client.request("/qfn_engine_control", request | {"sha256": "0" * 64})
        assert not result["wrong_digest_rejected"]["passed"]
        result["prepared"] = client.request("/qfn_engine_control", request)
        assert result["prepared"]["passed"]
        result["applied"] = client.request(
            "/qfn_engine_control",
            {key: value for key, value in request.items() if key != "source"} | {"operation": "apply"},
        )
        assert result["applied"]["passed"]
        assert result["applied"]["applied"]["async_scheduling"] is False
        result["source"] = SOURCE
        result["resume"] = client.resume()
    finally:
        output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["applied"], indent=2))


if __name__ == "__main__":
    main()
