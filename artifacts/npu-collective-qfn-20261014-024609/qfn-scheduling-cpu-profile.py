"""One bounded C1 CPU diagnostic using existing worker profiler methods."""

import argparse
import hashlib
import json
import uuid
from pathlib import Path

from tools.sidecard.qfn_resident_server import QfnResidentClient

ROOT = Path("/srv/ai/storage/artifacts/npu-collective-qfn-20261014-024609")
SOURCE = """
def operation(worker):
    from tools.sidecard.qfn_collective_profile import QfnProfileWorkerExtension
    if START:
        if worker.profiler is not None or getattr(worker, '_qfn_native_profile_session', None) is not None:
            raise RuntimeError('another profiler is active')
        return QfnProfileWorkerExtension.qfn_cpu_profile_start(worker)
    return QfnProfileWorkerExtension.qfn_cpu_profile_stop(worker, OUTPUT)
class Candidate:
    def validate(self, engine):
        if not engine._pause_complete() or engine.batch_queue or engine.scheduler.get_num_unfinished_requests():
            raise RuntimeError('drain before changing profiler ownership')
        return dict(passed=True)
    def apply(self, engine):
        self.validate(engine)
        rows = engine.model_executor.collective_rpc(operation)
        return dict(passed=len(rows)==4 and all(row.get(RESULT_KEY) for row in rows), workers=rows)
def prepare(engine):
    return Candidate()
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", default="v1")
    args = parser.parse_args()
    if Path(args.label).name != args.label:
        raise ValueError("plain label required")
    OUTPUT = ROOT / "resident-live-v6" / f"cpu-scheduling-{args.label}"
    OUTPUT.mkdir(exist_ok=False)
    client = QfnResidentClient("http://127.0.0.1:8001", ROOT / "resident-live-v6/control.token")
    result = {}

    def save():
        (OUTPUT / "receipt.json").write_text(json.dumps(result, indent=2) + "\n")

    def control(start):
        source = (
            SOURCE.replace("START", repr(start))
            .replace("OUTPUT", repr(str(OUTPUT)))
            .replace("RESULT_KEY", repr("started" if start else "stopped"))
        )
        revision, digest = "profile_" + uuid.uuid4().hex, hashlib.sha256(source.encode()).hexdigest()
        client.request("/pause?mode=wait&clear_cache=false")
        prepared = client.request(
            "/qfn_engine_control", dict(operation="prepare", revision=revision, sha256=digest, source=source)
        )
        assert prepared.get("passed"), prepared
        applied = client.request("/qfn_engine_control", dict(operation="apply", revision=revision, sha256=digest))
        result["start" if start else "stop"] = applied
        save()
        assert applied.get("passed"), applied
        client.resume()

    control(True)
    try:
        result["response"] = client.request(
            "/v1/chat/completions",
            dict(
                model="qwen38-flash-next",
                messages=[
                    dict(
                        role="user",
                        content="Explain how a hash table works and discuss collision resolution strategies.",
                    )
                ],
                chat_template_kwargs=dict(enable_thinking=False),
                temperature=0,
                max_tokens=128,
                min_tokens=128,
                ignore_eos=True,
            ),
        )
        assert result["response"]["usage"]["completion_tokens"] == 128
        save()
    finally:
        control(False)
    result["unpaused"] = not client.request("/is_paused", method="GET")["is_paused"]
    save()
    print(json.dumps(dict(completed=True, unpaused=result["unpaused"])))


if __name__ == "__main__":
    main()
