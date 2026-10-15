"""Bounded decode timing and garbage-collection diagnosis; restore every hook."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from tools.qwen4exp.apply_ple_serving import apply
from tools.sidecard.qfn_resident_server import QfnResidentClient

SOURCE = """
def gc_record(recorder):
    import gc, time
    started = {}
    def callback(phase, info):
        generation = info["generation"]
        if phase == "start": started[generation] = time.monotonic_ns()
        else:
            begin = started.pop(generation, None)
            if begin is not None:
                recorder.total += 1
                recorder.records.append(dict(label="gc",start_ns=begin,
                    duration_ns=time.monotonic_ns()-begin,generation=generation,
                    collected=info["collected"],uncollectable=info["uncollectable"]))
    gc.callbacks.append(callback)
    recorder.gc_callback = callback

def worker_action(wrapper):
    import gc, json
    from pathlib import Path
    from tools.qwen4exp.mixed_prefill_profile import install_worker,stop_worker
    worker = getattr(wrapper,"worker",wrapper)
    if START:
        result = install_worker(worker)
        gc_record(worker._qfn_mixed_profile)
        return result
    gc.callbacks.remove(worker._qfn_mixed_profile.gc_callback)
    result = stop_worker(worker)
    Path(OUTPUT,f"rank{worker.rank}-spans.json").write_text(json.dumps(result))
    return dict(rank=worker.rank,passed=True,total=result["total"],dropped=result["dropped"])

class Candidate:
    def validate(self,engine):
        assert engine._pause_complete() and not engine.batch_queue
        assert not engine.scheduler.get_num_unfinished_requests()
        return dict(passed=True)
    def apply(self,engine):
        import gc, json
        from pathlib import Path
        from tools.qwen4exp.mixed_prefill_profile import Recorder
        self.validate(engine)
        if START:
            assert getattr(engine,"_qfn_tail_profile",None) is None
            recorder = engine._qfn_tail_profile = Recorder()
            recorder.wrap(engine.scheduler,"schedule","schedule")
            # The busy loop stores a bound step function; wrap that instance seam.
            recorder.wrap(engine,"step_fn","engine_step")
            gc_record(recorder)
        rows = engine.model_executor.collective_rpc(worker_action)
        if not START:
            recorder = engine._qfn_tail_profile
            gc.callbacks.remove(recorder.gc_callback)
            recorder.restore()
            Path(OUTPUT,"engine-spans.json").write_text(json.dumps(recorder.snapshot()))
            engine._qfn_tail_profile = None
        return dict(passed=all(row["passed"] for row in rows),workers=rows)
def prepare(engine): return Candidate()
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    client = QfnResidentClient("http://127.0.0.1:8001", args.run_root / "control.token", timeout=900)

    def control(start):
        return apply(
            client,
            f"START={start!r}\nOUTPUT={str(args.output)!r}\n" + SOURCE,
            args.output / ("start" if start else "stop"),
        )

    control(True)
    try:
        subprocess.run(
            [
                sys.executable,
                str(args.run_root.parent / "qsa-decode-benchmark.py"),
                "--run-root",
                str(args.run_root),
                "--label",
                "high-end-tail-v2",
                "--trials",
                "5",
            ],
            check=True,
        )
    finally:
        control(False)
    print(json.dumps(dict(restored=True)), flush=True)


if __name__ == "__main__":
    main()
