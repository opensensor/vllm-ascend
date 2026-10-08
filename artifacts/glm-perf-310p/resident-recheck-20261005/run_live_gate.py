"""Bounded live gate that restores the candidate active when it starts."""

import hashlib
import importlib.util
import json
import sys
import time
import uuid
from pathlib import Path

from tools.glm_perf.resident_control import MODES, Control
from tools.glm_perf.resident_harness import ResidentClient
from tools.glm_perf.suite import Case, run_groups, run_request


def main():
    output = Path(sys.argv[1])
    output.mkdir(parents=True, exist_ok=True)
    source_path = Path("/home/matteius/experiments/glm-completed-pools-20261005/applied-candidate.py")
    source = source_path.read_text()
    reference = Path("tools/glm_perf/resident_candidates/mtp_norm_reference.py").read_text()
    digest = hashlib.sha256(source.encode()).hexdigest()
    rows = []
    switches = []
    client_class = ResidentClient
    if len(sys.argv) > 2:
        client_path = Path(sys.argv[2])
        spec = importlib.util.spec_from_file_location("live_resident_client", client_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        client_class = module.ResidentClient
        (output / "client-sha256.txt").write_text(hashlib.sha256(client_path.read_bytes()).hexdigest() + "\n")
    with (output / "receipts.jsonl").open("w") as log:

        def record(kind, **fields):
            log.write(json.dumps({"kind": kind, "timestamp": time.time(), **fields}) + "\n")
            log.flush()

        class LoggedClient(client_class):
            def request(self, path, payload=None, method="POST"):
                started = time.monotonic()
                try:
                    result = super().request(path, payload, method)
                except Exception as error:
                    record("control_error", path=path, error=repr(error))
                    raise
                record("control", path=path, payload=payload, result=result, elapsed_s=time.monotonic() - started)
                return result

        client = LoggedClient("http://127.0.0.1:8001", timeout=180)
        initial = client.rpc("resident_status")
        initial_paused = client.request("/is_paused", method="GET")["is_paused"]
        assert not initial_paused, "the server must be running before this gate"
        assert len(initial) == 4 and all(
            worker["candidate"] == "completed_pools"
            and worker["digest"] == digest
            and not worker["graphs_dirty"]
            and not worker["native_failed"]
            for worker in initial
        ), "the candidate snapshot must match every live worker"
        identity = {(r["rank"], r["pid"], r["weight_storage_digest"]) for r in initial}
        (output / "initial-status.json").write_text(json.dumps(initial, indent=2) + "\n")
        (output / "original-candidate.py").write_text(source)

        def switch(mode, name, text, recapture=False):
            started = time.monotonic()
            workers = client.switch(Control(uuid.uuid4().hex, mode, name, text, recapture))
            assert {(r["rank"], r["pid"], r["weight_storage_digest"]) for r in workers} == identity
            result = {"mode": mode, "candidate": name, "elapsed_s": time.monotonic() - started}
            switches.append(result)
            print(json.dumps({"event": "switch", **result}), flush=True)
            return workers

        def inference(label):
            case = Case(label, "arithmetic", "What is 17 + 28? Answer only the number.", "45", 64)
            row = run_request(case, client.base_url, "glm53-flash-selective-w3", 42, label, timeout_s=180)
            record("inference", result=row)
            rows.append(row)
            print(
                json.dumps(
                    {
                        "event": "inference",
                        "stage": label,
                        "valid": row["valid"],
                        "passed": row.get("passed"),
                        "content": row.get("content"),
                        "finish_reason": row.get("finish_reason"),
                        "elapsed_s": row.get("elapsed_s"),
                    }
                ),
                flush=True,
            )
            assert row["valid"], row.get("error")

        completed = False
        try:
            for mode in MODES:
                switch(mode, "completed_pools", source)
                inference(mode)
            switch("graph", "mtp_norm_reference", reference)
            inference("reference_recaptured")
            switch("graph", "completed_pools", source)
            inference("original_restored")
            cases = [
                Case(f"concurrent_{i}", "arithmetic", "What is 17 + 28? Answer only the number.", "45", 64)
                for i in range(4)
            ]
            for row in run_groups(
                [("restored_c4", cases)], client.base_url, "glm53-flash-selective-w3", 42, timeout_s=180
            ):
                record("inference", result=row)
                rows.append(row)
                assert row["valid"], row.get("error")
            print(
                json.dumps(
                    {
                        "event": "concurrent",
                        "valid": all(row["valid"] for row in rows[-4:]),
                        "passed": sum(bool(row.get("passed")) for row in rows[-4:]),
                    }
                ),
                flush=True,
            )
            completed = True
        finally:
            current = client.rpc("resident_status")
            if any(r["digest"] != digest or r["mode"] != initial[0]["mode"] or r["graphs_dirty"] for r in current):
                switch(initial[0]["mode"], "completed_pools", source)
            if client.request("/is_paused", method="GET")["is_paused"]:
                client.resume()
            final = client.rpc("resident_status")
            assert {(r["rank"], r["pid"], r["weight_storage_digest"]) for r in final} == identity
            assert all(
                r["digest"] == digest and r["mode"] == initial[0]["mode"] and not r["graphs_dirty"] for r in final
            )
            (output / "final-status.json").write_text(json.dumps(final, indent=2) + "\n")
            summary = {
                "control_gate_passed": completed,
                "inference_requests": len(rows),
                "valid_requests": sum(row["valid"] for row in rows),
                "scored_passes": sum(bool(row.get("passed")) for row in rows),
                "worker_and_weight_storage_unchanged": True,
                "restored_candidate": "completed_pools",
                "restored_digest": digest,
                "is_paused": client.request("/is_paused", method="GET")["is_paused"],
                "switches": switches,
            }
            (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
            print(json.dumps({"event": "complete", **summary}), flush=True)


if __name__ == "__main__":
    main()
