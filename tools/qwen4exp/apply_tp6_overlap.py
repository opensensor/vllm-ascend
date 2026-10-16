# SPDX-License-Identifier: Apache-2.0
"""Apply individually controlled early draft and complete-prefix selection."""

import argparse
import ast
import hashlib
import json
import uuid
from pathlib import Path

from tools.sidecard.qfn_resident_server import QfnResidentClient

SOURCE = """
def operation(worker):
    try:
        from tools.qwen4exp.tp6_overlap import OverlapControl
        owner = getattr(worker, '_qfn_tp6_overlap', None)
        if owner is None:
            owner = OverlapControl(worker, QSA_SOURCE)
            worker._qfn_tp6_overlap = owner
            owner.install()
        return owner.configure(**OPTIONS)
    except Exception as error:
        return dict(rank=worker.rank, passed=False, error=str(error))

class Candidate:
    def validate(self, engine):
        if not engine._pause_complete() or engine.batch_queue or engine.scheduler.get_num_unfinished_requests():
            raise RuntimeError('drain engine before overlap controls')
        if engine.vllm_config.parallel_config.tensor_parallel_size != 6:
            raise ValueError('requires six profiled workers')
        return dict(passed=True)
    def apply(self, engine):
        self.validate(engine)
        workers = engine.model_executor.collective_rpc(operation)
        passed = len(workers) == 6 and {w.get('rank') for w in workers} == set(range(6))
        return dict(passed=passed and all(w.get('passed') for w in workers), workers=workers)

def prepare(engine):
    return Candidate()
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--selection-source", type=Path, required=True)
    parser.add_argument("--early", action="store_true")
    parser.add_argument("--prefix", action="store_true")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("preserve previous receipts")
    code = args.selection_source.read_text()
    node = next(n for n in ast.parse(code).body if getattr(n, "name", None) == "qsa_indexer_select_groups_310")
    function = "".join(code.splitlines(True)[node.lineno - 1 : node.end_lineno])
    options = dict(early=args.early, prefix=args.prefix, verify=args.verify)
    source = "QSA_SOURCE = " + repr(function) + "\nOPTIONS = " + repr(options) + "\n" + SOURCE
    client = QfnResidentClient(
        "http://127.0.0.1:8001", args.run_root / "control.token", expected_workers=6, timeout=600
    )
    result = dict(source=source, options=options, before=client.rpc("resident_status"))
    client.request("/pause?mode=wait&clear_cache=false")
    revision, digest = "tp6_overlap_" + uuid.uuid4().hex, hashlib.sha256(source.encode()).hexdigest()
    try:
        result["prepared"] = client.request(
            "/qfn_engine_control", dict(operation="prepare", revision=revision, sha256=digest, source=source)
        )
        if not result["prepared"].get("passed"):
            raise RuntimeError("overlap preparation failed")
        result["applied"] = client.request(
            "/qfn_engine_control", dict(operation="apply", revision=revision, sha256=digest)
        )
        if not result["applied"].get("passed"):
            raise RuntimeError("overlap apply failed; retain paused owners")
        result["after"] = client.rpc("resident_status")
        identity = lambda rows: [(r["rank"], r["pid"], r["weight_storage_digest"], r["digest"]) for r in rows]
        if identity(result["before"]) != identity(result["after"]):
            raise RuntimeError("existing resident identity changed")
        result["resumed"] = client.resume()
    finally:
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["applied"]))


if __name__ == "__main__":
    main()
