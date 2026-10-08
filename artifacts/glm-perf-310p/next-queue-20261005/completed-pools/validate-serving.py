# SPDX-License-Identifier: Apache-2.0
"""Apply one resident candidate, validate it, and leave it running on success."""

import json
import time
import urllib.request
import uuid
from pathlib import Path

import psutil

from tools.glm_perf.resident_control import Control
from tools.glm_perf.resident_harness import ResidentClient
from tools.glm_perf.suite import load_tokenizer_json, make_groups, retrieval_case, run_groups, summarize
from tools.glm_perf.worker_affinity import apply_bindings, plan_bindings


def main():
    root = Path(__file__).resolve().parent
    pid = int(Path("/home/matteius/experiments/glm-kpool-live-score-20261005/completed-pools.pid").read_text())
    process = psutil.Process(pid)
    for _ in range(600):
        if not process.is_running() or process.status() == psutil.STATUS_ZOMBIE:
            raise RuntimeError("GLM server exited during startup")
        try:
            with urllib.request.urlopen("http://127.0.0.1:8001/health", timeout=2) as response:
                if response.status == 200:
                    break
        except OSError:
            pass
        time.sleep(2)
    else:
        raise TimeoutError("GLM startup")
    workers = sorted((p for p in process.children(recursive=True) if "Worker_TP" in p.name()), key=lambda p: p.name())
    assert len(workers) == 4
    masks = [set(range(s, s + 4)) | set(range(s + 32, s + 36)) for s in (8, 12, 16, 20)]
    bindings = plan_bindings(pid, {p.pid: mask for p, mask in zip(workers, masks)})
    apply_bindings(bindings)
    (root / "serving-affinity.json").write_text(json.dumps(bindings, indent=2))
    client = ResidentClient("http://127.0.0.1:8001")
    source = (root / "kpool_completed_prefill.py").read_text().replace("def replacements(", "def base_replacements(")
    source += "\n" + (root / "serving-audit.py").read_text()
    (root / "applied-candidate.py").write_text(source)
    receipts = client.switch(Control(uuid.uuid4().hex, candidate="completed_pools", source=source))
    (root / "serving-initial-status.json").write_text(json.dumps(receipts, indent=2))
    print("CANDIDATE ACTIVE", json.dumps(receipts), flush=True)
    count_tokens, tokenizer = load_tokenizer_json(Path("/srv/ai/models/GLM-5.3-Flash-selective-W3-310p/tokenizer.json"))
    groups = [("cold8192", [retrieval_case(8192, variant=301, count_tokens=count_tokens)])]
    groups += make_groups(["short", "tool"], [])
    with (root / "serving-results.jsonl").open("w") as output:

        def record(row):
            output.write(json.dumps(row) + "\n")
            output.flush()
            print(
                json.dumps({k: row.get(k) for k in ("case_id", "passed", "ttft_s", "decode_tokens_per_s", "error")}),
                flush=True,
            )

        rows = run_groups(
            groups,
            client.base_url,
            "glm53-flash-selective-w3",
            42,
            on_result=record,
            tokenizer_identity=tokenizer,
            timeout_s=900,
        )
    result = summarize(rows)
    (root / "serving-summary.json").write_text(json.dumps(result, indent=2))
    receipts = client.rpc("resident_status")
    (root / "serving-final-status.json").write_text(json.dumps(receipts, indent=2))
    assert all(r["completed_pool_audit"]["compact_calls"] > 0 and not r["graphs_dirty"] for r in receipts), receipts
    print("SUMMARY", json.dumps(result), flush=True)
    if not result["passed"]:
        raise RuntimeError("candidate serving checks failed; inspect outputs before handing off")
    print("LEFT RUNNING: http://192.168.53.187:8001 glm53-flash-selective-w3", flush=True)


if __name__ == "__main__":
    main()
