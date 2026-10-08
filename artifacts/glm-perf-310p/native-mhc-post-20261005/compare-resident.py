# SPDX-License-Identifier: Apache-2.0
"""Cold ABBA prefill comparison with worker/weight identity receipts."""

import json
import time
import urllib.request
import uuid
from pathlib import Path

import psutil

from tools.glm_perf.resident_control import Control
from tools.glm_perf.resident_harness import ResidentClient
from tools.glm_perf.suite import load_tokenizer_json, make_groups, retrieval_case, run_groups
from tools.glm_perf.worker_affinity import apply_bindings, plan_bindings

root = Path(__file__).resolve().parent
base = "http://127.0.0.1:8001"
for _ in range(600):
    try:
        with urllib.request.urlopen(base + "/health", timeout=2) as response:
            if response.status == 200:
                break
    except OSError:
        pass
    time.sleep(2)
else:
    raise TimeoutError("server startup")
server_pid = int((root / "resident640.pid").read_text())
workers = sorted(
    (p for p in psutil.Process(server_pid).children(recursive=True) if "Worker_TP" in p.name()), key=lambda p: p.name()
)
if len(workers) != 4:
    raise RuntimeError(f"Expected four TP workers, found {workers}")
masks = [list(range(start, start + 4)) + list(range(start + 32, start + 36)) for start in (8, 12, 16, 20)]
plan = plan_bindings(server_pid, {process.pid: set(cpus) for process, cpus in zip(workers, masks, strict=True)})
apply_bindings(plan)
(root / "affinity-resident640.json").write_text(json.dumps(plan, indent=2))
count, tokenizer_identity = load_tokenizer_json(Path("/srv/ai/models/GLM-5.3-Flash-selective-W3-310p/tokenizer.json"))
case = retrieval_case(8192, count_tokens=count)
client = ResidentClient(base, 4, 1200)
identity = None
with (root / "resident640-results.jsonl").open("w") as out:
    for index, name in enumerate(("baseline", "native", "native", "baseline")):
        source = (root / "resident_candidate.py").read_text() if name == "native" else ""
        control = Control.from_dict(
            dict(generation=uuid.uuid4().hex, mode="graph", candidate=name, source=source, recapture=False)
        )
        receipts = client.switch(control)
        current = {(item["rank"], item["pid"], item["weight_storage_digest"]) for item in receipts}
        if identity is not None and identity != current:
            raise RuntimeError("resident model identity changed")
        identity = current
        print(f"ROUND {index} {name}: cold cache; weights unchanged", flush=True)
        groups = [("cold_8k", [case])]
        if index < 2:
            # Exercise the decode path at both batch sizes. These short
            # prompts do not qualify the native prefill math by themselves.
            groups += make_groups(["fault", "fault4"], [])

        def record(row, name=name, index=index, receipts=receipts):
            row["candidate"] = name
            row["round"] = index
            row["workers"] = receipts
            out.write(json.dumps(row) + "\n")
            out.flush()
            print(
                json.dumps(
                    {
                        key: row.get(key)
                        for key in ("case_id", "candidate", "ttft_s", "decode_tokens_per_s", "passed", "error")
                    }
                ),
                flush=True,
            )

        rows = run_groups(
            groups,
            base,
            "glm53-flash-selective-w3",
            42,
            on_result=record,
            tokenizer_identity=tokenizer_identity,
            timeout_s=1200,
        )
        if any(row.get("error") for row in rows):
            raise RuntimeError("request failed; stop comparison")
print("ABBA comparison complete", flush=True)
