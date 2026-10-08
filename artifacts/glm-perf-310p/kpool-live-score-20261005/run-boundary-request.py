# SPDX-License-Identifier: Apache-2.0
"""One cold 20K retrieval through the instrumented graph server."""

import json
import re
import time
import urllib.request
from pathlib import Path

import psutil

from tools.glm_perf.suite import load_tokenizer_json, retrieval_case, run_groups
from tools.glm_perf.worker_affinity import apply_bindings, plan_bindings

root = Path(__file__).resolve().parent
results = root / "boundary-isolation"
pid = int((root / "boundary-trace.pid").read_text())
for attempt in range(120):
    try:
        with urllib.request.urlopen("http://127.0.0.1:8001/health", timeout=2) as response:
            if response.status == 200:
                break
    except OSError:
        time.sleep(5)
else:
    raise RuntimeError("Boundary server did not become ready")
bindings = {}
for worker in psutil.Process(pid).children(recursive=True):
    match = re.search(r"Worker_TP(\d+)", worker.name())
    if match:
        rank = int(match[1])
        bindings[worker.pid] = set(range(8 + rank * 4, 12 + rank * 4)) | set(range(40 + rank * 4, 44 + rank * 4))
assert len(bindings) == 4, bindings
plan = plan_bindings(pid, bindings)
apply_bindings(plan)
(results / "affinity.json").write_text(json.dumps(plan, indent=2))
count, identity = load_tokenizer_json(Path("/srv/ai/models/GLM-5.3-Flash-selective-W3-310p/tokenizer.json"))
case = retrieval_case(20480, variant=5, count_tokens=count)
print(json.dumps({"event": "request", "raw_tokens": case.raw_prompt_tokens}), flush=True)
with (results / "retrieval-20k.jsonl").open("w") as output:

    def record(row):
        output.write(json.dumps(row) + "\n")
        output.flush()
        print(json.dumps(row), flush=True)

    run_groups(
        [("boundary_20k", [case])],
        "http://127.0.0.1:8001",
        "glm53-flash-selective-w3",
        42,
        tokenizer_identity=identity,
        on_result=record,
        timeout_s=1200,
    )
