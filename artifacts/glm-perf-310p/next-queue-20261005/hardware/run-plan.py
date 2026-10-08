# SPDX-License-Identifier: Apache-2.0
"""Run an explicit serving comparison plan on the already loaded GLM workers."""

import argparse
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument("--pid-file", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    root = args.plan.parent
    client = ResidentClient("http://127.0.0.1:8001")
    pid = int(args.pid_file.read_text())
    process = psutil.Process(pid)
    for _ in range(600):
        if not process.is_running() or process.status() == psutil.STATUS_ZOMBIE:
            raise RuntimeError("selected server exited during startup")
        try:
            with urllib.request.urlopen("http://127.0.0.1:8001/health", timeout=2) as response:
                if response.status == 200:
                    break
        except OSError:
            pass
        time.sleep(2)
    else:
        raise TimeoutError("server startup")
    workers = sorted(
        (p for p in psutil.Process(pid).children(recursive=True) if "Worker_TP" in p.name()), key=lambda p: p.name()
    )
    assert len(workers) == 4
    masks = [set(range(s, s + 4)) | set(range(s + 32, s + 36)) for s in (8, 12, 16, 20)]
    binding = plan_bindings(pid, {p.pid: mask for p, mask in zip(workers, masks)})
    apply_bindings(binding)
    (root / (args.plan.stem + "-affinity.json")).write_text(json.dumps(binding, indent=2))
    count_tokens, tokenizer_identity = load_tokenizer_json(
        Path("/srv/ai/models/GLM-5.3-Flash-selective-W3-310p/tokenizer.json")
    )
    identities = None
    summaries = {}
    try:
        with (root / (args.plan.stem + "-serving.jsonl")).open("w") as output:
            for trial in plan:
                name = trial["name"]
                # Restore first: candidate factories may capture the current method.
                client.switch(Control(uuid.uuid4().hex))
                source = Path(trial["source"]).read_text() if trial.get("source") else ""
                receipts = (
                    client.switch(Control(uuid.uuid4().hex, candidate=name if source else "baseline", source=source))
                    if source
                    else client.rpc("resident_status")
                )
                current = {(r["rank"], r["pid"], r["weight_storage_digest"]) for r in receipts}
                assert identities is None or current == identities
                identities = current
                (root / (name + "-status.json")).write_text(json.dumps(receipts, indent=2))
                if client.request("/is_paused", method="GET")["is_paused"]:
                    client.resume()
                print("TRIAL", name, flush=True)

                def record(row, candidate=name):
                    row["candidate"] = candidate
                    output.write(json.dumps(row) + "\n")
                    output.flush()
                    print(
                        json.dumps(
                            {
                                k: row.get(k)
                                for k in ("candidate", "case_id", "passed", "decode_tokens_per_s", "ttft_s", "error")
                            }
                        ),
                        flush=True,
                    )

                groups = make_groups(trial.get("workloads", []), [])
                for size in trial.get("cold", []):
                    groups.append(
                        (f"cold{size}", [retrieval_case(size, variant=trial["variant"], count_tokens=count_tokens)])
                    )
                if trial.get("cold_first"):
                    groups.sort(key=lambda group: not group[0].startswith("cold"))
                rows = run_groups(
                    groups,
                    "http://127.0.0.1:8001",
                    "glm53-flash-selective-w3",
                    42,
                    on_result=record,
                    tokenizer_identity=tokenizer_identity,
                    timeout_s=1200,
                )
                summaries[name] = summarize(rows)
                (root / (args.plan.stem + "-summary.json")).write_text(json.dumps(summaries, indent=2))
                with urllib.request.urlopen("http://127.0.0.1:8001/metrics", timeout=10) as response:
                    (root / (name + "-metrics.txt")).write_bytes(response.read())
                print("SUMMARY", name, json.dumps(summaries[name]), flush=True)
                if not summaries[name]["valid"]:
                    raise RuntimeError("invalid serving trial: " + name)
    finally:
        client.switch(Control(uuid.uuid4().hex))
        if client.request("/is_paused", method="GET")["is_paused"]:
            client.resume()


if __name__ == "__main__":
    main()
