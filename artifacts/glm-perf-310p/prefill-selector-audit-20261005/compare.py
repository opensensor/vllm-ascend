# SPDX-License-Identifier: Apache-2.0
import json
import uuid
from pathlib import Path

from tools.glm_perf.resident_control import Control
from tools.glm_perf.resident_harness import ResidentClient
from tools.glm_perf.suite import load_tokenizer_json, retrieval_case, run_groups

root = Path(__file__).resolve().parent
client = ResidentClient("http://127.0.0.1:8001")
count, identity = load_tokenizer_json(Path("/srv/ai/models/GLM-5.3-Flash-selective-W3-310p/tokenizer.json"))
case = retrieval_case(8192, variant=7, count_tokens=count)
workers = None
try:
    with (root / "serving-ab.jsonl").open("w") as out:
        for name in ("baseline", "native_prefill"):
            source = (
                Path(
                    "/srv/ai/src/glm-selective-w3-nz-test-20261004/tools/glm_perf/resident_candidates/kpool_prefill_direct.py"
                ).read_text()
                if name != "baseline"
                else ""
            )
            receipts = client.switch(Control(uuid.uuid4().hex, candidate=name, source=source))
            current = {(r["rank"], r["pid"], r["weight_storage_digest"]) for r in receipts}
            assert workers is None or current == workers
            workers = current
            print("ROUND", name, flush=True)

            def record(row, name=name):
                row["candidate"] = name
                out.write(json.dumps(row) + "\n")
                out.flush()
                print(
                    json.dumps({k: row.get(k) for k in ("candidate", "ttft_s", "passed", "content", "error")}),
                    flush=True,
                )

            results = run_groups(
                [("cold_8k", [case])],
                "http://127.0.0.1:8001",
                "glm53-flash-selective-w3",
                42,
                on_result=record,
                tokenizer_identity=identity,
                timeout_s=1200,
            )
            if any(r.get("error") for r in results):
                raise RuntimeError("request failed")
finally:
    restored = client.switch(Control(uuid.uuid4().hex))
    if client.request("/is_paused", method="GET")["is_paused"]:
        client.resume()
    (root / "comparison-restored.json").write_text(json.dumps(restored, indent=2))
