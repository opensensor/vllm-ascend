# SPDX-License-Identifier: Apache-2.0
import json
import time
from pathlib import Path
import urllib.error
from tools.glm_perf.resident_harness import ResidentClient
from tools.glm_perf.resident_native import NativeManifest
root = Path(__file__).resolve().parent
client = ResidentClient("http://127.0.0.1:8001")
for _ in range(180):
    try:
        client.request("/health", method="GET")
        break
    except OSError:
        time.sleep(2)
else:
    raise RuntimeError("server not ready")
def save(name, data):
    (root / name).write_text(json.dumps(data, indent=2) + "\n")
request = {"model": "glm53-flash-selective-w3", "messages": [{"role":"user", "content":"What is 7 times 8? Answer with only the number."}], "temperature":0, "seed":42, "max_tokens":32, "chat_template_kwargs":{"reasoning_effort":"low"}}
before = client.rpc("resident_status")
save("load-before-status.json", before)
assert all(not worker["native_loaded"] for worker in before)
first = client.request("/v1/chat/completions", request)
save("request-before.json", first)
manifest = NativeManifest(json.loads((root / "prefill-v1.json").read_text()))
start = time.monotonic()
receipts = client.load_native(manifest)
save("load-receipts.json", receipts)
print(json.dumps({"event":"loaded", "elapsed_s":time.monotonic()-start,"receipts":receipts}), flush=True)
after = client.rpc("resident_status")
save("load-after-status.json", after)
second = client.request("/v1/chat/completions", request)
save("request-after.json", second)
assert first["choices"][0]["message"]["content"] == second["choices"][0]["message"]["content"]
assert "56" in second["choices"][0]["message"]["content"]
repeat = client.load_native(manifest)
save("load-repeat-receipts.json", repeat)
print(json.dumps({"event":"complete", "content":second["choices"][0]["message"]["content"], "workers":[r["pid"] for r in after], "active_candidates":[r["candidate"] for r in after]}), flush=True)
