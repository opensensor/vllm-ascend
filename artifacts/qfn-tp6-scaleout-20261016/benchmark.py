"""Matched C1/C6 streaming measurements with fixed generated-token budgets."""

import concurrent.futures
import json
import statistics
import sys
import time
from pathlib import Path

from tools.qwen4exp.thermal_guard import admit, query
from tools.qwen38_decode_study.benchmark import PROMPTS, request_json, stream_completion

ROOT = Path("/srv/ai/storage/artifacts/qfn-tp6-scaleout-20261016")
label = sys.argv[1]
concurrent_tokens = int(sys.argv[2]) if len(sys.argv) > 2 else 128
assert concurrent_tokens in (128, 512)
path = ROOT / (label + "-benchmark.json")
assert not path.exists()
base = "http://127.0.0.1:8001"
model = request_json(base, "/v1/models")["data"][0]["id"]


def sample(index, length):
    return stream_completion(
        base,
        dict(
            model=model,
            messages=[dict(role="user", content=PROMPTS[index % 3])],
            max_tokens=length,
            temperature=0,
            seed=42,
            ignore_eos=True,
            chat_template_kwargs=dict(enable_thinking=False),
            stream=True,
            stream_options=dict(include_usage=True),
        ),
    )


report = dict(label=label, concurrent_tokens=concurrent_tokens, serial=[], concurrent=[])


def save():
    path.write_text(json.dumps(report, indent=2) + "\n")


assert admit(query(), devices=tuple(range(6)), max_temperature_c=90)["passed"]
report["warmup"] = sample(0, 32)
save()
for i in range(3):
    assert admit(query(), devices=tuple(range(6)), max_temperature_c=90)["passed"]
    result = sample(i, 256)
    assert result["usage"]["completion_tokens"] == 256
    report["serial"].append(result)
    save()
assert admit(query(), devices=tuple(range(6)), max_temperature_c=90)["passed"]
start = time.perf_counter()
with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
    report["concurrent"] = list(pool.map(lambda i: sample(i, concurrent_tokens), range(6)))
report["concurrent_wall_s"] = time.perf_counter() - start
assert all(r["usage"]["completion_tokens"] == concurrent_tokens for r in report["concurrent"])
report["serial_median_decode_tok_s"] = statistics.median(r["client_decode_tok_s"] for r in report["serial"])
report["concurrent_aggregate_e2e_tok_s"] = (
    sum(r["usage"]["completion_tokens"] for r in report["concurrent"]) / report["concurrent_wall_s"]
)
report["temperatures_after"] = admit(query(), devices=tuple(range(6)), max_temperature_c=90)
save()
print(json.dumps({k: v for k, v in report.items() if k not in ("serial", "concurrent", "warmup")}, indent=2))
