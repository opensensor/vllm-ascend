"""Exercise six simultaneous decoders while native PLE shadow checks are on."""

import concurrent.futures
import json
import sys
from pathlib import Path

from tools.qwen38_decode_study.benchmark import request_json

ROOT = Path("/srv/ai/storage/artifacts/qfn-tp6-scaleout-20261016")
run = ROOT / sys.argv[1]
assert run.parent == ROOT and run.name.startswith("resident-")
output = run / "six-decode-gate.json"
assert not output.exists()


def request(index):
    response = request_json(
        "http://127.0.0.1:8001",
        "/v1/chat/completions",
        dict(
            model="qwen38-flash-next",
            messages=[dict(role="user", content=f"For reviewer {index}, explain how a hash table works in detail.")],
            max_tokens=256,
            ignore_eos=True,
            temperature=0,
            seed=42,
            chat_template_kwargs=dict(enable_thinking=False),
        ),
    )
    assert response["usage"]["completion_tokens"] == 256
    assert response["choices"][0]["message"]["content"]
    return response


with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
    responses = list(pool.map(request, range(6)))
output.write_text(json.dumps(dict(passed=True, responses=responses), indent=2) + "\n")
print("Six concurrent 256-token completions passed; inspect PLE shadow receipts for 18-token coverage.", flush=True)
