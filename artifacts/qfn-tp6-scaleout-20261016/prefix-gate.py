"""Cold multi-chunk retrieval and six concurrent branches sharing a prefix."""

import concurrent.futures
import json
import sys
from pathlib import Path

from tools.qwen38_decode_study.benchmark import request_json

ROOT = Path("/srv/ai/storage/artifacts/qfn-tp6-scaleout-20261016")
base = "http://127.0.0.1:8001"
context = "Read these inventory records carefully.\n" + "\n".join(
    f"Record {i}: warehouse section {i % 17}, shelf {i % 31}, quantity {100 + i}." for i in range(250)
)
context += "\nThe audit passphrase is KITE-731.\n"


def request(index):
    question = "\nState the audit passphrase, then explain in one sentence why record identifiers matter."
    if index is not None:
        question += f" Address your answer to auditor {index}."
    response = request_json(
        base,
        "/v1/chat/completions",
        dict(
            model="qwen38-flash-next",
            messages=[dict(role="user", content=context + question)],
            max_tokens=96,
            temperature=0,
            seed=42,
            chat_template_kwargs=dict(enable_thinking=False),
        ),
    )
    text = response["choices"][0]["message"].get("content") or ""
    return dict(index=index, passed="KITE-731" in text, response=response)


report = dict(cold=request(None))
path = ROOT / sys.argv[1] / "prefix-gate.json"
assert path.parent.parent == ROOT
assert not path.exists()
path.write_text(json.dumps(report, indent=2) + "\n")
assert report["cold"]["passed"]
with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
    report["branches"] = list(pool.map(request, range(6)))
path.write_text(json.dumps(report, indent=2) + "\n")
assert all(r["passed"] for r in report["branches"])
assert all(r["response"]["usage"]["prompt_tokens_details"]["cached_tokens"] > 0 for r in report["branches"])
print(
    json.dumps(
        dict(
            passed=True,
            cold_tokens=report["cold"]["response"]["usage"]["prompt_tokens"],
            cached_tokens=[
                r["response"]["usage"]["prompt_tokens_details"]["cached_tokens"] for r in report["branches"]
            ],
        )
    )
)
