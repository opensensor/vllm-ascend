"""Check public inference, admissions and bounded hardware health."""

import datetime
import json
import urllib.request
from pathlib import Path

from tools.qwen4exp.thermal_guard import admit, query
from tools.sidecard.qfn_resident_server import QfnResidentClient

ROOT = Path("/srv/ai/storage/artifacts/qfn-local-pack-20261016")
RUN = Path("/srv/ai/storage/artifacts/npu-collective-qfn-20261014-024609/resident-live-v16")
BASE_URL = "http://127.0.0.1:8001"


def main():
    client = QfnResidentClient(BASE_URL, RUN / "control.token")
    with urllib.request.urlopen(BASE_URL + "/health", timeout=10) as response:
        health = response.status
    with urllib.request.urlopen(BASE_URL + "/v1/models", timeout=10) as response:
        models = [row["id"] for row in json.load(response)["data"]]
    body = dict(
        model="qwen38-flash-next",
        messages=[dict(role="user", content="What is 17 times 23? Reply with only the number.")],
        max_tokens=64,
        temperature=0,
        chat_template_kwargs=dict(enable_thinking=False),
    )
    request = urllib.request.Request(
        BASE_URL + "/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        result = json.load(response)
        answer = result["choices"][0]["message"]["content"]
    paused = client.request("/is_paused", method="GET")["is_paused"]
    state = dict(
        utc=datetime.datetime.now(datetime.UTC).isoformat(),
        health=health,
        models=models,
        is_paused=paused,
        smoke_answer=answer,
        finish_reason=result["choices"][0]["finish_reason"],
        usage=result["usage"],
        hardware=admit(query()),
        dma_query_spin_us=Path("/sys/module/drv_pcie_host/parameters/dma_query_spin_us").read_text().strip(),
    )
    (ROOT / "public-state.json").write_text(json.dumps(state, indent=2) + "\n")
    print(json.dumps(state), flush=True)
    assert health == 200 and "qwen38-flash-next" in models and not paused and "391" in answer
    assert state["hardware"]["passed"]


if __name__ == "__main__":
    main()
