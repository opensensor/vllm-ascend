"""Read final serving state and run one short real-weight request."""

import json
from pathlib import Path

from tools.sidecard.qfn_resident_server import QfnResidentClient

ROOT = Path("/srv/ai/storage/artifacts/qfn-mixed-prefill-20261015")
RUN = Path("/srv/ai/storage/artifacts/npu-collective-qfn-20261014-024609/resident-live-v13")


def main():
    client = QfnResidentClient("http://127.0.0.1:8001", RUN / "control.token", timeout=60)
    result = dict(
        models=client.request("/v1/models", method="GET"),
        paused=client.request("/is_paused", method="GET"),
        profilers=client.rpc("qfn_native_profile_status"),
        status=client.rpc("resident_status"),
    )
    assert not result["paused"]["is_paused"]
    assert all(not row["owned"] and not row["profiler_present"] for row in result["profilers"])
    result["smoke"] = client.request(
        "/v1/chat/completions",
        dict(
            model="qwen38-flash-next",
            messages=[dict(role="user", content="Reply with the word ready.")],
            max_tokens=16,
            temperature=0,
            chat_template_kwargs=dict(enable_thinking=False),
        ),
    )
    assert result["smoke"]["choices"][0]["message"]["content"].strip()
    (ROOT / "final-health.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            dict(
                passed=True, profilers=result["profilers"], content=result["smoke"]["choices"][0]["message"]["content"]
            )
        )
    )


if __name__ == "__main__":
    main()
