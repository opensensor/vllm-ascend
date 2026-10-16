"""Short counterbalanced serving comparison with temperature receipts."""

import hashlib
import json
import subprocess
import sys
import time
import urllib.request
import uuid
from pathlib import Path

from tools.qwen4exp.thermal_guard import admit, query

ROOT = Path("/srv/ai/storage/artifacts/qfn-local-pack-20261016")


def probe():
    paragraph = (
        "The cache stores immutable entries indexed by a stable key. Readers verify ownership before use. "
        "Writers allocate a new version and retire older entries only after pending readers finish. "
        "Requests may arrive concurrently, so the scheduler tracks their progress independently.\n"
    )
    content = (
        f"Experiment {uuid.uuid4().hex}. Read these engineering notes.\n" + paragraph * 80 + "Summarize the design."
    )
    body = dict(
        model="qwen38-flash-next",
        messages=[dict(role="user", content=content)],
        temperature=0,
        max_tokens=16,
        min_tokens=16,
        ignore_eos=True,
        chat_template_kwargs=dict(enable_thinking=False),
        stream=True,
        stream_options=dict(include_usage=True),
    )
    request = urllib.request.Request(
        "http://127.0.0.1:8001/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    start = time.monotonic()
    times, parts, usage = [], [], None
    with urllib.request.urlopen(request, timeout=180) as response:
        for line in response:
            if not line.startswith(b"data: ") or line[6:].strip() == b"[DONE]":
                continue
            event = json.loads(line[6:])
            assert "error" not in event, event
            if event.get("usage"):
                usage = event["usage"]
            for choice in event.get("choices", []):
                text = choice.get("delta", {}).get("content")
                if text:
                    times.append(time.monotonic() - start)
                    parts.append(text)
    assert usage and usage["completion_tokens"] == 16 and times
    return dict(
        usage=usage,
        ttft_s=times[0],
        total_s=time.monotonic() - start,
        output_sha256=hashlib.sha256("".join(parts).encode()).hexdigest(),
    )


def control(action, label):
    subprocess.run([sys.executable, str(ROOT / "local-control.py"), action, label], check=True)


if __name__ == "__main__":
    output = ROOT / "local-comparison.json"
    assert not output.exists()
    records = []
    admission = admit(query(), max_temperature_c=85)
    assert admission["passed"], admission
    shadow = probe()
    (ROOT / "local-shadow.json").write_text(json.dumps(shadow, indent=2) + "\n")
    control("status", "local-shadow-status")
    time.sleep(20)
    try:
        for index, action in enumerate(("off", "on", "on", "off")):
            control(action, f"local-prefill-{index}-{action}")
            admission = admit(query(), max_temperature_c=85)
            if not admission["passed"]:
                raise RuntimeError(f"benchmark admission rejected: {admission}")
            result = dict(action=action, before=admission, measurement=probe(), after=admit(query()))
            records.append(result)
            output.write_text(json.dumps(records, indent=2) + "\n")
            print("MEASUREMENT", json.dumps(result), flush=True)
            if index < 3:
                time.sleep(15)
    finally:
        control("on", "local-final-on")
