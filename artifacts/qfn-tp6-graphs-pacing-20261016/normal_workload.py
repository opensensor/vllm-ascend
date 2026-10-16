# SPDX-License-Identifier: Apache-2.0
import hashlib
import json
import statistics
import threading
import time
import urllib.request
from pathlib import Path
from types import SimpleNamespace

root = Path("/srv/ai/storage/artifacts/qfn-tp6-graphs-pacing-20261016")
client = SimpleNamespace(base_url="http://127.0.0.1:8001")
trigger = threading.Event()
output = root


def stream(name, content, tokens):
    body = dict(
        model="qwen38-flash-next",
        messages=[dict(role="user", content=content)],
        temperature=0,
        seed=42,
        max_tokens=tokens,
        chat_template_kwargs=dict(enable_thinking=False),
        stream=True,
        stream_options=dict(include_usage=True),
    )
    request = urllib.request.Request(
        client.base_url + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    result = dict(
        name=name,
        start_ns=time.monotonic_ns(),
        events=[],
        content=[],
        prompt_sha256=hashlib.sha256(content.encode()).hexdigest(),
    )
    with urllib.request.urlopen(request, timeout=600) as response:
        for line in response:
            if not line.startswith(b"data: "):
                continue
            raw = line[6:].strip()
            if raw == b"[DONE]":
                break
            event = json.loads(raw)
            if "error" in event:
                raise RuntimeError(event["error"])
            result["request_id"] = event.get("id", result.get("request_id"))
            if event.get("usage"):
                result["usage"] = event["usage"]
            for choice in event.get("choices", []):
                text = choice.get("delta", {}).get("content")
                if text:
                    result["content"].append(text)
                    result["events"].append(dict(time_ns=time.monotonic_ns(), characters=len(text)))
                    if name == "decode" and len(result["events"]) >= 8:
                        trigger.set()
    result["end_ns"] = time.monotonic_ns()
    events = [e["time_ns"] for e in result["events"]]
    gaps = [(b - a) / 1e9 for a, b in zip(events, events[1:])]
    result["ttft_s"] = (events[0] - result["start_ns"]) / 1e9
    result["max_content_gap_s"] = max(gaps, default=0)
    result["median_content_gap_s"] = statistics.median(gaps) if gaps else None
    (output / f"{name}-stream.json").write_text(json.dumps(result, indent=2))
    return result
