"""Pure and overlapping serving shadows, returning the retained outputs."""

import concurrent.futures
import json
import threading
import time
import urllib.request
import uuid

from compare import ROOT, admit, control, probe, query


def decode(started):
    body = dict(
        model="qwen38-flash-next",
        messages=[
            dict(
                role="user",
                content="Experiment "
                + uuid.uuid4().hex
                + ". Explain how a database transaction preserves consistency.",
            )
        ],
        temperature=0,
        max_tokens=192,
        min_tokens=192,
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
    times = []
    usage = None
    with urllib.request.urlopen(request, timeout=180) as response:
        for line in response:
            if not line.startswith(b"data: ") or line[6:].strip() == b"[DONE]":
                continue
            event = json.loads(line[6:])
            assert "error" not in event, event
            if event.get("usage"):
                usage = event["usage"]
            for choice in event.get("choices", []):
                if choice.get("delta", {}).get("content"):
                    times.append(time.monotonic() - start)
                    started.set()
    assert times and usage and usage["completion_tokens"] == 192
    return dict(
        ttft_s=times[0],
        total_s=time.monotonic() - start,
        usage=usage,
        max_stream_gap_s=max(b - a for a, b in zip(times, times[1:])),
    )


if __name__ == "__main__":
    record = {}
    try:
        record["before"] = admit(query(), max_temperature_c=85)
        assert record["before"]["passed"]
        record["pure"] = probe()
        started = threading.Event()
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            future = pool.submit(decode, started)
            assert started.wait(60), "decode never started"
            record["prefill"] = probe()
            record["decode"] = future.result(timeout=180)
        record["after"] = admit(query())
        control("status", "shadow-status")
    finally:
        (ROOT / "shadow-results.json").write_text(json.dumps(record, indent=2) + "\n")
        control("off", "shadow-final-off")
    print(json.dumps(record), flush=True)
