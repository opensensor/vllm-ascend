import concurrent.futures
import hashlib
import json
import statistics
import subprocess
import threading
import time
import urllib.request
import uuid
from pathlib import Path
from types import SimpleNamespace

root = Path("/srv/ai/storage/artifacts/qfn-tp6-mixed-20261016")
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
        min_tokens=tokens,
        ignore_eos=True,
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
                    result["events"].append(dict(time_ns=time.monotonic_ns(), characters=len(text)))
                    if name == "decode" and len(result["events"]) >= 8:
                        trigger.set()
    result["end_ns"] = time.monotonic_ns()
    events = [e["time_ns"] for e in result["events"]]
    gaps = [(b - a) / 1e9 for a, b in zip(events, events[1:])]
    result["ttft_s"] = (events[0] - result["start_ns"]) / 1e9
    result["max_content_gap_s"] = max(gaps, default=0)
    result["median_content_gap_s"] = statistics.median(gaps) if gaps else None
    assert result["usage"]["completion_tokens"] == tokens
    (output / f"{name}-stream.json").write_text(json.dumps(result, indent=2))
    return result


def prompt():
    return (
        f"Experiment {uuid.uuid4().hex}. Read these engineering notes.\n"
        + (
            "The cache stores immutable entries indexed by a stable key. Readers verify ownership before use. "
            "Writers allocate a new version and retire older entries only after pending readers finish. "
            "Requests may arrive concurrently, so the scheduler tracks their progress independently.\n" * 70
        )
        + "Summarize the design in one sentence."
    )


def control(enabled, label):
    action = "enable" if enabled else "disable"
    subprocess.run(
        [
            "python3",
            str(root / "control.py"),
            "--source",
            str(root / f"shared-{action}-v3.py"),
            "--output",
            str(root / f"ab-{label}-control.json"),
            "--apply",
        ],
        check=True,
    )


results = []
try:
    for label, enabled in [("a1", False), ("b1", True), ("b2", True), ("a2", False), ("a3", False), ("b3", True)]:
        control(enabled, label)
        output = root / f"ab-{label}"
        output.mkdir(exist_ok=False)
        for i in range(2):
            row = stream(f"prefill{i}", prompt(), 16)
            results.append(dict(label=label, enabled=enabled, kind="prefill", **row))
            print(json.dumps(dict(label=label, kind="prefill", ttft_s=row["ttft_s"], usage=row["usage"])), flush=True)
        trigger.clear()
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            decoder = pool.submit(
                stream,
                "decode",
                f"Experiment {uuid.uuid4().hex}. Write a detailed numbered guide to "
                "implementing a hash table, with Python examples and tests.",
                512,
            )
            assert trigger.wait(60)
            prefill = pool.submit(stream, "prefill", prompt(), 16)
            for kind, task in [("mixed_prefill", prefill), ("mixed_decode", decoder)]:
                row = task.result()
                results.append(dict(label=label, enabled=enabled, kind=kind, **row))
                print(
                    json.dumps(
                        dict(
                            label=label,
                            kind=kind,
                            ttft_s=row["ttft_s"],
                            gap=row["max_content_gap_s"],
                            usage=row["usage"],
                        )
                    ),
                    flush=True,
                )
        (root / "ab-results.json").write_text(json.dumps(results, indent=2))
finally:
    control(True, "final-restore")
