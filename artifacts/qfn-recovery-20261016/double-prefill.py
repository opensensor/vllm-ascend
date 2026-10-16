"""Exercise overlapping long prefills and first-decode transitions over HTTP."""

import argparse
import concurrent.futures
import json
import threading
import time
import urllib.request
import uuid
from pathlib import Path

from tools.qwen4exp.thermal_guard import admit, query

ROOT = Path("/srv/ai/storage/artifacts/qfn-recovery-20261016")


def stream(paragraphs, barrier, headers_ready=None):
    content = (
        f"Experiment {uuid.uuid4().hex}. Read these engineering notes.\n"
        + (
            "The cache stores immutable entries indexed by a stable key. Readers verify ownership before use. "
            "Writers allocate a new version and retire older entries only after pending readers finish. "
            "Requests may arrive concurrently, so the scheduler tracks their progress independently.\n" * paragraphs
        )
        + "\nSummarize the design."
    )
    body = dict(
        model="qwen38-flash-next",
        messages=[dict(role="user", content=content)],
        temperature=0,
        max_tokens=64,
        min_tokens=64,
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
    barrier.wait(timeout=30)
    start = time.monotonic()
    times, usage = [], None
    with urllib.request.urlopen(request, timeout=900) as response:
        if headers_ready is not None:
            headers_ready.set()
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
    assert usage and usage["completion_tokens"] == 64 and times
    return dict(
        paragraphs=paragraphs,
        started_at=start,
        usage=usage,
        ttft_s=times[0],
        total_s=time.monotonic() - start,
        max_content_gap_s=max((b - a for a, b in zip(times, times[1:])), default=0),
    )


class OrderedAdmission:
    def __init__(self, event, delay):
        self.event, self.delay = event, delay

    def wait(self, timeout):
        if self.event is not None and not self.event.wait(timeout):
            raise TimeoutError("first request did not receive HTTP headers")
        if self.delay:
            time.sleep(self.delay)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", default="double-prefill")
    parser.add_argument("--paragraphs", type=int, nargs=2, default=(450, 500))
    parser.add_argument("--pairs", type=int, choices=(1, 2), default=2)
    parser.add_argument("--stagger-second", type=float, default=0)
    args = parser.parse_args()
    assert 0 <= args.stagger_second <= 5
    assert Path(args.label).name == args.label
    path = ROOT / (args.label + ".json")
    assert not path.exists()
    records = []
    assert all(count > 0 for count in args.paragraphs)
    for counts in (args.paragraphs, args.paragraphs[::-1])[: args.pairs]:
        admission = admit(query())
        if not admission["passed"]:
            raise RuntimeError(f"benchmark admission rejected: {admission}")
        barrier = threading.Barrier(2)
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            if args.stagger_second:
                ready = threading.Event()
                futures = [
                    pool.submit(stream, counts[0], OrderedAdmission(None, 0), ready),
                    pool.submit(stream, counts[1], OrderedAdmission(ready, args.stagger_second)),
                ]
            else:
                futures = [pool.submit(stream, count, barrier) for count in counts]
            records.append([future.result() for future in futures])
        path.write_text(json.dumps(records, indent=2) + "\n")
        print(json.dumps(records[-1]), flush=True)
