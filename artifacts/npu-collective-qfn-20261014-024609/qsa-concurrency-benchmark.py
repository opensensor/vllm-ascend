"""Bounded simultaneous HTTP requests against the unchanged resident service."""

import argparse
import concurrent.futures
import json
import statistics
import threading
import time
import urllib.request
from pathlib import Path

from tools.qwen38_decode_study.benchmark import stream_completion
from tools.sidecard.qfn_resident_server import QfnResidentClient

ROOT = Path("/srv/ai/storage/artifacts/npu-collective-qfn-20261014-024609")
PROMPTS = (
    "Explain how a hash table works and discuss collision resolution strategies.",
    "Explain how a database index works and discuss its performance tradeoffs.",
    "Explain how TCP provides reliable delivery and handles lost packets.",
    "Explain how an LRU cache works and discuss a practical implementation.",
)


def metrics(client):
    request = urllib.request.Request(client.base_url + "/metrics", headers={"x-qfn-resident-token": client.token})
    names = (
        "vllm:num_requests_running",
        "vllm:num_requests_waiting",
        "vllm:request_success_total",
        "vllm:spec_decode_num_draft_tokens_total",
        "vllm:spec_decode_num_accepted_tokens_total",
    )
    values = {name: 0.0 for name in names}
    with urllib.request.urlopen(request, timeout=15) as response:
        for line in response.read().decode().splitlines():
            for name in names:
                if line.startswith(name + "{") or line.startswith(name + " "):
                    values[name] += float(line.rsplit(" ", 1)[1])
    return values


def identity(rows):
    return sorted((row["rank"], row["pid"], row["weight_storage_digest"]) for row in rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    parser.add_argument("--run-root", type=Path, default=ROOT / "resident-live-v3")
    parser.add_argument("--concurrency", type=int, choices=(1, 2, 3, 4), default=4)
    parser.add_argument("--tokens", type=int, default=64)
    args = parser.parse_args()
    if not 32 <= args.tokens <= 512 or Path(args.label).name != args.label:
        raise ValueError("use a plain label and 32..512 tokens")
    output = ROOT / "qsa-resident-v1" / f"concurrency-{args.label}.json"
    if output.exists():
        raise ValueError("preserve existing evidence; choose a new label")
    client = QfnResidentClient("http://127.0.0.1:8001", args.run_root / "control.token")
    before_metrics = metrics(client)
    if before_metrics["vllm:num_requests_running"] or before_metrics["vllm:num_requests_waiting"]:
        raise RuntimeError("server is serving other requests; defer measurement")
    before = client.rpc("resident_status")
    assert len(before) == 4 and all(not row["graphs_dirty"] and not row["native_failed"] for row in before)
    assert client.request("/is_paused", method="GET")["is_paused"] is False
    result = dict(
        concurrency=args.concurrency,
        tokens_per_request=args.tokens,
        before=before,
        metrics_before=before_metrics,
        graph_inventory=client.rpc("qfn_graph_inventory"),
        scope=(
            "Synchronized finite C requests; HTTP aggregate includes prefill and tails. SSE decode uses N-1 estimate."
        ),
        requests=[],
    )
    output.write_text(json.dumps(result, indent=2) + "\n")
    barrier = threading.Barrier(args.concurrency, timeout=15)

    def run(index):
        request = dict(
            model="qwen38-flash-next",
            messages=[dict(role="user", content=PROMPTS[index])],
            temperature=0,
            seed=42,
            max_tokens=args.tokens,
            min_tokens=args.tokens,
            ignore_eos=True,
            chat_template_kwargs={"enable_thinking": False},
            stream=True,
            stream_options={"include_usage": True},
        )
        barrier.wait()
        started = time.perf_counter()
        receipt = stream_completion(client.base_url, request)
        finished = time.perf_counter()
        assert receipt["usage"]["completion_tokens"] == args.tokens
        assert receipt["finish_reason"] == "length" and receipt["text"].strip()
        return dict(index=index, started=started, finished=finished, request=request, **receipt)

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as executor:
            futures = [executor.submit(run, index) for index in range(args.concurrency)]
            for future in concurrent.futures.as_completed(futures):
                row = future.result()
                result["requests"].append(row)
                output.write_text(json.dumps(result, indent=2) + "\n")
                print(
                    json.dumps({key: row[key] for key in ("index", "ttft_s", "client_decode_tok_s", "e2e_s")}),
                    flush=True,
                )
    except BaseException as error:
        result["error"] = f"{type(error).__name__}: {error}"
        output.write_text(json.dumps(result, indent=2) + "\n")
        raise
    result["after"] = client.rpc("resident_status")
    result["metrics_after"] = metrics(client)
    assert identity(before) == identity(result["after"])
    assert all(not row["graphs_dirty"] and not row["native_failed"] for row in result["after"])
    rows = result["requests"]
    result["makespan_s"] = max(row["finished"] for row in rows) - min(row["started"] for row in rows)
    result["start_spread_s"] = max(row["started"] for row in rows) - min(row["started"] for row in rows)
    result["aggregate_http_output_tok_s"] = args.concurrency * args.tokens / result["makespan_s"]
    result["median_client_decode_tok_s"] = statistics.median(row["client_decode_tok_s"] for row in rows)
    result["requests"].sort(key=lambda row: row["index"])
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "concurrency",
                    "tokens_per_request",
                    "makespan_s",
                    "start_spread_s",
                    "aggregate_http_output_tok_s",
                    "median_client_decode_tok_s",
                )
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
