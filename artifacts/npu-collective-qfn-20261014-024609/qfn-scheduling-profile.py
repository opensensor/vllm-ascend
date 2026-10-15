"""Bounded all-rank profiling of the retained C1/C4 graph configurations."""

import argparse
import concurrent.futures
import json
import subprocess
import threading
import time
import urllib.request
from pathlib import Path

from tools.sidecard.qfn_resident_server import QfnResidentClient

ROOT = Path("/srv/ai/storage/artifacts/npu-collective-qfn-20261014-024609")
OUTPUT_BASE = ROOT / "resident-live-v6/native-timeline-scheduling"
PROMPTS = (
    "Explain how a hash table works and discuss collision resolution strategies.",
    "Explain how a database index works and discuss its performance tradeoffs.",
    "Explain how TCP provides reliable delivery and handles lost packets.",
    "Explain how an LRU cache works and discuss a practical implementation.",
)


def metrics(client):
    request = urllib.request.Request(client.base_url + "/metrics", headers={"x-qfn-resident-token": client.token})
    with urllib.request.urlopen(request, timeout=15) as response:
        return response.read().decode()


def sensors():
    return subprocess.run(["npu-smi", "info"], capture_output=True, text=True, timeout=15, check=True).stdout


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tag", choices=("c1-prefix", "c4-prefix", "c1-general"))
    parser.add_argument("--label", required=True)
    args = parser.parse_args()
    if Path(args.label).name != args.label:
        raise ValueError("plain label required")
    OUTPUT = OUTPUT_BASE / args.label
    count = 4 if args.tag == "c4-prefix" else 1
    output = OUTPUT / f"{args.tag}-receipt.json"
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    client = QfnResidentClient("http://127.0.0.1:8001", ROOT / "resident-live-v6/control.token")
    before_metrics = metrics(client)
    for line in before_metrics.splitlines():
        if line.startswith(("vllm:num_requests_running{", "vllm:num_requests_waiting{")):
            if float(line.rsplit(" ", 1)[1]):
                raise RuntimeError("defer profiling while other requests are active")
    assert client.request("/is_paused", method="GET")["is_paused"] is False
    before = client.rpc("resident_status")
    assert len(before) == 4 and all(
        row["candidate"] == "qsa_combined_queued_observer"
        and not row["graphs_dirty"]
        and not row["native_failed"]
        for row in before
    )
    inventory = client.rpc("qfn_graph_inventory")
    record = dict(
        tag=args.tag,
        before=before,
        inventory=inventory,
        metrics_before=before_metrics,
        sensors_before=sensors(),
        scope="Eight profiled engine iterations after eight warmup iterations; not throughput evidence.",
    )
    output.write_text(json.dumps(record, indent=2) + "\n")
    barrier = threading.Barrier(count, timeout=15)

    def request(index):
        prompt = PROMPTS[index]
        if args.tag == "c1-general":
            prompt = "A hash table stores keys and values in buckets. " * 230 + "\n" + prompt
        payload = dict(
            model="qwen38-flash-next",
            messages=[dict(role="user", content=prompt)],
            chat_template_kwargs={"enable_thinking": False},
            temperature=0,
            seed=42,
            max_tokens=128,
            min_tokens=128,
            ignore_eos=True,
        )
        barrier.wait()
        start = time.monotonic()
        response = client.request("/v1/chat/completions", payload)
        assert response["usage"]["completion_tokens"] == 128
        if args.tag == "c1-general":
            assert response["usage"]["prompt_tokens"] > 2051
        return dict(index=index, response=response, seconds=time.monotonic() - start)

    client.request("/pause?mode=wait&clear_cache=false")
    try:
        record["start"] = client.rpc(
            "qfn_native_profile_start", json.dumps(dict(output_dir=str(OUTPUT), tag=args.tag, delay=8, max=8))
        )
    finally:
        client.request("/resume")
    try:
        assert all(row.get("armed") and "error" not in row for row in record["start"])
        with concurrent.futures.ThreadPoolExecutor(max_workers=count) as executor:
            record["responses"] = list(executor.map(request, range(count)))
    finally:
        client.request("/pause?mode=wait&clear_cache=false")
        try:
            record["stop"] = client.rpc("qfn_native_profile_stop")
        finally:
            client.request("/resume")
        record["after"] = client.rpc("resident_status")
        record["metrics_after"] = metrics(client)
        record["sensors_after"] = sensors()
        output.write_text(json.dumps(record, indent=2) + "\n")
    assert all(row.get("ownership_released") and row.get("recorded") for row in record["stop"])
    identity = lambda rows: sorted((r["rank"], r["pid"], r["weight_storage_digest"]) for r in rows)
    assert identity(record["after"]) == identity(before)
    assert all(not row["graphs_dirty"] and not row["native_failed"] for row in record["after"])
    print(
        json.dumps(
            dict(tag=args.tag, stopped=record["stop"], usages=[r["response"]["usage"] for r in record["responses"]])
        ),
        flush=True,
    )
    print(record["sensors_after"], flush=True)


if __name__ == "__main__":
    main()
