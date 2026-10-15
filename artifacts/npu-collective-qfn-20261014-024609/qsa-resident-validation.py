"""Exercise resident QSA boundaries and concurrency without changing weights."""

import argparse
import concurrent.futures
import json
import time
from pathlib import Path

from tools.sidecard.qfn_resident_server import QfnResidentClient


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--parallel-requests", type=int, choices=(2, 4), default=2)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("preserve existing validation receipts")
    client = QfnResidentClient("http://127.0.0.1:8001", args.run_root / "control.token")
    result = {"before": client.rpc("resident_status"), "trials": []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    base = client.request(
        "/tokenize",
        {
            "model": "qwen38-flash-next",
            "messages": [{"role": "user", "content": "Explain why a hash table needs collision resolution."}],
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )["tokens"]
    filler = client.request(
        "/tokenize",
        {
            "model": "qwen38-flash-next",
            "prompt": "A hash table stores keys and values in buckets. ",
            "add_special_tokens": False,
        },
    )["tokens"]

    def run_case(label, length, budget):
        extra = length - len(base)
        prompt = base[:3] + (filler * ((extra + len(filler) - 1) // len(filler)))[:extra] + base[3:]
        request = dict(
            model="qwen38-flash-next",
            prompt=prompt,
            temperature=0,
            max_tokens=budget,
            min_tokens=budget,
            ignore_eos=True,
            return_token_ids=True,
        )
        started = time.monotonic()
        response = client.request("/v1/completions", request)
        elapsed = time.monotonic() - started
        assert response["usage"]["prompt_tokens"] == length
        assert response["usage"]["completion_tokens"] == budget
        text = response["choices"][0]["text"].strip()
        # A minimal semantic smoke gate for this fixed hash-table prompt. This
        # catches the observed corrupted prefix; it is not a quality benchmark.
        assert "hash" in text.lower() or "collision" in text.lower(), text[:240]
        return dict(label=label, prompt_tokens=length, output_tokens=budget, seconds=elapsed, response=response)

    def save(rows):
        result["trials"].extend(rows)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        for row in rows:
            print(json.dumps({key: value for key, value in row.items() if key != "response"}), flush=True)

    for label, length, budget in (
        ("short_before", len(base), 64),
        ("cross_boundary", 2030, 96),
        ("above_boundary", 2100, 64),
        ("long_context", 8192, 64),
        ("short_after_long", len(base), 64),
    ):
        save([run_case(label, length, budget)])
    # Packed requests include a boundary-crossing sequence and mixed contexts.
    lengths = (512, 2100) if args.parallel_requests == 2 else (512, 2030, 2100, 4096)
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.parallel_requests) as executor:
        futures = [executor.submit(run_case, f"concurrent_{index}", length, 96) for index, length in enumerate(lengths)]
        save([future.result() for future in futures])
    save([run_case("short_after_concurrent", len(base), 64)])
    result["after"] = client.rpc("resident_status")
    identity = lambda rows: sorted((row["rank"], row["pid"], row["weight_storage_digest"]) for row in rows)
    assert identity(result["before"]) == identity(result["after"])
    assert all(not row["native_failed"] and not row["graphs_dirty"] for row in result["after"])
    result["inventory"] = client.request("/collective_rpc", {"method": "qfn_graph_inventory", "timeout": 60})
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
