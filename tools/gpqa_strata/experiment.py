"""Replay frozen GPQA prompts against an isolated Strata server and audit scores."""

import argparse
import hashlib
import json
import math
import os
import signal
import socket
import statistics
import subprocess
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import regex as re

CASES_SHA256 = "fba8e938bb45b148f2027eea5976db039a6b182070331920605ed63f94ff9fa3"
QUESTION_COUNT = 198
OUTPUT_LIMIT = 8192
ANSWER_PATTERN = re.compile(r"ANSWER\s*:\s*([A-D])", re.IGNORECASE)
GENERATION = {
    "ignore_eos": False,
    "min_p": 0.0,
    "presence_penalty": 0.0,
    "repetition_penalty": 1.0,
    "seed": 1024,
    "temperature": 0.0,
    "top_k": 20,
    "top_p": 1.0,
    "max_tokens": OUTPUT_LIMIT,
    "reasoning_effort": "high",  # Strata maps high to Qwen's xhigh template.
}


def digest(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def read_cases(path):
    if digest(path) != CASES_SHA256:
        raise ValueError("Cases differ from the frozen October 2 replay")
    cases = json.loads(path.read_text())
    if len(cases) != QUESTION_COUNT or [c["id"] for c in cases] != list(range(QUESTION_COUNT)):
        raise ValueError("Expected all 198 cases in the original order")
    if any(c["gold"] not in "ABCD" or len(c["gold"]) != 1 or not c["prompt"] for c in cases):
        raise ValueError("Invalid gold answer or empty prompt")
    return cases


def extract(text):
    matches = ANSWER_PATTERN.findall(text)
    return matches[-1].upper() if matches else None


def final_answer(row):
    if "final_answer" in row:
        return row["final_answer"]
    text = row["prediction"]
    if "</think>" in text:
        return extract(text.rsplit("</think>", 1)[-1])
    return None if "<think>" in text else extract(text)


def completion(case, url, model, timeout):
    body = {**GENERATION, "model": model, "stream": False, "messages": [{"role": "user", "content": case["prompt"]}]}
    request = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
    started = time.monotonic()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.load(response)
    choice = payload["choices"][0]
    message = choice["message"]
    reasoning = message.get("reasoning_content") or message.get("reasoning") or ""
    content = message.get("content") or ""
    combined = reasoning + "</think>" + content if reasoning else content
    # Old servers can return raw think tags in content; unfinished thinking is
    # credited only by the historical extractor, never as a final answer.
    final = content.rsplit("</think>", 1)[-1] if "</think>" in content else content
    if "<think>" in content and "</think>" not in content:
        final = ""
    if choice.get("finish_reason") not in ("stop", "length"):
        raise ValueError(f"Unexpected finish reason: {choice.get('finish_reason')!r}")
    tokens = payload["usage"]["completion_tokens"]
    if type(tokens) is not int or not 0 < tokens <= OUTPUT_LIMIT or not combined:
        raise ValueError("Missing text or invalid completion token usage")
    return {
        "id": case["id"],
        "gold": case["gold"],
        "success": True,
        "prompt_sha256": hashlib.sha256(case["prompt"].encode()).hexdigest(),
        "prediction": combined,
        "answer": extract(combined),
        "final_answer": extract(final),
        "output_tokens": tokens,
        "finish_reason": choice["finish_reason"],
        "seconds": time.monotonic() - started,
    }


def read_results(path):
    successes = {}
    if path.exists():
        for line in path.read_text().splitlines():
            row = json.loads(line)  # Refuse corrupt ledgers instead of silently dropping data.
            if row.get("success"):
                successes[row["id"]] = row
    return successes


def check_rows(cases, rows):
    by_id = {c["id"]: c for c in cases}
    for item_id, row in rows.items():
        case = by_id.get(item_id)
        if case is None or row["gold"] != case["gold"]:
            raise ValueError("Result IDs or labels differ from frozen cases")
        expected = hashlib.sha256(case["prompt"].encode()).hexdigest()
        if row.get("prompt_sha256", expected) != expected:
            raise ValueError("Result prompt digest differs from frozen cases")


def bind_manifest(path, identity):
    if path.exists():
        if json.loads(path.read_text()) != identity:
            raise ValueError("Refusing to resume a run with changed configuration or inputs")
    else:
        with path.open("x") as output:
            json.dump(identity, output, indent=2)


def summarize(cases, rows, baseline):
    check_rows(cases, rows)
    selected_ids = {c["id"] for c in cases}
    baseline = {item_id: row for item_id, row in baseline.items() if item_id in selected_ids}
    check_rows(cases, baseline)
    selected = [rows[c["id"]] for c in cases if c["id"] in rows]
    correct = sum(extract(r["prediction"]) == r["gold"] for r in selected)
    pairs = [(rows[c["id"]], baseline[c["id"]]) for c in cases if c["id"] in rows and c["id"] in baseline]
    wins = losses = 0
    for candidate, reference in pairs:
        current = extract(candidate["prediction"]) == candidate["gold"]
        previous = extract(reference["prediction"]) == reference["gold"]
        wins += current and not previous
        losses += previous and not current
    discordant = wins + losses
    p_value = min(1.0, 2 * sum(math.comb(discordant, k) for k in range(min(wins, losses) + 1)) / 2**discordant)
    rates = [r["output_tokens"] / r["seconds"] for r in selected if r.get("seconds", 0) > 0]
    return {
        "completed": len(selected),
        "expected": len(cases),
        "correct": correct,
        "full_198_score": len(selected) == len(cases) == QUESTION_COUNT,
        "accuracy_pct": 100 * correct / len(cases),
        "extracted": sum(extract(r["prediction"]) is not None for r in selected),
        "correct_final": sum(final_answer(r) == r["gold"] for r in selected),
        "final_extracted": sum(final_answer(r) is not None for r in selected),
        "output_limit_hits": sum(
            r.get("finish_reason") == "length" or r["output_tokens"] == OUTPUT_LIMIT for r in selected
        ),
        "cap_without_answer": sum(
            (r.get("finish_reason") == "length" or r["output_tokens"] == OUTPUT_LIMIT)
            and extract(r["prediction"]) is None
            for r in selected
        ),
        "median_client_output_tok_s": statistics.median(rates) if rates else None,
        "paired_items": len(pairs),
        "baseline_wrong_candidate_right": wins,
        "baseline_right_candidate_wrong": losses,
        "mcnemar_exact_two_sided_p": p_value,
    }


def stop_server(server):
    if server.poll() is None:
        os.killpg(server.pid, signal.SIGTERM)
        try:
            server.wait(timeout=20)
        except subprocess.TimeoutExpired:
            os.killpg(server.pid, signal.SIGKILL)
            server.wait()


def wait_ready(server, url):
    deadline = time.monotonic() + 600
    while time.monotonic() < deadline:
        if server.poll() is not None:
            raise RuntimeError("Strata exited during startup; inspect server.log")
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                health = json.load(response)
            if health.get("loaded") is True and health.get("status") == "ok":
                return health
        except (OSError, ValueError):
            pass
        time.sleep(2)
    raise TimeoutError("Strata startup exceeded 10 minutes")


def check_port_available(port):
    # Allow connections from our previous server to finish TIME_WAIT while
    # continuing to refuse a port owned by any live listening server.
    with socket.socket() as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        probe.bind(("127.0.0.1", port))


def evaluate(cases, args, config, baseline):
    """Own the server lifecycle and durably record each completed question."""
    config = {**config, "log": str(args.output.resolve() / "engine.log")}
    executable = Path(config["exe"])
    root = args.root.resolve()
    source = root / "Strata"
    python = source / ".venv/bin/python"
    args.output.mkdir(parents=True, exist_ok=True)
    for record in config["experiment"]["model_files"]:
        stat = Path(record["path"]).stat()
        if (stat.st_size, stat.st_mtime_ns) != (record["size"], record["mtime_ns"]):
            raise ValueError("Model files changed after checksum verification; rerun preparation")
    for record in config["experiment"]["runtime_files"]:
        if digest(Path(record["path"])) != record["sha256"]:
            raise ValueError("Packed weights or MTP runtime changed after preparation")
    if subprocess.check_output(["git", "-C", str(source), "status", "--porcelain", "--untracked-files=no"]):
        raise ValueError("Strata has modified source files; freeze a clean revision before scoring")
    identity = {
        "cases_sha256": CASES_SHA256,
        "ids": [c["id"] for c in cases],
        "config": config,
        "executable_sha256": digest(executable),
        "strata_revision": subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip(),
        "server_source_sha256": digest(source / "serve/server.py"),
        "tokenizer_sha256": {p.name: digest(p) for p in sorted(Path(config["tokenizer"]).iterdir()) if p.is_file()},
        "runner_sha256": digest(Path(__file__)),
        "generation": GENERATION,
        "baseline_sha256": digest(args.baseline),
        "client_concurrency": 1,
    }
    ledger = args.output / "results.jsonl"
    manifest_path = args.output / "manifest.json"
    if ledger.exists() and not manifest_path.exists():
        raise ValueError("Result ledger exists without its identity manifest")
    bind_manifest(manifest_path, identity)
    rows = read_results(ledger)
    check_rows(cases, rows)
    pending = [c for c in cases if c["id"] not in rows]
    if not pending:
        return summarize(cases, rows, baseline)
    gpu = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,uuid,driver_version,memory.total,memory.used", "--format=csv"], text=True
    )
    if "RTX PRO 6000" not in gpu:
        raise RuntimeError("This experiment requires the RTX PRO 6000")
    # Refuse an occupied port; do not interfere with an existing inference server.
    check_port_available(config["port"])
    invocation = {
        "utc": datetime.now(timezone.utc).isoformat(),
        "gpu": gpu,
        "kernel": os.uname().release,
        "pending": len(pending),
    }
    with (args.output / "invocations.jsonl").open("a") as output:
        output.write(json.dumps(invocation) + "\n")
    server_config = args.output / "server-config.json"
    server_config.write_text(json.dumps(config, indent=2))
    command = [
        str(python),
        "-m",
        "serve.server",
        "--engine",
        "strata",
        "--config",
        str(server_config.resolve()),
        "--host",
        "127.0.0.1",
        "--port",
        str(config["port"]),
    ]
    base_url = f"http://127.0.0.1:{config['port']}"
    with (args.output / "server.log").open("a") as log:
        server = subprocess.Popen(command, cwd=source, stdout=log, stderr=log, start_new_session=True)
        try:
            health = wait_ready(server, base_url + "/health")
            (args.output / "health.json").write_text(json.dumps(health, indent=2))
            warmup = completion(
                {"id": -1, "gold": "A", "prompt": "Reply with exactly Answer: A."},
                base_url + "/v1/chat/completions",
                config["model_name"],
                args.timeout,
            )
            with (args.output / "warmups.jsonl").open("a") as output:
                output.write(json.dumps(warmup) + "\n")
            with ledger.open("a") as output:
                for case in pending:
                    try:
                        row = completion(case, base_url + "/v1/chat/completions", config["model_name"], args.timeout)
                    except (OSError, ValueError, KeyError, IndexError) as error:
                        output.write(json.dumps({"id": case["id"], "success": False, "error": str(error)}) + "\n")
                        output.flush()
                        raise
                    output.write(json.dumps(row, ensure_ascii=False) + "\n")
                    output.flush()
                    os.fsync(output.fileno())
                    rows[case["id"]] = row
                    print(
                        json.dumps(
                            {
                                "id": row["id"],
                                "completed": len(rows),
                                "answer": row["answer"],
                                "correct": row["answer"] == row["gold"],
                                "seconds": row["seconds"],
                            }
                        ),
                        flush=True,
                    )
        finally:
            stop_server(server)
    return summarize(cases, rows, baseline)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("run", "summarize"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=QUESTION_COUNT)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--summary-name", default="summary.json")
    args = parser.parse_args()
    if not 1 <= args.limit <= QUESTION_COUNT or args.timeout < 1:
        parser.error("--limit must be 1..198 and --timeout must be positive")
    cases = read_cases(args.cases)[: args.limit]
    baseline = read_results(args.baseline)
    check_rows(read_cases(args.cases), baseline)
    if len(baseline) != QUESTION_COUNT:
        parser.error("Baseline must have 198 successful results")
    config = json.loads(args.config.read_text())
    if config.get("parallel", 1) != 1 or config.get("host", "127.0.0.1") != "127.0.0.1":
        parser.error("This serial experiment requires parallel=1 and loopback hosting")
    if args.mode == "run":
        summary = evaluate(cases, args, config, baseline)
    else:
        summary = summarize(cases, read_results(args.output / "results.jsonl"), baseline)
    summary["baseline_sha256"] = digest(args.baseline)
    (args.output / args.summary_name).write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
