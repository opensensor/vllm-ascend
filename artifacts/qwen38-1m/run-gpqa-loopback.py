#!/usr/bin/env python3
"""Run the pinned AISBench GPQA Diamond replay over server loopback.

The AISBench client on a different subnet can lose large HTTP responses. This
runner uses the same deterministic prompts, API parameters, and GPQA answer
extraction while keeping the HTTP connection on the server host.
"""

import argparse
import csv
import json
import time
import urllib.error
import urllib.request
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from threading import Lock

import regex as re

ALIGN_PROMPT = (
    "Answer the following multiple choice question. The last line of your "
    "response should be of the following format: 'Answer: $LETTER' (without "
    "quotes) where LETTER is one of ABCD. Think step by step before answering."
    "\n\n{question}\n\nA) {A}\nB) {B}\nC) {C}\nD) {D}"
)
SHUFFLE_PATTERNS = ("ABCD", "BCDA", "CDAB", "DABC")
ANSWER_PATTERN = re.compile(r"ANSWER\s*:\s*([A-D])", re.IGNORECASE)
GENERATION_KWARGS = {
    "ignore_eos": False,
    "min_p": 0.0,
    "presence_penalty": 0.0,
    "repetition_penalty": 1.0,
    "seed": 1024,
    "temperature": 0.0,
    "top_k": 20,
    "top_p": 1.0,
    "max_tokens": 8192,
}


def prepare_cases(dataset_path: Path) -> list[dict]:
    """Match the pinned AISBench GPQADataset and zero-shot CoT template."""
    cases = []
    with dataset_path.open(newline="", encoding="utf-8") as source:
        for row in csv.reader(source):
            if row[7] == "Question":
                continue
            number = len(cases) + 1
            options = row[8:12]
            pattern = SHUFFLE_PATTERNS[number % len(SHUFFLE_PATTERNS)]
            ordered = [options[ord(letter) - ord("A")] for letter in pattern]
            values = dict(zip("ABCD", ordered))
            cases.append(
                {
                    "id": number - 1,
                    "prompt": ALIGN_PROMPT.format(question=row[7], **values),
                    "gold": "ABCD"[ordered.index(options[0])],
                }
            )
    return cases


def extract_answer(prediction: str) -> str | None:
    matches = ANSWER_PATTERN.findall(prediction)
    return matches[-1].upper() if matches else None


def infer(case: dict, url: str, model: str, timeout: int, active: dict, lock: Lock) -> dict:
    request_body = {
        "stream": False,
        "messages": [{"role": "user", "content": case["prompt"]}],
        **GENERATION_KWARGS,
        "model": model,
    }
    request = urllib.request.Request(
        url,
        data=json.dumps(request_body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.monotonic()
    with lock:
        active[case["id"]] = started
    try:
        for attempt in range(1, 3):
            try:
                with urllib.request.urlopen(request, timeout=timeout) as response:
                    payload = json.load(response)
                choice = payload["choices"][0]
                message = choice["message"]
                reasoning = message.get("reasoning_content") or message.get("reasoning") or ""
                content = message.get("content") or ""
                prediction = reasoning + "</think>" + content if reasoning else content
                return {
                    "id": case["id"],
                    "gold": case["gold"],
                    "success": True,
                    "prediction": prediction,
                    "answer": extract_answer(prediction),
                    "output_tokens": payload.get("usage", {}).get("completion_tokens"),
                    "finish_reason": choice.get("finish_reason"),
                    "seconds": round(time.monotonic() - started, 2),
                    "attempts": attempt,
                }
            except (OSError, ValueError, KeyError, IndexError) as error:
                failure = f"{type(error).__name__}: {error}"
        return {
            "id": case["id"],
            "gold": case["gold"],
            "success": False,
            "error": failure,
            "seconds": round(time.monotonic() - started, 2),
            "attempts": 2,
        }
    finally:
        with lock:
            active.pop(case["id"], None)


def load_successes(result_path: Path) -> dict[int, dict]:
    successes = {}
    if result_path.exists():
        for line in result_path.read_text(encoding="utf-8").splitlines():
            try:
                result = json.loads(line)
            except json.JSONDecodeError:
                continue
            if result.get("success"):
                successes[result["id"]] = result
    return successes


def report(cases: list[dict], successes: dict[int, dict]) -> None:
    correct = sum(successes[case["id"]].get("answer") == case["gold"] for case in cases if case["id"] in successes)
    score = f"; accuracy {100 * correct / len(cases):.2f}%" if len(successes) == len(cases) else ""
    print(f"Completed {len(successes)}/{len(cases)}; correct {correct}{score}", flush=True)


def run(cases: list[dict], result_path: Path, url: str, model: str, workers: int, timeout: int) -> None:
    result_path.parent.mkdir(parents=True, exist_ok=True)
    successes = load_successes(result_path)
    pending_cases = [case for case in cases if case["id"] not in successes]
    active = {}
    lock = Lock()
    report(cases, successes)
    print(f"Submitting {len(pending_cases)} cases to {url} with {workers} workers", flush=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(infer, case, url, model, timeout, active, lock): case["id"] for case in pending_cases}
        with result_path.open("a", encoding="utf-8") as output:
            while futures:
                done, _ = wait(futures, timeout=30, return_when=FIRST_COMPLETED)
                if not done:
                    with lock:
                        running = [(case_id, round(time.monotonic() - start)) for case_id, start in active.items()]
                    print(
                        f"Heartbeat: {len(successes)}/{len(cases)} saved; active case IDs and ages (s): {running}",
                        flush=True,
                    )
                    continue
                for future in done:
                    case_id = futures.pop(future)
                    result = future.result()
                    output.write(json.dumps(result, ensure_ascii=False) + "\n")
                    output.flush()
                    if result["success"]:
                        successes[case_id] = result
                    print(
                        f"Case {case_id}: success={result['success']} "
                        f"tokens={result.get('output_tokens')} "
                        f"seconds={result['seconds']}",
                        flush=True,
                    )
                report(cases, successes)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", type=Path, help="AISBench gpqa_diamond.csv input")
    parser.add_argument("--cases", type=Path, required=True, help="Prepared cases JSON")
    parser.add_argument("--results", type=Path, help="Append-only result JSONL")
    parser.add_argument("--url", default="http://127.0.0.1:8001/v1/chat/completions")
    parser.add_argument("--model", default="qwen38-w4-pipeline-c1-dispatch")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=1200)
    args = parser.parse_args()
    if args.prepare:
        cases = prepare_cases(args.prepare)
        args.cases.write_text(json.dumps(cases, ensure_ascii=False), encoding="utf-8")
        print(f"Prepared {len(cases)} cases in {args.cases}")
    elif args.results:
        cases = json.loads(args.cases.read_text(encoding="utf-8"))
        if args.workers < 1 or args.timeout < 1:
            parser.error("--workers and --timeout must be positive")
        run(cases, args.results, args.url, args.model, args.workers, args.timeout)
    else:
        parser.error("provide --prepare or --results")


if __name__ == "__main__":
    main()
