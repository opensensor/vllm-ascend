"""Parser-aware, streaming GLM quality and serving workload runner.

Run ``python3 -m tools.glm_perf.suite --help`` for the workload selectors. A
JSONL row is written even for a failed request; the summary rejects failed or
incomplete runs. Token lengths are targets until server usage reports the exact
tokenization of the installed checkpoint and chat template.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import threading
import time
import urllib.request
from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import regex as re

DEFAULT_WORKLOADS = Path(__file__).with_name("workloads.json")
RAW_THINKING = re.compile(r"<\s*/?\s*think\b", re.IGNORECASE)
RETRIEVAL_CODE = "BLUE-ORCHID-7319"
RETRIEVAL_FILLER = "The quick brown fox jumps over the lazy dog. Keep this context in mind.\n"
CHAT_TEMPLATE_TOKEN_RESERVE = 64
WINDOW_COMPLETION_TOKENS = 256
SHORT_PROMPT = (
    "Generate a long numbered list of distinct everyday objects, starting "
    "at one. Continue writing until the response token limit stops you."
)
TOOL_DEFINITION = {
    "type": "function",
    "function": {
        "name": "get_order_status",
        "description": "Look up the status of an order by its ID.",
        "parameters": {
            "type": "object",
            "properties": {"order_id": {"type": "string"}},
            "required": ["order_id"],
        },
    },
}


@dataclass(frozen=True)
class Case:
    case_id: str
    category: str
    prompt: str
    expected: str | None
    max_tokens: int
    target_prompt_tokens: int | None = None
    tool: dict[str, Any] | None = None
    raw_prompt_tokens: int | None = None
    context_tier_tokens: int | None = None


def load_quality_cases(path: Path = DEFAULT_WORKLOADS) -> tuple[list[Case], dict[str, Any]]:
    data = json.loads(path.read_text())
    if data.get("version") != 1 or not isinstance(data.get("quality_cases"), list):
        raise ValueError("unsupported workload file")
    cases = [
        Case(item["id"], item["category"], item["prompt"], item["expected"], 256) for item in data["quality_cases"]
    ]
    if len(cases) < 20 or len({case.case_id for case in cases}) != len(cases):
        raise ValueError("quality suite requires at least 20 unique cases")
    if {case.category for case in cases} != {"arithmetic", "instruction", "code", "retrieval"}:
        raise ValueError("quality categories are incomplete")
    return cases, data["tool_case"]


def retrieval_case(
    target_tokens: int,
    *,
    category: str = "retrieval",
    variant: int = 0,
    count_tokens: Callable[[str], int] | None = None,
) -> Case:
    if target_tokens < 1024:
        raise ValueError("retrieval target must be at least 1024 tokens")
    code = RETRIEVAL_CODE if variant == 0 else f"{RETRIEVAL_CODE}-{variant}"

    def build(repeats: int) -> str:
        return (
            f"The secret code is {code}.\n"
            + RETRIEVAL_FILLER * repeats
            + "What is the secret code? Answer only the code."
        )

    if count_tokens is None:
        repeats = max(1, (target_tokens - 40) // 18)
        raw_tokens = None
    else:
        # Bracket then binary-search the number of complete filler records.
        # The closest of the two bracketing candidates is deterministic.
        low, high = 0, 1
        count_low = count_tokens(build(low))
        if count_low > target_tokens:
            raise ValueError("retrieval prefix exceeds token target")
        while count_tokens(build(high)) < target_tokens:
            low, high = high, high * 2
            if high > target_tokens * 2:
                raise ValueError("could not bracket retrieval token target")
        for _ in range(32):
            if high - low <= 1:
                break
            middle = (low + high) // 2
            if count_tokens(build(middle)) < target_tokens:
                low = middle
            else:
                high = middle
        candidates = ((abs(count_tokens(build(n)) - target_tokens), n) for n in (low, high))
        repeats = min(candidates)[1]
        raw_tokens = count_tokens(build(repeats))
    prompt = build(repeats)
    return Case(
        f"{category}_{target_tokens}_{variant}",
        category,
        prompt,
        code,
        WINDOW_COMPLETION_TOKENS,
        target_tokens,
        raw_prompt_tokens=raw_tokens,
    )


def load_tokenizer_json(path: Path) -> tuple[Callable[[str], int], dict[str, str]]:
    """Load a local fast tokenizer without importing model or NPU packages."""
    from tokenizers import Tokenizer  # Optional CLI dependency.

    tokenizer = Tokenizer.from_file(str(path))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return (
        lambda prompt: len(tokenizer.encode(prompt, add_special_tokens=False).ids),
        {"path": str(path.resolve()), "sha256": digest},
    )


def percentile(values: list[float], percentage: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentage / 100
    lower = math.floor(position)
    upper = math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def parse_sse(lines: Iterable[bytes], clock: Callable[[], float]) -> dict[str, Any]:
    """Parse OpenAI chat SSE, rejecting missing terminal/usage records."""
    content: list[str] = []
    reasoning: list[str] = []
    tools: dict[int, dict[str, str]] = {}
    timestamps: list[float] = []
    usage: dict[str, Any] | None = None
    finish_reason: str | None = None
    done = False
    buffer: list[str] = []

    def consume(raw: str) -> None:
        nonlocal done, usage, finish_reason
        if raw == "[DONE]":
            done = True
            return
        event = json.loads(raw)
        if not isinstance(event, dict) or "error" in event:
            raise ValueError(f"stream error: {event!r}")
        if event.get("usage") is not None:
            usage = event["usage"]
        choices = event.get("choices") or []
        if not isinstance(choices, list):
            raise ValueError("malformed choices")
        for choice in choices:
            if choice.get("finish_reason") is not None:
                finish_reason = choice["finish_reason"]
            delta = choice.get("delta") or {}
            final_piece = delta.get("content") or ""
            reasoning_piece = delta.get("reasoning_content") or delta.get("reasoning") or ""
            tool_chunks = delta.get("tool_calls") or []
            if final_piece or reasoning_piece or tool_chunks:
                timestamps.append(clock())
            content.append(final_piece)
            reasoning.append(reasoning_piece)
            for part in tool_chunks:
                index = part.get("index")
                if not isinstance(index, int):
                    raise ValueError("tool-call chunk missing index")
                call = tools.setdefault(index, {"id": "", "name": "", "arguments": ""})
                call["id"] += part.get("id") or ""
                function = part.get("function") or {}
                call["name"] += function.get("name") or ""
                call["arguments"] += function.get("arguments") or ""

    for chunk in lines:
        for decoded in chunk.decode("utf-8").splitlines():
            if decoded.startswith("data:"):
                buffer.append(decoded[5:].lstrip())
            elif not decoded and buffer:
                consume("\n".join(buffer))
                buffer.clear()
                if done:
                    break
        if done:
            break
    if buffer:
        consume("\n".join(buffer))
    if not done or finish_reason is None or not isinstance(usage, dict):
        raise ValueError("incomplete stream: terminal marker, finish reason, or usage missing")
    for key in ("prompt_tokens", "completion_tokens"):
        if not isinstance(usage.get(key), int) or usage[key] < 0:
            raise ValueError(f"missing or invalid {key}")
    if usage["completion_tokens"] > 0 and not timestamps:
        raise ValueError("completion tokens reported without streamed output")
    return {
        "content": "".join(content),
        "reasoning": "".join(reasoning),
        "tool_calls": [tools[index] for index in sorted(tools)],
        "finish_reason": finish_reason,
        "usage": usage,
        "token_timestamps_s": timestamps,
    }


def score(case: Case, response: dict[str, Any]) -> dict[str, Any]:
    content = response["content"]
    markers = bool(RAW_THINKING.search(content))
    completed = response["finish_reason"] in ("stop", "tool_calls")
    if case.category == "fault":
        return {
            "passed": response["usage"]["completion_tokens"] > 0,
            "tool_call_success": None,
            "raw_thinking_marker": markers,
            "completed_final": bool(content.strip()) and completed,
        }
    if case.category == "short":
        full_budget = response["usage"]["completion_tokens"] >= case.max_tokens
        return {
            "passed": full_budget and not markers,
            "tool_call_success": None,
            "raw_thinking_marker": markers,
            "completed_final": False,
            "speed_sample_complete": full_budget,
        }
    if case.tool:
        calls = response["tool_calls"]
        success = False
        if response["finish_reason"] == "tool_calls" and len(calls) == 1:
            try:
                args = json.loads(calls[0]["arguments"])
            except (ValueError, TypeError):
                args = None
            success = calls[0]["name"] == case.tool["function"] and args == case.tool["arguments"]
        return {
            "passed": completed and success and not markers,
            "tool_call_success": success,
            "raw_thinking_marker": markers,
            "completed_final": success,
        }
    final = bool(content.strip()) and response["finish_reason"] == "stop"
    exact = case.expected is None or content.strip() == case.expected
    return {
        "passed": final and exact and not markers,
        "tool_call_success": None,
        "raw_thinking_marker": markers,
        "completed_final": final,
        "exact_answer": exact,
    }


def request_body(case: Case, model: str, seed: int) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": case.prompt}],
        "max_tokens": case.max_tokens,
        "temperature": 0,
        "seed": seed,
        "chat_template_kwargs": {"reasoning_effort": "low"},
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if case.tool:
        body["tools"] = [TOOL_DEFINITION]
        body["tool_choice"] = "auto"
    return body


def run_request(
    case: Case,
    base_url: str,
    model: str,
    seed: int,
    group_id: str,
    *,
    opener: Callable[..., Any] = urllib.request.urlopen,
    clock: Callable[[], float] = time.perf_counter,
    tokenizer_identity: dict[str, str] | None = None,
    timeout_s: float = 900,
) -> dict[str, Any]:
    body = request_body(case, model, seed)
    settings = {key: value for key, value in body.items() if key != "messages"}
    row: dict[str, Any] = {
        "case_id": case.case_id,
        "category": case.category,
        "group_id": group_id,
        "target_prompt_tokens": case.target_prompt_tokens,
        "context_tier_tokens": case.context_tier_tokens,
        "raw_prompt_tokens": case.raw_prompt_tokens,
        "tokenizer": tokenizer_identity,
        "expected": case.expected,
        "request_settings": settings,
        "prompt": case.prompt,
        "model": model,
        "base_url": base_url,
        "valid": False,
    }
    request = urllib.request.Request(
        base_url.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    start = clock()
    row["start_s"] = start
    try:
        with opener(request, timeout=timeout_s) as stream:
            response = parse_sse(stream, clock)
        end = clock()
        times = response.pop("token_timestamps_s")
        first = times[0] if times else None
        last = times[-1] if times else None
        token_count = response["usage"]["completion_tokens"]
        if first is None or last is None or token_count == 0:
            raise ValueError("no generated tokens")
        gaps = [right - left for left, right in zip(times, times[1:])]
        decode_s = last - first
        row.update(response)
        row.update(
            {
                "start_s": start,
                "end_s": end,
                "first_token_s": first,
                "last_token_s": last,
                "ttft_s": first - start,
                "elapsed_s": end - start,
                "decode_s": decode_s,
                "decode_tokens_per_s": token_count / decode_s if decode_s > 0 and token_count > 1 else None,
                "stream_event_gap_p50_s": percentile(gaps, 50),
                "stream_event_gap_p95_s": percentile(gaps, 95),
                "stream_events": len(times),
                "prompt_token_target_error": (
                    response["usage"]["prompt_tokens"] - case.target_prompt_tokens
                    if case.target_prompt_tokens is not None
                    else None
                ),
                "early_eos": response["finish_reason"] == "stop" and token_count < case.max_tokens,
            }
        )
        row.update(score(case, response))
        if case.target_prompt_tokens is not None and case.target_prompt_tokens >= 1024:
            tolerance = max(32, math.ceil(case.target_prompt_tokens * 0.05))
            row["prompt_target_met"] = abs(row["prompt_token_target_error"]) <= tolerance
            row["prompt_target_tolerance"] = tolerance
            if not row["prompt_target_met"]:
                row["passed"] = False
        else:
            row["prompt_target_met"] = None
        if case.context_tier_tokens is not None:
            row["context_fit"] = response["usage"]["prompt_tokens"] + case.max_tokens <= case.context_tier_tokens
            if not row["context_fit"]:
                row["passed"] = False
        else:
            row["context_fit"] = None
        row["valid"] = True
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"
        row["end_s"] = clock()
    return row


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("empty workload")
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(row["group_id"], []).append(row)
    result: dict[str, Any] = {
        "request_count": len(rows),
        "valid": all(row.get("valid") for row in rows),
        "passed": all(row.get("valid") and row.get("passed") for row in rows),
        "early_eos_count": sum(bool(row.get("early_eos")) for row in rows),
        "groups": {},
    }
    for group_id, items in groups.items():
        valid = [row for row in items if row.get("valid")]
        rates = [row["decode_tokens_per_s"] for row in valid if row["decode_tokens_per_s"] is not None]
        ttfts = [row["ttft_s"] for row in valid]
        if valid:
            first = min(row["first_token_s"] for row in valid)
            last = max(row["last_token_s"] for row in valid)
            span = last - first
            decoded = sum(row["usage"]["completion_tokens"] for row in valid)
            aggregate = decoded / span if span > 0 else None
        else:
            aggregate = None
        result["groups"][group_id] = {
            "requests": len(items),
            "valid_requests": len(valid),
            "aggregate_decode_tokens_per_s": aggregate,
            "per_request_decode_p50": percentile(rates, 50),
            "per_request_decode_p95": percentile(rates, 95),
            "ttft_p50_s": percentile(ttfts, 50),
            "ttft_p95_s": percentile(ttfts, 95),
            "fairness_min_max_ratio": min(rates) / max(rates) if rates and max(rates) else None,
            "early_eos_count": sum(bool(row.get("early_eos")) for row in items),
        }
    return result


def make_groups(
    workloads: list[str],
    tiers: list[int],
    count_tokens: Callable[[str], int] | None = None,
    window_count: int = 4,
) -> list[tuple[str, list[Case]]]:
    if not 1 <= window_count <= 4:
        raise ValueError("window count must be between 1 and 4")
    quality, tool = load_quality_cases()
    groups: list[tuple[str, list[Case]]] = []
    for workload in workloads:
        if workload == "quality":
            groups.extend((case.case_id, [case]) for case in quality)
        elif workload == "short":
            for streams in (1, 4):
                groups.append(
                    (
                        f"short_{streams}",
                        [Case(f"short_{streams}_{i}", "short", SHORT_PROMPT, None, 256, 25) for i in range(streams)],
                    )
                )
        elif workload == "retrieval":
            groups.extend(
                (f"retrieval_{size}", [retrieval_case(size, count_tokens=count_tokens)]) for size in (8192, 32768)
            )
        elif workload == "windows":
            for size in tiers:
                prompt_target = size - WINDOW_COMPLETION_TOKENS - CHAT_TEMPLATE_TOKEN_RESERVE
                if prompt_target < 1024:
                    raise ValueError("window tier is too small for the 256-token completion and chat template reserve")
                groups.append(
                    (
                        f"windows_{size}",
                        [
                            replace(
                                retrieval_case(prompt_target, category="window", variant=i, count_tokens=count_tokens),
                                context_tier_tokens=size,
                            )
                            for i in range(window_count)
                        ],
                    )
                )
        elif workload == "fault":
            groups.append(("fault_32", [Case("fault_32", "fault", SHORT_PROMPT, None, 32, 25)]))
        elif workload == "fault4":
            groups.append(
                (
                    "fault4_32",
                    [Case(f"fault4_32_{index}", "fault", SHORT_PROMPT, None, 32, 25) for index in range(4)],
                )
            )
        elif workload == "tool":
            groups.append(("tool", [Case(tool["id"], "tool", tool["prompt"], None, 256, tool=tool)]))
    return groups


def run_groups(
    groups: list[tuple[str, list[Case]]],
    base_url: str,
    model: str,
    seed: int,
    tokenizer_identity: dict[str, str] | None = None,
    on_result: Callable[[dict[str, Any]], None] | None = None,
    timeout_s: float = 900,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def record(row: dict[str, Any]) -> None:
        rows.append(row)
        if on_result is not None:
            on_result(row)

    for group_id, cases in groups:
        if len(cases) == 1:
            record(
                run_request(
                    cases[0],
                    base_url,
                    model,
                    seed,
                    group_id,
                    tokenizer_identity=tokenizer_identity,
                    timeout_s=timeout_s,
                )
            )
            continue
        start = threading.Event()

        def submit(case: Case, gate: threading.Event = start, group: str = group_id) -> dict[str, Any]:
            gate.wait()
            return run_request(
                case, base_url, model, seed, group, tokenizer_identity=tokenizer_identity, timeout_s=timeout_s
            )

        with concurrent.futures.ThreadPoolExecutor(max_workers=len(cases)) as pool:
            futures = [pool.submit(submit, case) for case in cases]
            start.set()
            for future in futures:
                record(future.result())
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8001")
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--workload",
        choices=("quality", "short", "retrieval", "windows", "fault", "fault4", "tool"),
        action="append",
        required=True,
        help="repeat to combine workload types",
    )
    parser.add_argument(
        "--tier-tokens",
        type=int,
        action="append",
        default=None,
        help="four-window target; default 32768, 131072, 262144",
    )
    parser.add_argument(
        "--window-count",
        type=int,
        default=4,
        help="concurrent requests per window tier, 1–4 (default 4)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--request-timeout-s", type=float, default=900, help="per-request timeout in seconds")
    parser.add_argument(
        "--tokenizer-json",
        type=Path,
        help="local tokenizer.json for raw-prompt calibration; served usage remains authoritative",
    )
    parser.add_argument("--output", type=Path, required=True, help="new JSONL path; summary uses .summary.json")
    args = parser.parse_args()
    if args.output.exists() or args.output.with_suffix(".summary.json").exists():
        parser.error("output or summary already exists")
    tiers = args.tier_tokens or [32768, 131072, 262144]
    if any(tier < 1024 for tier in tiers):
        parser.error("tier-tokens must be at least 1024")
    if not math.isfinite(args.request_timeout_s) or args.request_timeout_s <= 0:
        parser.error("request-timeout-s must be a positive finite number")
    try:
        count_tokens, identity = load_tokenizer_json(args.tokenizer_json) if args.tokenizer_json else (None, None)
        groups = make_groups(args.workload, tiers, count_tokens, args.window_count)
    except (OSError, ValueError, ImportError) as exc:
        parser.error(f"tokenizer calibration failed: {exc}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as output:

        def write_result(row: dict[str, Any]) -> None:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
            output.flush()

        rows = run_groups(
            groups,
            args.base_url,
            args.model,
            args.seed,
            identity,
            on_result=write_result,
            timeout_s=args.request_timeout_s,
        )
    summary = summarize(rows)
    args.output.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    if not summary["valid"] or not summary["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
