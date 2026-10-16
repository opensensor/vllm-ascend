# SPDX-License-Identifier: Apache-2.0
"""Complete scored generation, context and lifecycle acceptance requests.

Use the authenticated T1/T2 fixture manifest, augmented with acceptance_cases.
Each case has name, tier, category, request, canonical request_sha256, optional
expected_prompt_tokens (mandatory for context), and score. Scores support exact,
contains, json, code (AST equality), tool and capacity. Synthetic quality cases
are loaded from the repository when acceptance_cases is absent. Context inputs
must be tokenized by the installed model and supplied explicitly: this client
does not estimate token counts or shorten prompts. All responses are consumed.

Stability additionally requires lifecycle_receipts: authenticated path/SHA256
references to local-owner checks. The owner performs restart/worker-loss actions;
this client never kills workers or launches processes. Receipts contain kind,
run_manifest_sha256, passed, completed, evidence and checks. Checks must include
the boolean keys in LIFECYCLE_CHECKS. Recovery actions additionally provide an
authenticated before/after run_manifest reference and a completed post-action
request/response with score. Missing receipts remain pending and prevent pass.
Prefix/state-reset receipts include completed_requests with full case/response
proofs (two repeated requests for prefix, repeat/distinct/repeat for reset).
Cancellation provides a completed post_action proof after the owner observes
retirement. Every proof includes done=true; no lifecycle checkbox alone passes.
Private prompts, outputs and tool arguments are omitted from public reports.
"""

import argparse
import ast
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

from .performance_suite import authenticate_service, canonical_sha256, load_authenticated

CONTEXT_LIMIT = 131072
CLIENT_OUTPUT_RESERVE = 8192
CAPACITY_PROMPT_TOKENS = 131008
CAPACITY_OUTPUT_TOKENS = 64
MIN_QUALITY_CASES = 20
MIN_STABILITY_REQUESTS = 100
CATEGORIES = frozenset({"arithmetic", "instruction", "code", "retrieval", "tool"})
LIFECYCLE_CHECKS = {
    "restart": ("startup_gates_passed", "bridge_owners_normal", "old_slots_retired"),
    "worker-loss": ("failure_reported", "requests_unblocked", "old_slots_retired", "bridge_owners_normal"),
    "cancellation": ("disconnect_observed", "request_retired", "slots_retired", "next_request_completed"),
    "prefix": ("support_or_explicit_fallback", "repeated_completed_answers_equal", "cache_accounting_valid"),
    "state-reset": ("distinct_request_intervened", "repeated_completed_answers_equal"),
}
PUBLIC_CASES = Path(__file__).resolve().parents[2] / "tests/e2e/nightly/sidecard/fixtures/acceptance-quality.json"


def output_budget(body):
    budget = body.get("max_completion_tokens", body.get("max_tokens"))
    if type(budget) is not int or budget < 1:
        raise ValueError("explicit positive output budget required")
    return budget


def validate_case(case, model):
    body = case["request"]
    if canonical_sha256(body) != case["request_sha256"]:
        raise ValueError("acceptance request identity changed")
    if body.get("model", model) != model or body.get("n", 1) != 1:
        raise ValueError("one sequence of the manifest model required")
    if ("prompt" in body) == ("messages" in body):
        raise ValueError("exactly one prompt or messages required")
    if body.get("stream") and body.get("stream_options", {}).get("include_usage") is not True:
        raise ValueError("streamed acceptance fixture must request completion usage")
    budget = output_budget(body)
    if "max_tokens" in body and "max_completion_tokens" in body and body["max_tokens"] != budget:
        raise ValueError("conflicting output budgets")
    if "expected_tool_count" in case and len(body.get("tools", [])) != case["expected_tool_count"]:
        raise ValueError("tool schema count changed")
    if case["tier"] not in {"quality", "context"} or case["category"] not in CATEGORIES | {"capacity"}:
        raise ValueError("unknown acceptance tier/category")
    if case["tier"] == "context":
        count = case.get("expected_prompt_tokens")
        if type(count) is not int or count < 1 or count + budget > CONTEXT_LIMIT:
            raise ValueError("context case requires exact in-budget prompt geometry")
        if case["category"] == "capacity" and (
            count != CAPACITY_PROMPT_TOKENS or budget != CAPACITY_OUTPUT_TOKENS or body.get("ignore_eos") is not True
        ):
            raise ValueError("capacity requires actual 131008+64 with ignore_eos")
    elif budget != CLIENT_OUTPUT_RESERVE:
        raise ValueError("quality/client fixture must preserve 8192 output reservation")
    if case["score"]["kind"] not in {"exact", "contains", "json", "code", "tool", "capacity"}:
        raise ValueError("unsupported score kind")
    if (case["score"]["kind"] == "capacity") != (case["category"] == "capacity"):
        raise ValueError("capacity score cannot qualify answer quality")


def validate_manifest(manifest, root):
    if manifest.get("schema_version") != 1 or not manifest.get("model"):
        raise ValueError("schema_version=1/model required")
    if manifest["service"]["identity"].get("max_model_len") != CONTEXT_LIMIT:
        raise ValueError("131072 service identity required")
    authenticate_service(manifest["service"], root)
    # Original private Code/tool requests remain independently authenticated.
    for fixture in manifest.get("fixtures", []):
        if canonical_sha256(fixture["request"]) != fixture["request_sha256"]:
            raise ValueError("original fixture changed")
        if (
            "expected_tool_count" in fixture
            and len(fixture["request"].get("tools", [])) != fixture["expected_tool_count"]
        ):
            raise ValueError("original tools changed")
    cases = manifest.get("acceptance_cases")
    if cases is None:
        cases = json.loads(PUBLIC_CASES.read_bytes())["acceptance_cases"]
    if not cases or len({case["name"] for case in cases}) != len(cases):
        raise ValueError("unique nonempty acceptance case names required")
    for case in cases:
        validate_case(case, manifest["model"])
    return cases


def strip_code_fence(text):
    lines = text.strip().splitlines()
    if lines and lines[0].startswith("```") and lines[-1] == "```":
        return "\n".join(lines[1:-1])
    return text.strip()


def score_answer(message, score):
    """Score final content only; reasoning is never a substitute for an answer."""
    text = (message.get("content") or "").strip()
    kind = score["kind"]
    if kind == "capacity":
        return True  # Capacity checks usage and full stream completion separately.
    if kind == "tool":
        calls = message.get("tool_calls") or []
        if len(calls) != 1:
            return False
        function = calls[0].get("function", {})
        try:
            arguments = json.loads(function.get("arguments", ""))
        except (TypeError, ValueError):
            return False
        return function.get("name") == score["name"] and arguments == score["arguments"]
    if not text or "<think>" in text or "</think>" in text:
        return False
    if kind == "exact":
        return text == score["expected"]
    if kind == "contains":
        return all(value in text for value in score["required"]) and not any(
            value in text for value in score.get("forbidden", [])
        )
    if kind == "json":
        try:
            return json.loads(text) == score["expected"]
        except ValueError:
            return False
    if kind == "code":
        try:
            return ast.dump(ast.parse(strip_code_fence(text))) == ast.dump(ast.parse(score["expected"]))
        except (SyntaxError, ValueError):
            return False
    raise ValueError("unsupported score kind")


def completed_result(response, case, *, done=True):
    """Distinguish completed protocol, answer quality and full capacity."""
    flags = []
    choices = response.get("choices", [])
    usage = response.get("usage") or {}
    if not done or len(choices) != 1 or choices[0].get("index", 0) != 0:
        flags.append("incomplete_response")
    choice = choices[0] if choices else {}
    finish = choice.get("finish_reason")
    message = choice.get("message") or {"content": choice.get("text", "")}
    budget = output_budget(case["request"])
    prompt, count = usage.get("prompt_tokens"), usage.get("completion_tokens")
    if type(prompt) is not int or prompt < 1 or type(count) is not int or not 1 <= count <= budget:
        flags.append("invalid_usage")
    elif usage.get("total_tokens") != prompt + count:
        flags.append("invalid_total_usage")
    if case.get("expected_prompt_tokens") is not None and prompt != case["expected_prompt_tokens"]:
        flags.append("prompt_identity_mismatch")
    if type(prompt) is int and prompt + budget > CONTEXT_LIMIT:
        flags.append("reservation_over_limit")
    capacity = case["category"] == "capacity"
    expected_finish = "length" if capacity else "tool_calls" if case["score"]["kind"] == "tool" else "stop"
    if finish != expected_finish:
        flags.append("unfinished_answer" if not capacity else "incomplete_capacity")
    if capacity and (prompt != CAPACITY_PROMPT_TOKENS or count != CAPACITY_OUTPUT_TOKENS):
        flags.append("incomplete_capacity")
    quality = score_answer(message, case["score"]) if not capacity else None
    if quality is False:
        flags.append("incorrect_final_answer")
    protocol = not any(flag != "incorrect_final_answer" for flag in flags)
    return {
        "passed": not flags,
        "completed": protocol,
        "quality_passed": quality if protocol else False if not capacity else None,
        "capacity_passed": protocol if capacity else None,
        "flags": sorted(set(flags)),
        "finish_reason": finish,
        "prompt_tokens": prompt,
        "completion_tokens": count,
        "response_sha256": canonical_sha256(response),
        "final_content_sha256": canonical_sha256(message.get("content") or choice.get("text", "")),
        "reasoning_present": bool(message.get("reasoning_content") or message.get("reasoning")),
    }


def read_stream(lines):
    """Consume SSE to DONE and reconstruct separated reasoning/final/tool fields."""
    message = {"content": "", "reasoning_content": "", "tool_calls": []}
    calls, usage, finish, done = {}, None, None, False
    for raw in lines:
        if not raw.startswith(b"data:"):
            continue
        value = raw[5:].strip()
        if value == b"[DONE]":
            done = True
            continue
        if done:
            raise ValueError("event after DONE")
        event = json.loads(value)
        if event.get("error"):
            raise ValueError("stream reported request failure")
        if event.get("usage") is not None:
            usage = event["usage"]
        for choice in event.get("choices", []):
            if choice.get("index", 0) != 0:
                raise ValueError("unexpected output sequence")
            delta = choice.get("delta") or {}
            message["content"] += choice.get("text") or delta.get("content") or ""
            message["reasoning_content"] += delta.get("reasoning_content") or delta.get("reasoning") or ""
            for part in delta.get("tool_calls") or []:
                index = part.get("index", 0)
                if type(index) is not int or index < 0:
                    raise ValueError("invalid tool index")
                target = calls.setdefault(index, {"function": {"name": "", "arguments": ""}})
                for key in ("name", "arguments"):
                    target["function"][key] += part.get("function", {}).get(key) or ""
            if choice.get("finish_reason") is not None:
                finish = choice["finish_reason"]
    if set(calls) != set(range(len(calls))):
        raise ValueError("noncontiguous tool sequence")
    message["tool_calls"] = [calls[index] for index in sorted(calls)]
    return {"choices": [{"index": 0, "message": message, "finish_reason": finish}], "usage": usage}, done


def endpoint(base_url, body):
    base = base_url.rstrip("/")
    if base.endswith("/v1"):
        base = base[:-3]
    return base + ("/v1/chat/completions" if "messages" in body else "/v1/completions")


def send_case(base_url, case, model, *, timeout=7200):
    body = dict(case["request"])
    body.setdefault("model", model)
    query = urllib.request.Request(
        endpoint(base_url, body), data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
    )
    start = time.monotonic()
    with urllib.request.urlopen(query, timeout=timeout) as response:
        if response.status != 200:
            raise ValueError("request did not succeed")
        if body.get("stream", False):
            value, done = read_stream(response)
        else:
            value, done = json.load(response), True
    return {"elapsed_seconds": time.monotonic() - start, **completed_result(value, case, done=done)}


def lifecycle_results(manifest, root):
    """Authenticate explicit owner checks; absent external actions stay pending."""
    rows = []
    supplied = {}
    for reference in manifest.get("lifecycle_receipts", []):
        value, path = load_authenticated(reference, root)
        kind = value["kind"]
        if kind in supplied or kind not in LIFECYCLE_CHECKS:
            raise ValueError("duplicate or unknown lifecycle receipt")
        supplied[kind] = (value, path)
    for kind, checks in LIFECYCLE_CHECKS.items():
        if kind not in supplied:
            rows.append({"name": kind, "status": "pending", "passed": False, "reason": "local-owner receipt required"})
            continue
        value, path = supplied[kind]
        passed = (
            value.get("run_manifest_sha256") == manifest["service"]["run_manifest"]["sha256"]
            and value.get("passed") is True
            and value.get("completed") is True
            and bool(value.get("evidence"))
            and all(value.get("checks", {}).get(check) is True for check in checks)
        )
        if kind in {"restart", "worker-loss"}:
            before, _ = load_authenticated(value["before_run_manifest"], path.parent)
            after, _ = load_authenticated(value["after_run_manifest"], path.parent)
            first, last = before["performance_contract"], after["performance_contract"]
            passed &= (
                first["launch_id"] != last["launch_id"]
                and first["launch_started_unix_seconds"] < last["launch_started_unix_seconds"]
                and last.get("startup_gates_passed") is True
                and last["identity"] == manifest["service"]["identity"]
                and value["after_run_manifest"]["sha256"] == manifest["service"]["run_manifest"]["sha256"]
            )
        if kind in {"restart", "worker-loss", "cancellation"}:
            proof = value["post_action"]
            validate_case(proof["case"], manifest["model"])
            passed &= completed_result(proof["response"], proof["case"], done=proof.get("done") is True)["passed"]
        else:
            proofs = value.get("completed_requests", [])
            expected = 3 if kind == "state-reset" else 2
            passed &= len(proofs) == expected
            results = []
            for proof in proofs:
                validate_case(proof["case"], manifest["model"])
                result = completed_result(proof["response"], proof["case"], done=proof.get("done") is True)
                passed &= result["passed"]
                results.append(result)
            if len(proofs) == expected:
                passed &= (
                    proofs[0]["case"]["request_sha256"] == proofs[-1]["case"]["request_sha256"]
                    and results[0]["final_content_sha256"] == results[-1]["final_content_sha256"]
                )
                if kind == "state-reset":
                    passed &= proofs[1]["case"]["request_sha256"] != proofs[0]["case"]["request_sha256"]
                else:
                    for proof in proofs:
                        usage = proof["response"]["usage"]
                        cached = usage.get("prompt_tokens_details", {}).get("cached_tokens")
                        passed &= type(cached) is int and 0 <= cached <= usage["prompt_tokens"]
        rows.append({"name": kind, "status": "completed" if passed else "failed", "passed": bool(passed)})
    return rows


def over_limit_result(base_url, model):
    # An output reservation alone exceeding context must be rejected before work.
    body = {"model": model, "prompt": "hi", "max_tokens": CONTEXT_LIMIT + 1}
    query = urllib.request.Request(
        endpoint(base_url, body), data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(query, timeout=30) as response:
            response.read()
            status = response.status
    except urllib.error.HTTPError as error:
        status = error.code
        # Do not publish a server error body that might contain prompt content.
        error.read()
    return {"name": "over-limit", "status": status, "passed": status in {400, 422}}


def run(base_url, fixtures, output, tier, *, sender=send_case):
    fixtures, output = Path(fixtures), Path(output)
    if output.exists():
        raise FileExistsError("acceptance receipt already exists")
    manifest = json.loads(fixtures.read_bytes())
    cases = validate_manifest(manifest, fixtures.parent)
    selected = [case for case in cases if case["tier"] == ("quality" if tier == "stability" else tier)]
    if tier in {"quality", "stability"}:
        if len(selected) < MIN_QUALITY_CASES or {case["category"] for case in selected} != CATEGORIES:
            raise ValueError("at least 20 cases covering arithmetic/instruction/code/retrieval/tool required")
    if tier == "context":
        retrieval = {case.get("context_tier") for case in selected if case["category"] == "retrieval"}
        if not {32768, CONTEXT_LIMIT} <= retrieval or not any(case["category"] == "capacity" for case in selected):
            raise ValueError("32K/128K scored retrieval and 131008+64 full capacity fixtures required")
        for case in selected:
            if (
                case["category"] == "retrieval"
                and case["expected_prompt_tokens"] + output_budget(case["request"]) != case["context_tier"]
            ):
                raise ValueError("retrieval tier must exactly exercise its total reservation")
    report = {
        "schema_version": 1,
        "tier": tier,
        "passed": False,
        "hardware_validated": False,
        "service": manifest["service"],
        "fixtures_sha256": canonical_sha256(manifest),
        "client_output_reserve": CLIENT_OUTPUT_RESERVE,
        "cases": [],
        "lifecycle": [],
        "pending": [],
    }
    try:
        count = MIN_STABILITY_REQUESTS if tier == "stability" else len(selected)
        for index in range(count):
            case = selected[index % len(selected)]
            try:
                result = sender(base_url, case, manifest["model"])
            except (OSError, ValueError, KeyError, TypeError) as error:
                result = {"passed": False, "completed": False, "error_type": type(error).__name__}
            report["cases"].append(
                {
                    "name": case["name"],
                    "iteration": index,
                    "category": case["category"],
                    "request_sha256": case["request_sha256"],
                    **result,
                }
            )
        if tier == "stability":
            report["lifecycle"] = lifecycle_results(manifest, fixtures.parent)
            report["lifecycle"].append(over_limit_result(base_url, manifest["model"]))
            report["pending"] = [row["name"] for row in report["lifecycle"] if row.get("status") == "pending"]
        report["generation_passed"] = all(row["passed"] for row in report["cases"])
        report["passed"] = report["generation_passed"] and all(row["passed"] for row in report["lifecycle"])
        report["hardware_validated"] = report["passed"] and sender is send_case
        report["completed_requests"] = sum(row.get("completed") is True for row in report["cases"])
    finally:
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x") as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write("\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--fixtures", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--tier", choices=("quality", "context", "stability"), required=True)
    args = parser.parse_args()
    report = run(args.base_url, args.fixtures, args.output, args.tier)
    print(json.dumps({"passed": report["passed"], "tier": report["tier"], "pending": report["pending"]}))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
