# SPDX-License-Identifier: Apache-2.0
"""Collect private-fixture API receipts and compare authenticated launch pairs.

Fixture schema version 1 contains ``model``, ``service`` and ``fixtures``.
Each fixture has ``name``, ``request``, its canonical ``request_sha256``,
``expected_prompt_tokens`` and optional ``expected_tool_count``. Service has
``identity`` (source/config/checkpoint/native SHA256, parser, buffering_policy, architecture,
max_model_len), ``run_manifest`` (path and raw-file SHA256), ``startup_gates``
(passed and evidence), ``warmup`` (completed, policy and evidence), and
``cache_policy`` (mode disabled or fresh-process-per-request, and evidence).
Paths are relative to the containing JSON. Request text never enters receipts.

An index for --compare-pairs has schema_version=1 and pairs containing pair_id,
order (baseline,candidate or its reverse), and baseline/candidate path+sha256
references; required_fixtures names the full intended comparison coverage.
The authenticated run manifest contains performance_contract with identity,
prefix_cache_enabled, startup_gates_passed, warmup_completed, warmup_policy,
launch_id, and launch_started_unix_seconds. Collection runs never restart
services or flush caches. Fresh-process
cold measurements require one fixture/request per invocation. A local operator
must alternate actual launches and assemble at least five pairs per fixture.

API TTFT measures the first nonempty output emission, not an unobservable device
token timestamp. Decode uses completion-token usage (reasoning included) and the
first/last output emissions; parser buffering/coalescing is explicitly flagged.
Model-forward timing is unavailable here and must come from a separate receipt.
Unequal token/emission counts are informational: special tokens, incomplete
Unicode and parser buffering can withhold text. Sampling confidence intervals
do not estimate this unobservable token-to-emission timing bias.
"""

import argparse
import hashlib
import json
import math
import random
import statistics
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from .service_validate import request

MIN_PAIRS = 5
BOOTSTRAP_SAMPLES = 2000
IDENTITY_HASH_FIELDS = ("source_sha256", "config_sha256", "checkpoint_sha256", "native_sha256")
INFORMATIONAL_FLAGS = frozenset({"coalesced_emission", "fragmented_emission"})
METRIC_BOUNDARY = "usage_accounted_output_emission_proxy"


def uncertainty_metadata():
    return {
        "accounting": "completion_tokens from usage, including reasoning and withheld special tokens",
        "timing": "first/last nonempty output emissions; internal token timestamps are unobserved",
        "buffering_bias": "unknown; detokenizer or parser can withhold/coalesce output",
        "confidence_interval_scope": "paired-launch sampling variation; excludes unobserved buffering bias",
    }


def canonical_sha256(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def load_authenticated(reference, root):
    path = Path(reference["path"])
    if not path.is_absolute():
        path = root / path
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != reference["sha256"]:
        raise ValueError(f"artifact authentication failed: {path}")
    return json.loads(raw), path


def positive_integer(value):
    return type(value) is int and value > 0


def authenticate_service(service, root):
    run, _ = load_authenticated(service["run_manifest"], root)
    contract = run["performance_contract"]
    if contract["identity"] != service["identity"]:
        raise ValueError("run manifest service identity differs from fixture manifest")
    if not contract.get("launch_id") or not isinstance(contract.get("launch_started_unix_seconds"), (int, float)):
        raise ValueError("authenticated launch identity/time required")
    if not math.isfinite(contract["launch_started_unix_seconds"]) or contract["launch_started_unix_seconds"] <= 0:
        raise ValueError("invalid launch timestamp")
    if contract.get("startup_gates_passed") is not True or contract.get("warmup_completed") is not True:
        raise ValueError("run manifest does not authenticate startup/warmup completion")
    if contract.get("warmup_policy") != service["warmup"]["policy"]:
        raise ValueError("run manifest warmup policy differs")
    if service["cache_policy"]["mode"] == "disabled" and contract.get("prefix_cache_enabled") is not False:
        raise ValueError("disabled caching is not authenticated by run manifest")
    return contract


def validate_manifest(manifest, root):
    if manifest.get("schema_version") != 1 or not isinstance(manifest.get("model"), str) or not manifest["model"]:
        raise ValueError("fixture schema_version=1 and model are required")
    service = manifest["service"]
    identity = service["identity"]
    for name in IDENTITY_HASH_FIELDS:
        value = identity.get(name)
        if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError(f"service identity requires SHA256: {name}")
    if (
        not identity.get("parser")
        or not identity.get("buffering_policy")
        or not identity.get("architecture")
        or identity.get("max_model_len") != 131072
    ):
        raise ValueError("parser, buffering policy, architecture and qualified 131072 context identity required")
    load_authenticated(service["run_manifest"], root)
    for name, flag in (("startup_gates", "passed"), ("warmup", "completed")):
        if service[name].get(flag) is not True or not service[name].get("evidence"):
            raise ValueError(f"completed {name} with evidence required")
    if not service["warmup"].get("policy"):
        raise ValueError("explicit warmup policy required")
    policy = service["cache_policy"]
    if policy.get("mode") not in ("disabled", "fresh-process-per-request") or not policy.get("evidence"):
        raise ValueError("authenticated external cold-cache policy required")
    authenticate_service(service, root)
    fixtures = manifest["fixtures"]
    if not fixtures or len({fixture["name"] for fixture in fixtures}) != len(fixtures):
        raise ValueError("unique nonempty fixture names required")
    for fixture in fixtures:
        body = fixture["request"]
        if canonical_sha256(body) != fixture["request_sha256"]:
            raise ValueError("private fixture request identity mismatch")
        if ("prompt" in body) == ("messages" in body) or not positive_integer(fixture["expected_prompt_tokens"]):
            raise ValueError("one completion or chat request and exact prompt-token count required")
        if "expected_tool_count" in fixture and len(body.get("tools", [])) != fixture["expected_tool_count"]:
            raise ValueError("fixture tool schema count changed")
        if "model" in body and body["model"] != manifest["model"]:
            raise ValueError("fixture model differs from manifest")
        if body.get("n", 1) != 1:
            raise ValueError("one output sequence per request required")
    return manifest


def output_fragments(event):
    """Return output presence and visible text, without tokenizing SSE chunks."""
    generated, visible = False, False
    for choice in event.get("choices", []):
        if choice.get("index", 0) != 0:
            raise ValueError("unexpected output sequence")
        text, delta = choice.get("text", ""), choice.get("delta") or {}
        content = delta.get("content") or ""
        reasoning = delta.get("reasoning_content") or delta.get("reasoning") or ""
        generated |= bool(text or content or reasoning or delta.get("tool_calls"))
        visible |= bool(text or content)
    return generated, visible


def measure_events(events, *, output_tokens, expected_prompt_tokens, elapsed_seconds):
    """Interpret (elapsed_seconds, decoded SSE event) pairs; portable test seam."""
    first = visible = last = None
    emissions, usage, done, finish_reason = 0, None, False, None
    flags = []
    previous = 0.0
    for elapsed, event in events:
        if not math.isfinite(elapsed) or elapsed < previous or elapsed > elapsed_seconds:
            raise ValueError("nonmonotonic event timing")
        previous = elapsed
        if event == "[DONE]":
            done = True
            continue
        if done:
            flags.append("event_after_done")
        if event.get("error"):
            flags.append("failed_request")
        generated, text = output_fragments(event)
        if generated:
            first = elapsed if first is None else first
            last = elapsed
            emissions += 1
        if text and visible is None:
            visible = elapsed
        for choice in event.get("choices", []):
            if choice.get("finish_reason") is not None:
                finish_reason = choice["finish_reason"]
        if event.get("usage"):
            if usage is not None and usage != event["usage"]:
                flags.append("conflicting_usage")
            usage = event["usage"]
    if not done:
        flags.append("incomplete_stream")
    if finish_reason is None:
        flags.append("missing_finish_reason")
    if first is None:
        flags.append("no_output_emission")
    completed = None if usage is None else usage.get("completion_tokens")
    cached = None if usage is None else (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
    if usage is None:
        flags.append("missing_usage")
    elif not positive_integer(completed):
        flags.append("invalid_output_usage")
    else:
        if usage.get("prompt_tokens") != expected_prompt_tokens:
            flags.append("prompt_identity_mismatch")
        if completed < output_tokens:
            flags.append("early_eos")
        if completed > output_tokens:
            flags.append("output_budget_exceeded")
        if completed > emissions:
            flags.append("coalesced_emission")
        elif completed < emissions:
            flags.append("fragmented_emission")
    if cached is None:
        flags.append("missing_cache_usage")
    elif type(cached) is not int or cached < 0:
        flags.append("invalid_cache_usage")
    elif cached != 0:
        flags.append("prefix_reuse")
    span = None if first is None or last is None else last - first
    rate = (completed - 1) / span if positive_integer(completed) and completed > 1 and span and span > 0 else None
    if rate is None or not math.isfinite(rate):
        rate = None
        flags.append("unobservable_decode_span")
    failure_flags = sorted(set(flags) - INFORMATIONAL_FLAGS)
    return {
        "passed": not failure_flags,
        "flags": sorted(set(flags)),
        "failure_flags": failure_flags,
        "usage": usage,
        "completion_tokens": completed,
        "cached_tokens": cached,
        "finish_reason": finish_reason,
        "output_emission_events": emissions,
        "cold_ttft_seconds": first,
        "first_visible_text_seconds": visible,
        "api_decode_tokens_per_second": rate,
        "api_decode_seconds": span,
        "total_latency_seconds": elapsed_seconds,
        "ttft_boundary": "first-nonempty-api-output-emission",
        "decode_boundary": "usage-minus-one-over-first-to-last-output-emission",
        "metric_boundary": METRIC_BOUNDARY,
        "uncertainty": uncertainty_metadata(),
        "model_forward_timing": None,
    }


def stream_case(base_url, fixture, model, output_tokens, timeout):
    body = dict(fixture["request"])
    body.update(
        model=model, max_tokens=output_tokens, temperature=0, stream=True, stream_options={"include_usage": True}
    )
    # Preserve ignore_eos, tools, reasoning and template choices from the private fixture.
    endpoint = "/v1/chat/completions" if "messages" in body else "/v1/completions"
    query = urllib.request.Request(
        base_url + endpoint, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
    )
    events = []
    started = time.monotonic()
    with urllib.request.urlopen(query, timeout=timeout) as response:
        for raw in response:
            line = raw.strip()
            if not line.startswith(b"data:"):
                continue
            data = line[5:].strip()
            events.append((time.monotonic() - started, "[DONE]" if data == b"[DONE]" else json.loads(data)))
    return measure_events(
        events,
        output_tokens=output_tokens,
        expected_prompt_tokens=fixture["expected_prompt_tokens"],
        elapsed_seconds=time.monotonic() - started,
    ), canonical_sha256(body)


def normalized_base_url(value):
    value = value.rstrip("/")
    return value[:-3] if value.endswith("/v1") else value


def collect(base_url, fixtures_path, output, *, repeats, output_tokens, pair_id, mode, pair_order, timeout=1800):
    fixtures_path, output = Path(fixtures_path), Path(output)
    if output.exists():
        raise ValueError("refusing to overwrite an original receipt")
    manifest = validate_manifest(json.loads(fixtures_path.read_bytes()), fixtures_path.parent)
    if not positive_integer(repeats) or not positive_integer(output_tokens) or not pair_id:
        raise ValueError("positive repeat/output counts and pair ID required")
    if mode not in ("baseline", "candidate") or pair_order not in ("baseline-first", "candidate-first"):
        raise ValueError("explicit comparison mode and launch order required")
    if manifest["service"]["cache_policy"]["mode"] == "fresh-process-per-request" and (
        repeats != 1 or len(manifest["fixtures"]) != 1
    ):
        raise ValueError("fresh-process cold policy requires one fixture/request per invocation")
    base_url = normalized_base_url(base_url)
    health = request(base_url, "/health", timeout=timeout)["response"]
    models = request(base_url, "/v1/models", timeout=timeout)["response"]
    identity = manifest["service"]["identity"]
    if (
        any(health.get(key) != identity[key] for key in ("architecture", "max_model_len"))
        or health.get("status") != "ok"
    ):
        raise ValueError("live health differs from authenticated service identity")
    if not any(
        entry.get("id") == manifest["model"] and entry.get("max_model_len") == identity["max_model_len"]
        for entry in models.get("data", [])
    ):
        raise ValueError("live model identity/context differs from fixture manifest")
    # Preserve the authenticated external run manifest alongside the observed API snapshot.
    run_manifest = manifest["service"]["run_manifest"].copy()
    _, run_path = load_authenticated(run_manifest, fixtures_path.parent)
    run_manifest["path"] = str(run_path.resolve())
    service = dict(manifest["service"], run_manifest=run_manifest)
    receipt = {
        "schema_version": 1,
        "passed": False,
        "pair_id": pair_id,
        "pair_order": pair_order,
        "mode": mode,
        "model": manifest["model"],
        "requested_output_tokens": output_tokens,
        "fixture_manifest_sha256": hashlib.sha256(fixtures_path.read_bytes()).hexdigest(),
        "service": service,
        "launch": authenticate_service(service, fixtures_path.parent),
        "observed_health": health,
        "observed_models": models,
        "started_unix_seconds": time.time(),
        "started_at": datetime.now(timezone.utc).isoformat(),
        "measurements": [],
    }
    if receipt["launch"]["launch_started_unix_seconds"] > receipt["started_unix_seconds"]:
        raise ValueError("collection precedes the authenticated launch")
    for repeat in range(repeats):
        for fixture in manifest["fixtures"]:
            row = {key: fixture[key] for key in ("name", "request_sha256", "expected_prompt_tokens")}
            row.update(repeat=repeat, requested_output_tokens=output_tokens)
            try:
                result, effective_sha = stream_case(base_url, fixture, manifest["model"], output_tokens, timeout)
                row.update(result, effective_request_sha256=effective_sha)
            except Exception as error:
                # Do not serialize response/request bodies or private generated text.
                row.update(passed=False, flags=["failed_request"], error_type=type(error).__name__)
            receipt["measurements"].append(row)
    receipt["ended_unix_seconds"] = time.time()
    receipt["passed"] = all(row["passed"] for row in receipt["measurements"])
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return receipt


def paired_interval(values):
    """Deterministic percentile bootstrap of the median paired relative change."""
    generator = random.Random(0)
    samples = sorted(statistics.median(generator.choices(values, k=len(values))) for _ in range(BOOTSTRAP_SAMPLES))
    return {
        "median": statistics.median(values),
        "ci95": [samples[int(0.025 * BOOTSTRAP_SAMPLES)], samples[int(0.975 * BOOTSTRAP_SAMPLES)]],
        "method": "paired-percentile-bootstrap-of-median",
        "pairs": len(values),
    }


def compare_pairs(index_path):
    index_path = Path(index_path)
    index = json.loads(index_path.read_bytes())
    if index.get("schema_version") != 1 or not index.get("pairs"):
        raise ValueError("authenticated schema_version=1 pair index required")
    required = index.get("required_fixtures")
    if not isinstance(required, list) or not required or len(set(required)) != len(required):
        raise ValueError("explicit unique required_fixtures coverage required")
    grouped, ids, consumed, launches, configuration = {}, set(), set(), set(), None
    fixture_signatures, last_pair_end = {}, 0
    for entry in index["pairs"]:
        pair_id, order = entry["pair_id"], entry["order"]
        if pair_id in ids or order not in (["baseline", "candidate"], ["candidate", "baseline"]):
            raise ValueError("duplicate pair ID or invalid order")
        ids.add(pair_id)
        loaded = {}
        for mode in ("baseline", "candidate"):
            receipt, path = load_authenticated(entry[mode], index_path.parent)
            if path.resolve() in consumed:
                raise ValueError("original receipt reused across pairs")
            consumed.add(path.resolve())
            if receipt.get("pair_id") != pair_id or receipt.get("mode") != mode or receipt.get("passed") is not True:
                raise ValueError("pair receipt identity or validation failed")
            if receipt.get("pair_order") != f"{order[0]}-first":
                raise ValueError("receipt does not attest actual pair order")
            contract = authenticate_service(receipt["service"], path.parent)
            if receipt.get("launch") != contract or contract["launch_id"] in launches:
                raise ValueError("launch identity reused or authentication differs")
            launches.add(contract["launch_id"])
            loaded[mode] = receipt
        first, second = (loaded[mode] for mode in order)
        if not (
            first["launch"]["launch_started_unix_seconds"]
            <= first["started_unix_seconds"]
            <= first["ended_unix_seconds"]
            <= second["launch"]["launch_started_unix_seconds"]
            <= second["started_unix_seconds"]
            <= second["ended_unix_seconds"]
        ):
            raise ValueError("receipts do not establish actual sequential launch order")
        if first["launch"]["launch_started_unix_seconds"] < last_pair_end:
            raise ValueError("pair index must preserve chronological, nonoverlapping launches")
        last_pair_end = second["ended_unix_seconds"]
        baseline, candidate = loaded["baseline"], loaded["candidate"]
        signature = []
        for receipt in (baseline, candidate):
            identity, service = receipt["service"]["identity"], receipt["service"]
            signature.append(
                {
                    "model": receipt["model"],
                    "checkpoint": identity["checkpoint_sha256"],
                    "parser": identity["parser"],
                    "buffering_policy": identity["buffering_policy"],
                    "max_model_len": identity["max_model_len"],
                    "warmup_policy": service["warmup"]["policy"],
                    "cache_policy": service["cache_policy"]["mode"],
                    "output_tokens": receipt["requested_output_tokens"],
                }
            )
        if signature[0] != signature[1] or (configuration is not None and configuration != signature[0]):
            raise ValueError("incompatible parser/buffering, checkpoint, cache, warmup or request configuration")
        configuration = signature[0]
        rows = [
            {(row["name"], row["repeat"]): row for row in receipt["measurements"]} for receipt in (baseline, candidate)
        ]
        if (
            not rows[0]
            or rows[0].keys() != rows[1].keys()
            or any(
                len(rows[i]) != len(loaded[mode]["measurements"]) for i, mode in enumerate(("baseline", "candidate"))
            )
        ):
            raise ValueError("paired fixture/repetition identities differ")
        for key, reference in rows[0].items():
            trial = rows[1][key]
            if any(
                set(row.get("flags", [])) - INFORMATIONAL_FLAGS or row.get("failure_flags") or not row.get("passed")
                for row in (reference, trial)
            ):
                raise ValueError("invalid cold measurement cannot pass comparison")
            if any(row.get("metric_boundary") != METRIC_BOUNDARY for row in (reference, trial)):
                raise ValueError("incompatible API measurement boundary")
            for field in (
                "request_sha256",
                "effective_request_sha256",
                "expected_prompt_tokens",
                "requested_output_tokens",
            ):
                if reference[field] != trial[field]:
                    raise ValueError(f"paired fixture configuration mismatch: {field}")
            fixture_signature = {
                field: reference[field]
                for field in ("request_sha256", "effective_request_sha256", "expected_prompt_tokens")
            }
            if key[0] in fixture_signatures and fixture_signatures[key[0]] != fixture_signature:
                raise ValueError("fixture identity changed across launch pairs")
            fixture_signatures[key[0]] = fixture_signature
            grouped.setdefault(key[0], []).append(
                {"pair_id": pair_id, "repeat": key[1], "order": order, "baseline": reference, "candidate": trial}
            )
    if set(grouped) != set(required):
        raise ValueError("comparison does not cover all required fixtures")
    results = {}
    for name, pairs in grouped.items():
        unique_pairs = {row["pair_id"] for row in pairs}
        if len(unique_pairs) < MIN_PAIRS:
            raise ValueError(f"at least {MIN_PAIRS} launch pairs per fixture required: {name}")
        # Independent units are launches, not repeated requests within one launch.
        per_launch = {}
        for row in pairs:
            per_launch.setdefault(row["pair_id"], []).append(row)
        ordered = list(per_launch.values())
        if any(a[0]["order"] == b[0]["order"] for a, b in zip(ordered, ordered[1:])):
            raise ValueError("actual launch order must alternate per fixture")
        metrics = {}
        for metric in (
            "cold_ttft_seconds",
            "first_visible_text_seconds",
            "api_decode_tokens_per_second",
            "total_latency_seconds",
        ):
            relative, medians = [], {"baseline": [], "candidate": []}
            for launch in ordered:
                values = {}
                for mode in medians:
                    observations = [row[mode][metric] for row in launch]
                    if any(value is None for value in observations):
                        values = None
                        break
                    if any(
                        not isinstance(value, (float, int)) or not math.isfinite(value) or value <= 0
                        for value in observations
                    ):
                        raise ValueError("invalid performance metric")
                    values[mode] = statistics.median(observations)
                    medians[mode].append(values[mode])
                if values is None:
                    relative = []
                    break
                relative.append(values["candidate"] / values["baseline"] - 1)
            metrics[metric] = (
                None
                if not relative
                else {
                    **paired_interval(relative),
                    "baseline_median": statistics.median(medians["baseline"]),
                    "candidate_median": statistics.median(medians["candidate"]),
                    "positive_means": "faster" if metric == "api_decode_tokens_per_second" else "slower",
                    "metric_boundary": METRIC_BOUNDARY,
                    "uncertainty": uncertainty_metadata(),
                }
            )
        decode, ttft = metrics["api_decode_tokens_per_second"], metrics["cold_ttft_seconds"]
        gate = bool(
            decode
            and ttft
            and (
                (decode["median"] >= 0.10 and ttft["median"] <= 0.03)
                or (ttft["median"] <= -0.20 and decode["median"] >= -0.03)
            )
        )
        results[name] = {
            "launch_pairs": len(unique_pairs),
            "measurements": pairs,
            "metrics": metrics,
            "proposed_relative_gate_met": gate,
        }
    return {
        "schema_version": 1,
        "passed": True,
        "fixtures": results,
        "all_fixtures_relative_gate_met": all(row["proposed_relative_gate_met"] for row in results.values()),
        "index_sha256": hashlib.sha256(index_path.read_bytes()).hexdigest(),
        "gate_status": "performance-only; correctness and 128K qualification required separately",
        "model_forward_timing": None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-url")
    parser.add_argument("--fixtures", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--output-tokens", type=int, default=64)
    parser.add_argument("--pair-id")
    parser.add_argument("--mode", choices=("baseline", "candidate"))
    parser.add_argument("--pair-order", choices=("baseline-first", "candidate-first"))
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--compare-pairs", type=Path)
    args = parser.parse_args()
    if args.compare_pairs:
        if any((args.base_url, args.fixtures, args.pair_id, args.mode, args.pair_order)):
            parser.error("comparison accepts an index and output without collection arguments")
        result = compare_pairs(args.compare_pairs)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
            stream.write("\n")
    else:
        if not all((args.base_url, args.fixtures, args.pair_id, args.mode, args.pair_order)):
            parser.error("collection requires base-url, fixtures, pair-id, mode and pair-order")
        result = collect(
            args.base_url,
            args.fixtures,
            args.output,
            repeats=args.repeats,
            output_tokens=args.output_tokens,
            pair_id=args.pair_id,
            mode=args.mode,
            pair_order=args.pair_order,
            timeout=args.timeout,
        )
    print(json.dumps({"passed": result["passed"], "output": str(args.output)}))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
