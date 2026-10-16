# SPDX-License-Identifier: Apache-2.0
"""API accounting, authenticated cold lifecycle and paired statistics tests."""

import copy
import hashlib
import json
from pathlib import Path

import pytest

from tools.sidecard import performance_suite as suite


def usage(prompt=2048, completed=4, cached=0):
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completed,
        "prompt_tokens_details": {"cached_tokens": cached},
        "total_tokens": prompt + completed,
    }


def stream_events():
    return [
        (0.2, {"choices": [{"delta": {"role": "assistant"}}]}),
        (1.0, {"choices": [{"delta": {"reasoning_content": "Think"}}]}),
        (1.1, {"choices": [{"delta": {"reasoning_content": " again"}}]}),
        (1.2, {"choices": [{"delta": {"content": "Answer"}}]}),
        (1.3, {"choices": [{"delta": {"content": "."}, "finish_reason": "length"}]}),
        (1.4, {"choices": [], "usage": usage()}),
        (1.5, "[DONE]"),
    ]


def measurement(events=None):
    return suite.measure_events(
        events or stream_events(), output_tokens=4, expected_prompt_tokens=2048, elapsed_seconds=1.6
    )


def test_reasoning_included_from_usage_not_chunk_count():
    row = measurement()
    assert row["passed"]
    assert row["completion_tokens"] == 4
    assert row["cold_ttft_seconds"] == 1
    assert row["first_visible_text_seconds"] == 1.2
    assert row["api_decode_tokens_per_second"] == pytest.approx(10)
    assert row["total_latency_seconds"] == 1.6
    assert row["model_forward_timing"] is None


@pytest.mark.parametrize(
    "change,flag",
    [
        ("eos", "early_eos"),
        ("cache", "prefix_reuse"),
        ("missing_cache", "missing_cache_usage"),
        ("no_usage", "missing_usage"),
        ("no_done", "incomplete_stream"),
        ("prompt", "prompt_identity_mismatch"),
        ("error", "failed_request"),
        ("invalid_cache", "invalid_cache_usage"),
        ("overshoot", "output_budget_exceeded"),
    ],
)
def test_stream_failure_modes(change, flag):
    events = stream_events()
    if change == "eos":
        events[5][1]["usage"]["completion_tokens"] = 3
    elif change == "cache":
        events[5][1]["usage"]["prompt_tokens_details"]["cached_tokens"] = 1024
    elif change == "missing_cache":
        events[5][1]["usage"].pop("prompt_tokens_details")
    elif change == "no_usage":
        events.pop(5)
    elif change == "no_done":
        events.pop()
    elif change == "prompt":
        events[5][1]["usage"]["prompt_tokens"] = 2049
    elif change == "error":
        events[0][1]["error"] = {"message": "request failed"}
    elif change == "invalid_cache":
        events[5][1]["usage"]["prompt_tokens_details"]["cached_tokens"] = False
    else:
        events[5][1]["usage"]["completion_tokens"] = 5
    row = measurement(events)
    assert not row["passed"] and flag in row["flags"]


def test_tool_only_output_preserves_unobservable_visible_text():
    events = stream_events()
    for _, event in events[1:5]:
        event["choices"][0]["delta"] = {"tool_calls": [{"function": {"arguments": "x"}}]}
    row = measurement(events)
    assert row["passed"] and row["first_visible_text_seconds"] is None


@pytest.mark.parametrize("kind", ["special-token", "unicode", "reasoning", "tool"])
def test_buffered_stream_uses_accounting_proxy_without_assuming_one_token_per_event(kind):
    events = stream_events()
    if kind == "special-token":
        events[2][1]["choices"][0]["delta"] = {}
    elif kind == "unicode":
        events[2][1]["choices"][0]["delta"] = {"content": ""}
        events[3][1]["choices"][0]["delta"] = {"content": "你好"}
    elif kind == "reasoning":
        events[2][1]["choices"][0]["delta"] = {}
        events[3][1]["choices"][0]["delta"] = {"reasoning_content": "buffered reasoning"}
    else:
        events[2][1]["choices"][0]["delta"] = {}
        events[3][1]["choices"][0]["delta"] = {"tool_calls": [{"function": {"arguments": "{}"}}]}
    row = measurement(events)
    assert row["passed"] and not row["failure_flags"]
    assert "coalesced_emission" in row["flags"]
    assert row["completion_tokens"] == 4 and row["output_emission_events"] == 3
    assert row["api_decode_tokens_per_second"] == pytest.approx(10)
    assert row["metric_boundary"] == "usage_accounted_output_emission_proxy"
    assert "unknown" in row["uncertainty"]["buffering_bias"]
    assert "excludes" in row["uncertainty"]["confidence_interval_scope"]


@pytest.mark.parametrize("remaining", [0, 1])
def test_zero_or_one_nonempty_emission_cannot_measure_decode(remaining):
    events = stream_events()
    for index, (_, event) in enumerate(events[1:5]):
        if index >= remaining:
            event["choices"][0]["delta"] = {}
    row = measurement(events)
    assert not row["passed"]
    assert "unobservable_decode_span" in row["failure_flags"]
    assert row["api_decode_tokens_per_second"] is None


def test_monotonic_stream_time_required():
    events = stream_events()
    events[2] = (0.5, events[2][1])
    with pytest.raises(ValueError, match="nonmonotonic"):
        measurement(events)


def write_reference(path, value):
    path.write_text(json.dumps(value))
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def service(root, launch="launch-1", started=1, cache_enabled=False):
    identity = {field: "a" * 64 for field in suite.IDENTITY_HASH_FIELDS}
    identity.update(
        parser="glm53",
        buffering_policy="incremental-detokenizer-and-glm-parser-v1",
        architecture="cuda-backbone-ascend-experts",
        max_model_len=131072,
    )
    contract = {
        "identity": identity,
        "prefix_cache_enabled": cache_enabled,
        "startup_gates_passed": True,
        "warmup_completed": True,
        "warmup_policy": "fixed-4-tokens",
        "launch_id": launch,
        "launch_started_unix_seconds": started,
    }
    reference = write_reference(root / f"{launch}-run.json", {"performance_contract": contract})
    return {
        "identity": identity,
        "run_manifest": reference,
        "cache_policy": {"mode": "disabled", "evidence": "authenticated runtime cache options"},
        "startup_gates": {"passed": True, "evidence": "startup-gates.json"},
        "warmup": {"completed": True, "policy": "fixed-4-tokens", "evidence": "warmup.json"},
    }, contract


def manifest(root):
    body = {
        "messages": [{"role": "user", "content": "PRIVATE CODE"}],
        "tools": [{"type": "function", "function": {"name": f"tool_{i}"}} for i in range(23)],
        "ignore_eos": True,
    }
    return {
        "schema_version": 1,
        "model": "glm-sidecard",
        "service": service(root)[0],
        "fixtures": [
            {
                "name": "full-code",
                "request": body,
                "request_sha256": suite.canonical_sha256(body),
                "expected_prompt_tokens": 2048,
                "expected_tool_count": 23,
            }
        ],
    }


@pytest.mark.parametrize("mutation", ["request", "tools", "identity", "cache", "gates", "hash"])
def test_private_fixture_and_run_authentication(tmp_path, mutation):
    value = manifest(tmp_path)
    if mutation == "request":
        value["fixtures"][0]["request"]["messages"][0]["content"] = "COMPACT"
    elif mutation == "tools":
        value["fixtures"][0]["expected_tool_count"] = 22
    elif mutation == "identity":
        value["service"]["identity"]["config_sha256"] = "b" * 64
    elif mutation == "cache":
        value["service"], _ = service(tmp_path, "cache-enabled", cache_enabled=True)
    elif mutation == "gates":
        value["service"]["startup_gates"]["passed"] = False
    else:
        value["service"]["run_manifest"]["sha256"] = "b" * 64
    with pytest.raises(ValueError):
        suite.validate_manifest(value, tmp_path)


def collection_arguments(path, output):
    return dict(
        base_url="http://example.invalid/v1",
        fixtures_path=path,
        output=output,
        repeats=1,
        output_tokens=4,
        pair_id="pair-1",
        mode="baseline",
        pair_order="baseline-first",
    )


def test_collection_preserves_private_request_and_records_external_identity(tmp_path, monkeypatch):
    value = manifest(tmp_path)
    path, output = tmp_path / "private.json", tmp_path / "receipt.json"
    path.write_text(json.dumps(value))
    observed = []

    def request(base, endpoint, **kwargs):
        observed.append((base, endpoint))
        return {
            "response": (
                {"status": "ok", "architecture": "cuda-backbone-ascend-experts", "max_model_len": 131072}
                if endpoint == "/health"
                else {"data": [{"id": "glm-sidecard", "max_model_len": 131072}]}
            )
        }

    def stream(base, fixture, model, tokens, timeout):
        assert fixture["request"] == value["fixtures"][0]["request"]
        return measurement(), "e" * 64

    monkeypatch.setattr(suite, "request", request)
    monkeypatch.setattr(suite, "stream_case", stream)
    receipt = suite.collect(**collection_arguments(path, output))
    assert receipt["passed"]
    assert observed[0] == ("http://example.invalid", "/health")
    assert receipt["launch"]["launch_id"] == "launch-1"
    assert "PRIVATE CODE" not in output.read_text() and "tool_0" not in output.read_text()
    with pytest.raises(ValueError, match="overwrite"):
        suite.collect(**collection_arguments(path, output))


def test_failed_request_retained_without_private_error_body(tmp_path, monkeypatch):
    value = manifest(tmp_path)
    path, output = tmp_path / "private.json", tmp_path / "receipt.json"
    path.write_text(json.dumps(value))
    monkeypatch.setattr(
        suite,
        "request",
        lambda base, endpoint, **kwargs: {
            "response": (
                {"status": "ok", "architecture": "cuda-backbone-ascend-experts", "max_model_len": 131072}
                if endpoint == "/health"
                else {"data": [{"id": "glm-sidecard", "max_model_len": 131072}]}
            )
        },
    )

    def fail(*args):
        raise ValueError("PRIVATE CODE error echo")

    monkeypatch.setattr(suite, "stream_case", fail)
    receipt = suite.collect(**collection_arguments(path, output))
    assert not receipt["passed"] and receipt["measurements"][0]["flags"] == ["failed_request"]
    assert "PRIVATE CODE" not in output.read_text()


def test_fresh_process_policy_prevents_repeated_cold_claims(tmp_path):
    value = manifest(tmp_path)
    value["service"]["cache_policy"]["mode"] = "fresh-process-per-request"
    path = tmp_path / "private.json"
    path.write_text(json.dumps(value))
    arguments = collection_arguments(path, tmp_path / "out.json")
    arguments["repeats"] = 2
    with pytest.raises(ValueError, match="one fixture/request"):
        suite.collect(**arguments)


def pair_index(root, count=5, fixtures=("2k", "8k", "full-code")):
    pairs = []
    for index in range(count):
        pair_id = f"pair-{index}"
        order = ["baseline", "candidate"] if index % 2 == 0 else ["candidate", "baseline"]
        entry = {"pair_id": pair_id, "order": order}
        for mode in order:
            start = 100 + index * 10 + order.index(mode) * 3
            spec, contract = service(root, f"{pair_id}-{mode}", start)
            receipt = {
                "schema_version": 1,
                "passed": True,
                "pair_id": pair_id,
                "mode": mode,
                "pair_order": f"{order[0]}-first",
                "model": "glm-sidecard",
                "service": spec,
                "launch": contract,
                "requested_output_tokens": 4,
                "started_unix_seconds": start + 1,
                "ended_unix_seconds": start + 2,
                "measurements": [],
            }
            for name in fixtures:
                row = dict(
                    measurement(),
                    name=name,
                    repeat=0,
                    request_sha256="c" * 64,
                    effective_request_sha256="d" * 64,
                    expected_prompt_tokens=2048,
                    requested_output_tokens=4,
                )
                if mode == "candidate":
                    row["api_decode_tokens_per_second"] *= 1.15
                receipt["measurements"].append(row)
            entry[mode] = write_reference(root / f"{pair_id}-{mode}.json", receipt)
        pairs.append(entry)
    path = root / "api-pairs.json"
    path.write_text(json.dumps({"schema_version": 1, "required_fixtures": list(fixtures), "pairs": pairs}))
    return path


def rewrite_receipt(index_path, pair=0, mode="candidate", mutate=None):
    index = json.loads(index_path.read_text())
    path = Path(index["pairs"][pair][mode]["path"])
    receipt = json.loads(path.read_text())
    mutate(receipt)
    index["pairs"][pair][mode] = write_reference(path, receipt)
    index_path.write_text(json.dumps(index))


def test_paired_comparison_preserves_each_fixture_and_uncertainty(tmp_path):
    result = suite.compare_pairs(pair_index(tmp_path))
    assert result["passed"] and result["all_fixtures_relative_gate_met"]
    assert set(result["fixtures"]) == {"2k", "8k", "full-code"}
    for fixture in result["fixtures"].values():
        metric = fixture["metrics"]["api_decode_tokens_per_second"]
        assert metric["median"] == pytest.approx(0.15)
        assert metric["ci95"] == pytest.approx([0.15, 0.15])
        assert metric["pairs"] == 5


def test_buffered_pair_proxy_comparisons_preserve_informational_flags(tmp_path):
    path = pair_index(tmp_path)
    for pair in range(5):
        for mode in ("baseline", "candidate"):

            def buffer(receipt):
                for row in receipt["measurements"]:
                    row.update(flags=["coalesced_emission"], output_emission_events=2)

            rewrite_receipt(path, pair, mode, buffer)
    result = suite.compare_pairs(path)
    assert result["passed"] and result["all_fixtures_relative_gate_met"]
    for fixture in result["fixtures"].values():
        assert fixture["measurements"][0]["baseline"]["flags"] == ["coalesced_emission"]
        metric = fixture["metrics"]["api_decode_tokens_per_second"]
        assert metric["metric_boundary"] == "usage_accounted_output_emission_proxy"
        assert "excludes" in metric["uncertainty"]["confidence_interval_scope"]


@pytest.mark.parametrize("field", ["parser", "buffering_policy"])
def test_authenticated_parser_or_buffering_mismatch_prevents_comparison(tmp_path, field):
    path = pair_index(tmp_path)

    def alter_policy(receipt):
        receipt["service"]["identity"][field] = "different-policy"
        receipt["launch"]["identity"][field] = "different-policy"
        reference = receipt["service"]["run_manifest"]
        run_path = Path(reference["path"])
        run = json.loads(run_path.read_text())
        run["performance_contract"]["identity"][field] = "different-policy"
        receipt["service"]["run_manifest"] = write_reference(run_path, run)

    rewrite_receipt(path, mutate=alter_policy)
    with pytest.raises(ValueError, match="incompatible parser/buffering"):
        suite.compare_pairs(path)


def test_fixture_regression_cannot_hide_in_average(tmp_path):
    path = pair_index(tmp_path)
    for pair in range(5):

        def regress(receipt):
            receipt["measurements"][2]["api_decode_tokens_per_second"] = 8

        rewrite_receipt(path, pair, mutate=regress)
    result = suite.compare_pairs(path)
    assert result["passed"] and not result["all_fixtures_relative_gate_met"]
    assert not result["fixtures"]["full-code"]["proposed_relative_gate_met"]


@pytest.mark.parametrize("mutation", ["parser", "order", "reuse", "fixture", "failed", "warmup", "launch"])
def test_comparison_rejects_invalid_pair_receipts(tmp_path, mutation):
    path = pair_index(tmp_path)

    def mutate(receipt):
        if mutation == "parser":
            receipt["service"]["identity"]["parser"] = "other"
        elif mutation == "order":
            receipt["started_unix_seconds"] = 99
        elif mutation == "fixture":
            receipt["measurements"][0]["effective_request_sha256"] = "e" * 64
        elif mutation == "failed":
            receipt["measurements"][0]["flags"] = ["unobservable_decode_span"]
        elif mutation == "warmup":
            receipt["service"]["warmup"]["policy"] = "different"
        else:
            receipt["launch"]["launch_id"] = "invented-label"

    if mutation == "reuse":
        index = json.loads(path.read_text())
        index["pairs"][1]["baseline"] = copy.deepcopy(index["pairs"][0]["baseline"])
        path.write_text(json.dumps(index))
    else:
        rewrite_receipt(path, mutate=mutate)
    with pytest.raises(ValueError):
        suite.compare_pairs(path)


def test_minimum_independent_pairs_not_repeat_count(tmp_path):
    path = pair_index(tmp_path, count=4)
    with pytest.raises(ValueError, match="at least 5"):
        suite.compare_pairs(path)


def test_coverage_required(tmp_path):
    path = pair_index(tmp_path, fixtures=("2k", "8k"))
    index = json.loads(path.read_text())
    index["required_fixtures"].append("full-code")
    path.write_text(json.dumps(index))
    with pytest.raises(ValueError, match="cover all"):
        suite.compare_pairs(path)


def test_authenticated_file_tamper_detected(tmp_path):
    path = pair_index(tmp_path)
    index = json.loads(path.read_text())
    candidate = Path(index["pairs"][0]["candidate"]["path"])
    candidate.write_text(candidate.read_text() + " ")
    with pytest.raises(ValueError, match="authentication failed"):
        suite.compare_pairs(path)


def test_fixture_identity_must_remain_constant_across_pairs(tmp_path):
    path = pair_index(tmp_path)
    for mode in ("baseline", "candidate"):
        rewrite_receipt(
            path, pair=1, mode=mode, mutate=lambda receipt: receipt["measurements"][0].update(request_sha256="f" * 64)
        )
    with pytest.raises(ValueError, match="changed across"):
        suite.compare_pairs(path)


def test_pair_index_cannot_reorder_actual_launches(tmp_path):
    path = pair_index(tmp_path)
    index = json.loads(path.read_text())
    index["pairs"][0], index["pairs"][2] = index["pairs"][2], index["pairs"][0]
    path.write_text(json.dumps(index))
    with pytest.raises(ValueError, match="chronological"):
        suite.compare_pairs(path)


def test_stream_http_preserves_tools_and_uses_usage(tmp_path, monkeypatch):
    fixture = manifest(tmp_path)["fixtures"][0]
    original = copy.deepcopy(fixture)
    observed = []
    ticks = iter([10, 10.2, 11, 11.1, 11.2, 11.3, 11.4, 11.5, 11.6])

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def __iter__(self):
            yield b": heartbeat\n"
            for _, event in stream_events():
                payload = "[DONE]" if event == "[DONE]" else json.dumps(event)
                yield ("data: " + payload + "\n").encode()
                yield b"\n"

    def open_request(query, timeout):
        observed.append((query.full_url, json.loads(query.data), timeout))
        return Response()

    monkeypatch.setattr(suite.urllib.request, "urlopen", open_request)
    monkeypatch.setattr(suite.time, "monotonic", lambda: next(ticks))
    result, effective_sha = suite.stream_case("http://example.invalid", fixture, "glm-sidecard", 4, 30)
    assert result["passed"]
    assert observed[0][0].endswith("/v1/chat/completions")
    assert observed[0][1]["tools"] == original["request"]["tools"]
    assert observed[0][1]["ignore_eos"] is True
    assert observed[0][1]["stream_options"] == {"include_usage": True}
    assert fixture == original and effective_sha == suite.canonical_sha256(observed[0][1])


def test_interval_is_deterministic_and_nonzero_for_variable_pairs():
    first = suite.paired_interval([0.01, 0.02, 0.03, 0.04, 0.05])
    assert first == suite.paired_interval([0.01, 0.02, 0.03, 0.04, 0.05])
    assert first["ci95"][0] < first["ci95"][1]
