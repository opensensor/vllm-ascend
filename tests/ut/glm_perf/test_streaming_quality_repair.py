# SPDX-License-Identifier: Apache-2.0
"""Offline admission faults and complete guarded supplemental chat workflows."""

import copy
import json
from pathlib import Path

import pytest
from test_streaming_resident import operations, workflow_controller

from tools.glm_perf import streaming_quality_repair as repair
from tools.glm_perf.streaming_protocol import (
    evidence_schema,
    file_digest,
    make_workloads,
    summarize_trials,
)
from tools.glm_perf.streaming_resident import materialize_language_inputs


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def evidence(kind, candidate, proof, result):
    schema = evidence_schema()
    return {
        "schema_version": 1,
        "kind": kind,
        "state": "complete",
        "identity": candidate.identity,
        "dispatch": candidate.dispatch,
        "coverage": schema["x-required-coverage"][kind],
        "artifacts": [{"path": proof.name, "sha256": file_digest(proof)}],
        "errors": [],
        "result": {**{key: True for key in schema["x-complete-result-rules"][kind]}, **result},
    }


@pytest.fixture
def setup(tmp_path, monkeypatch):
    live = tmp_path / "live"
    live.mkdir()
    guard, client = workflow_controller(live)
    workloads = make_workloads()
    workloads["tokenizer_hashes"] = {"tokenizer.json": "a" * 64}
    materialized = materialize_language_inputs(guard, workloads)
    rows = {row["id"]: row for row in materialized["workloads"]}
    primary = [
        {
            "variant": variant,
            "workload": "cold_6400",
            "cycle": cycle,
            "position": position,
            "complete": True,
            "reused_at_admission": 0,
            "ttft_s": 50 if variant == "reference" else 45,
            "timing_method": "streamed_token_event",
            "prompt_sha256": "d" * 64,
        }
        for cycle in range(3)
        for position, variant in enumerate(("reference", "candidate", "candidate", "reference"))
    ]
    samples = copy.deepcopy(primary)
    for variant in repair.VARIANTS:
        for name in repair.REGRESSION_CONTROLS:
            samples += [{"variant": variant, "workload": name, "complete": True, "generation_tok_s": 10}] * 3
        for name in repair.QUALITY_CONTROLS:
            sample = {
                "variant": variant,
                "workload": name,
                "complete": True,
                "request_id": f"{variant}-{name}",
                "prompt_tokens": rows[name]["materialized_prompt_tokens"],
                "prompt_sha256": rows[name]["materialized_prompt_sha256"],
                "completion_tokens": 192,
                "text": '{"remaining_jobs":3,"worker_count":3}' if name == "language" else "",
                "reasoning": "",
                "tool_calls": [{"name": "lookup_order", "arguments": '{"order_id":"ORD-2048"}'}]
                if name == "tool"
                else [],
                "events": [{"choices": [{"finish_reason": "stop"}]}],
            }
            if variant == "reference" and name == "language":
                sample.update(text="", reasoning="thinking", events=[{"choices": [{"finish_reason": "length"}]}])
            samples.append(sample)
    result = {
        "primary": primary,
        "samples": samples,
        "repeatability": summarize_trials(primary),
        "regressions": {name: 0.0 for name in repair.REGRESSION_CONTROLS},
        "language_tool_quality": {
            "reference": {"language": False, "tool": True},
            "candidate": {"language": True, "tool": True},
        },
        "promotion_eligible": False,
        "retained": "reference",
        "final_generation": guard.expected_generation,
        "final_workers": copy.deepcopy(client.workers),
    }
    result.update({key: True for key in evidence_schema()["x-complete-result-rules"]["trials"]})
    directory = tmp_path / "parent"
    directory.mkdir()
    raw = directory / "trials-raw.json"
    write(raw, {"result": result})
    parent = evidence("trials", guard.candidate, raw, result)
    proof = directory / "native-proof.json"
    write(proof, {"actual_passed_cases": "mocked complete native gate"})
    native = evidence("native_gates", guard.candidate, proof, {})
    shadow = evidence("shadow", guard.candidate, proof, {"comparisons": 344})
    paths = {
        "parent": directory / "trials.json",
        "shadow": directory / "shadow.json",
        "gates": directory / "gates.json",
        "workloads": directory / "workloads.json",
        "materialized": directory / "materialized-workloads.json",
        "raw": raw,
    }
    for key, value in (
        ("parent", parent),
        ("shadow", shadow),
        ("gates", native),
        ("workloads", workloads),
        ("materialized", materialized),
    ):
        write(paths[key], value)
    admitted = []

    def admission(reference, gates_path):
        admitted.append(gates_path)
        from tools.glm_perf.streaming_protocol import validate_evidence

        validate_evidence(json.loads(gates_path.read_text()), kind="native_gates", bindings=guard.candidate.identity)
        return guard.candidate

    monkeypatch.setattr(repair, "load_candidate", admission)
    client.calls.clear()
    return guard, client, paths, parent, admitted


def plan(setup):
    guard, _, paths, _, _ = setup
    return repair.load_repair_plan(
        guard.reference, paths["parent"], paths["shadow"], paths["gates"], paths["workloads"]
    )


def update_parent(setup, change):
    _, _, paths, parent, _ = setup
    change(parent)
    write(paths["raw"], {"result": parent["result"]})
    parent["artifacts"][0]["sha256"] = file_digest(paths["raw"])
    write(paths["parent"], parent)


def test_complete_failed_quality_parent_admits_only_full_native_and_shadow(setup):
    admitted = plan(setup)
    assert admitted.budget_failures == (
        {"variant": "reference", "workload": "language", "request_id": "reference-language"},
    )
    assert admitted.generation == setup[0].expected_generation
    assert setup[4] == [setup[2]["gates"]]
    assert not setup[1].calls


def test_mixed_single_token_prefill_has_no_decode_rate_and_is_not_a_failure(setup):
    def change(parent):
        parent["result"]["samples"] += [
            {"variant": variant, "workload": "mixed", "role": "mixed", "complete": True, "generation_tok_s": None}
            for variant in repair.VARIANTS
            for _ in range(3)
        ]

    update_parent(setup, change)
    assert plan(setup).generation == setup[0].expected_generation


@pytest.mark.parametrize(
    "fault",
    [
        "pending",
        "failed",
        "wrong_identity",
        "missing_flag",
        "no_repeatability",
        "bad_decode",
        "incomplete",
        "missing_quality",
        "stale_generation",
        "wrong_raw",
    ],
)
def test_parent_gate_failures_never_acquire_lease(setup, fault):
    def change(parent):
        result = parent["result"]
        if fault in ("pending", "failed"):
            parent["state"] = fault
        elif fault == "wrong_identity":
            parent["identity"] = {**parent["identity"], "source_sha256": "b" * 64}
        elif fault == "missing_flag":
            result["launch_audit"] = False
        elif fault == "no_repeatability":
            for sample in result["primary"]:
                sample["ttft_s"] = 50
            result["repeatability"] = summarize_trials(result["primary"])
        elif fault == "bad_decode":
            for sample in result["samples"]:
                if sample.get("variant") == "candidate" and sample.get("workload") == "decode_c4":
                    sample["generation_tok_s"] = 9
            result["regressions"]["decode_c4"] = 1 - 9 / 10
        elif fault == "incomplete":
            result["samples"][0]["complete"] = False
        elif fault == "missing_quality":
            result["samples"] = [s for s in result["samples"] if s.get("workload") != "tool"]
        elif fault == "stale_generation":
            result["final_generation"] = "b" * 32

    update_parent(setup, change)
    if fault == "wrong_raw":
        parent = json.loads(setup[2]["parent"].read_text())
        parent["result"]["objective_10pct_met"] = False
        write(setup[2]["parent"], parent)
    with pytest.raises(ValueError):
        plan(setup)
    assert not setup[1].calls


@pytest.mark.parametrize("kind", ["native", "shadow"])
def test_partial_correctness_evidence_cannot_be_repaired(setup, kind):
    path = setup[2]["gates" if kind == "native" else "shadow"]
    report = json.loads(path.read_text())
    report["state"] = "pending"
    write(path, report)
    with pytest.raises(ValueError, match="not complete"):
        plan(setup)
    assert not setup[1].calls


@pytest.mark.parametrize("fault", ["below_cap", "wrong_math", "stop", "missing_finish", "unrelated_prose"])
def test_uncapped_or_completed_wrong_quality_is_not_budget_failure(setup, fault):
    def change(parent):
        sample = next(s for s in parent["result"]["samples"] if s.get("request_id") == "reference-language")
        if fault == "below_cap":
            sample["completion_tokens"] = 191
        elif fault == "wrong_math":
            sample["text"] = '{"remaining_jobs":4,"worker_count":3}'
        elif fault == "stop":
            sample["events"][0]["choices"][0]["finish_reason"] = "stop"
        elif fault == "missing_finish":
            sample["events"] = []
        else:
            sample["text"] = "Four jobs remain."

    update_parent(setup, change)
    with pytest.raises(ValueError, match="not solely response-budget-limited"):
        plan(setup)


def test_successful_parent_quality_refuses_unnecessary_repair(setup):
    def change(parent):
        sample = next(s for s in parent["result"]["samples"] if s.get("request_id") == "reference-language")
        sample["text"] = '{"remaining_jobs":3,"worker_count":3}'
        parent["result"]["language_tool_quality"]["reference"]["language"] = True

    update_parent(setup, change)
    with pytest.raises(ValueError, match="unnecessary"):
        plan(setup)


@pytest.mark.parametrize("fault", ["partial", "wrong_arguments", "wrong_function"])
def test_tool_truncation_is_admissible_but_completed_wrong_call_is_not(setup, fault):
    def change(parent):
        sample = next(s for s in parent["result"]["samples"] if s.get("request_id") == "reference-tool")
        sample["events"] = [{"choices": [{"finish_reason": "length"}]}]
        sample["tool_calls"] = [
            {
                "name": "other" if fault == "wrong_function" else "lookup_order",
                "arguments": '{"order_id":' if fault == "partial" else '{"order_id":"ORD-WRONG"}',
            }
        ]
        parent["result"]["language_tool_quality"]["reference"]["tool"] = False

    update_parent(setup, change)
    if fault == "partial":
        assert len(plan(setup).budget_failures) == 2
    else:
        with pytest.raises(ValueError, match="not solely response-budget-limited"):
            plan(setup)


@pytest.mark.parametrize("fault", ["payload", "token_ids", "tokenizer", "rendered_hash"])
def test_original_materialized_sources_are_preserved(setup, fault):
    path = setup[2]["materialized"]
    doc = json.loads(path.read_text())
    language = next(row for row in doc["workloads"] if row["id"] == "language")
    if fault == "payload":
        language["input"]["messages"][0]["content"] = "changed"
    elif fault == "token_ids":
        language["materialized_token_ids"] = [33]
    elif fault == "tokenizer":
        doc["tokenizer_hashes"]["tokenizer.json"] = "b" * 64
    else:
        language["rendered_input_sha256"] = "b" * 64
    write(path, doc)
    with pytest.raises(ValueError):
        plan(setup)


@pytest.mark.parametrize("promote", [False, True])
def test_actual_guarded_repair_preserves_prompts_both_variants_and_retests_tool(setup, promote):
    guard, client, _, _, _ = setup
    admitted = plan(setup)
    parents = dict(admitted.files)
    report = repair.repair_workflow(guard, admitted, promote=promote)
    assert report["state"] == "complete" and report["promotion_eligible"]
    assert report["retained"] == ("candidate" if promote else "reference")
    assert not report["benchmark_rerun"]
    assert len(report["samples"]) == 4
    assert set(report["quality"]) == {"reference", "candidate"}
    assert all(all(checks.values()) for checks in report["quality"].values())
    assert all(file_digest(Path(path)) == digest for path, digest in parents.items())
    assert not client.maintenance and not client.paused
    inputs = {
        row["id"]: row["input"] for row in admitted.workloads["workloads"] if row["id"] in repair.QUALITY_CONTROLS
    }
    assert len(client.inference) == 4
    for request in client.inference:
        payload = copy.deepcopy(request["payload"])
        name = "tool" if "tools" in payload else "language"
        assert payload.pop("model") == guard.reference["runtime"]["settings"]["model_name"]
        assert payload.pop("max_tokens") == 768
        assert payload == {key: value for key, value in inputs[name].items() if key != "max_tokens"}
        assert request["prompt"] == [11, 22]
    assert all(proof["only_response_cap_changed"] for proof in report["prompt_proofs"].values())
    assert operations(client).index("/tokenize") < operations(client).index("resident_prepare")
    assert json.loads((guard.output_dir / "prospective-design.json").read_text())["date"] == "2026-10-09"


def test_failed_supplement_restores_reference_even_when_promotion_requested(setup):
    guard, client, _, _, _ = setup
    original = client.stream

    def wrong(*args, **kwargs):
        sample = original(*args, **kwargs)
        if client.version != 1021:
            sample.update(text="wrong", tool_calls=[])
        return sample

    client.stream = wrong
    report = repair.repair_workflow(guard, plan(setup), promote=True)
    assert report["state"] == "rejected" and not report["promotion_eligible"]
    assert report["retained"] == "reference" and client.version == 1021
    assert not client.maintenance and not client.paused


def test_changed_renderer_aborts_before_requests_and_restores(setup):
    guard, client, _, _, _ = setup
    admitted = plan(setup)
    client.tokens = [99]
    with pytest.raises(ValueError, match="renderer IDs/settings/source"):
        repair.repair_workflow(guard, admitted)
    assert not client.inference and client.version == 1021
    assert not client.maintenance and not client.paused
    assert json.loads((guard.output_dir / "quality-repair.json").read_text())["state"] == "failed"


def test_parent_changes_after_admission_cannot_start_repair(setup):
    guard, client, paths, _, _ = setup
    admitted = plan(setup)
    paths["parent"].write_text("changed")
    with pytest.raises(ValueError, match="evidence changed"):
        repair.repair_workflow(guard, admitted)
    assert not client.calls


def test_active_user_request_never_pauses_or_switches(setup):
    guard, client, _, _, _ = setup
    admitted = plan(setup)
    client.active_count = 1
    with pytest.raises(RuntimeError, match="active"):
        repair.repair_workflow(guard, admitted)
    assert not any(name.startswith("/pause") or name == "resident_prepare" for name in operations(client))


def test_output_cannot_overwrite_parent_directory(setup):
    guard, client, paths, _, _ = setup
    admitted = plan(setup)
    guard.output_dir = paths["parent"].parent
    with pytest.raises(ValueError, match="overwrite"):
        repair.repair_workflow(guard, admitted)
    assert not client.calls


def test_unresolved_apply_during_supplement_never_restores_or_resumes(setup):
    guard, client, _, _, _ = setup
    admitted = plan(setup)
    client.fault = ("resident_apply", "timeout")
    with pytest.raises(RuntimeError, match="unresolved"):
        repair.repair_workflow(guard, admitted)
    assert operations(client)[-1] == "resident_apply"
    assert client.maintenance and client.paused
    report = json.loads((guard.output_dir / "quality-repair.json").read_text())
    assert report["state"] == "failed" and not report["promotion_eligible"]


def test_default_cli_validates_offline_without_constructing_transport(setup, monkeypatch, capsys):
    import sys

    guard, client, paths, _, _ = setup
    monkeypatch.setattr(repair, "verify_reference", lambda *args: guard.reference)

    def forbidden(*args):
        pytest.fail("dry-run must not create live transport")

    monkeypatch.setattr(repair, "StreamingTransport", forbidden)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "streaming_quality_repair",
            "--reference",
            str(paths["parent"]),
            "--parent-trials",
            str(paths["parent"]),
            "--shadow-report",
            str(paths["shadow"]),
            "--gates",
            str(paths["gates"]),
            "--workloads",
            str(paths["workloads"]),
            "--output-dir",
            str(guard.output_dir),
        ],
    )
    repair.main()
    assert json.loads(capsys.readouterr().out)["dry_run"]
    assert not client.calls
