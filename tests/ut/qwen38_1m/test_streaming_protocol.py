# SPDX-License-Identifier: Apache-2.0
"""Offline protocol failures must never be mistaken for hardware admission."""

import copy
import importlib.util
import json
from pathlib import Path

import pytest

from tools.qwen4exp.streaming_protocol import (
    REQUIRED_GATES,
    canonical_sha256,
    freeze_reference,
    seal_manifest,
    source_sha256,
    validate_evidence,
    validate_manifest,
    write_append_only,
)

ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture(scope="module")
def reference():
    return freeze_reference(ROOT)


def evidence_for(manifest):
    identities = {
        "manifest_sha256": manifest["manifest_sha256"],
        "source_sha256": manifest["source"]["source_sha256"],
        "arithmetic_sha256": manifest["arithmetic_sha256"],
        "checkpoint_sha256": manifest["checkpoint"]["sha256"],
        "serving_bundle_sha256": manifest["serving_bundle"]["sha256"],
        "candidate_sha256": manifest["candidate"]["candidate_sha256"] if manifest.get("candidate") else None,
    }
    return {
        "schema_version": 1,
        **identities,
        "rank_receipts": [
            dict(identities, rank=rank, pid=100 + rank, execution_namespace="fixture") for rank in range(4)
        ],
        "trials": [],
        "gates": {name: {"status": "pending"} for name in REQUIRED_GATES},
    }


def test_frozen_source_is_committed_and_historical_only(reference):
    assert reference["source"]["commit"].startswith("2a2a4e416")
    assert reference["offline_reference"] == "complete"
    assert reference["live_reference"] == "incomplete"
    assert reference["live_runtime"] == {"status": "unknown", "queried": False}
    assert reference["historical_runtime"]["status"] == "thermally_unqualified"
    assert reference["checkpoint"]["sha256"] is None
    assert reference["serving_bundle"]["sha256"] is None
    assert reference["staged_native_build"]["receipt"]["hardware_validated"] is False
    assert validate_manifest(reference, ROOT) is reference


def test_complete_workload_matrix_and_fixed_c4_capacity(reference):
    workloads = reference["workloads"]
    assert len(workloads) == 11 * 4 * 3
    assert {w["submitted_requests"] for w in workloads} == {1, 2, 3, 4}
    assert {w["mtp_tokens"] for w in workloads} == {0, 1, 2}
    assert {w["kind"] for w in workloads} == {"text", "tool", "image", "mixed"}
    assert all(w["max_active_requests"] == 3 for w in workloads)
    assert all(w["token_ids"] is None and w["input_status"].startswith("pending_") for w in workloads)


def test_json_identity_is_deterministic_and_strict():
    assert canonical_sha256({"a": 1, "b": [2]}) == canonical_sha256({"b": [2], "a": 1})
    with pytest.raises(ValueError):
        canonical_sha256({"nan": float("nan")})


def test_append_only_output(tmp_path, reference):
    path = tmp_path / "receipt.json"
    write_append_only(path, reference)
    assert json.loads(path.read_text()) == reference
    with pytest.raises(FileExistsError):
        write_append_only(path, {"changed": True})
    assert json.loads(path.read_text()) == reference


def test_no_device_runtime_dependencies():
    path = ROOT / "tools/qwen4exp/streaming_protocol.py"
    spec = importlib.util.spec_from_file_location("protocol_isolated", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert not any(name in module.__dict__ for name in ("torch", "torch_npu", "requests", "ctypes"))


@pytest.mark.parametrize(
    "change", ["source", "arithmetic", "capacity", "live", "thermal", "duplicate", "missing", "cache", "warm"]
)
def test_inconsistent_manifest_rejected(reference, change):
    value = copy.deepcopy(reference)
    if change == "source":
        value["source"]["assets"][0]["sha256"] = "0" * 64
    elif change == "arithmetic":
        value["arithmetic"]["grouped_activation"] = "different FP32 baseline"
    elif change == "capacity":
        value["comparison_capacity"]["max_num_seqs"] = 4
    elif change == "live":
        value["live_runtime"]["status"] = "running"
    elif change == "thermal":
        value["historical_runtime"]["status"] = "qualified"
    elif change == "duplicate":
        value["workloads"][-1] = value["workloads"][0]
    elif change == "missing":
        value["workloads"].pop()
    elif change == "cache":
        value["workloads"][0]["cache_policy"] = "warm"
    elif change == "warm":
        next(w for w in value["workloads"] if w["cache_policy"] == "warm")["max_new_tokens"] += 1
    with pytest.raises(ValueError):
        validate_manifest(seal_manifest(value))


def test_resealed_changed_source_rejected_against_git(reference):
    value = copy.deepcopy(reference)
    value["source"]["assets"][0]["sha256"] = "0" * 64
    value["source"]["source_sha256"] = source_sha256(value["source"]["assets"], value["source"]["gitlinks"])
    with pytest.raises(ValueError, match="committed source"):
        validate_manifest(seal_manifest(value), ROOT)


def test_baseline_hash_changes_invalidate_rank_evidence(reference):
    evidence = evidence_for(reference)
    assert validate_evidence(evidence, reference) is evidence
    evidence["rank_receipts"][2]["arithmetic_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="numerical baseline"):
        validate_evidence(evidence, reference)


@pytest.mark.parametrize("change", ["missing", "duplicate", "bool", "pid", "namespace", "source", "checkpoint"])
def test_rank_receipts_require_all_ranks_and_identical_identity(reference, change):
    evidence = evidence_for(reference)
    ranks = evidence["rank_receipts"]
    if change == "missing":
        ranks.pop()
    elif change == "duplicate":
        ranks[-1] = ranks[0]
    elif change == "bool":
        ranks[0]["rank"] = False
    elif change == "pid":
        ranks[0]["pid"] = 0
    elif change == "namespace":
        ranks[0]["execution_namespace"] = "wrong"
    elif change == "source":
        ranks[0]["source_sha256"] = "0" * 64
    elif change == "checkpoint":
        evidence["checkpoint_sha256"] = "0" * 64
    with pytest.raises(ValueError):
        validate_evidence(evidence, reference)


@pytest.mark.parametrize("cache,cached", [("cold", 1), ("warm", 0)])
def test_cached_work_cannot_be_mislabeled(reference, cache, cached):
    workload = next(w for w in reference["workloads"] if w["cache_policy"] == cache)
    evidence = evidence_for(reference)
    fields = (
        "token_ids_sha256",
        "conversation_sha256",
        "image_sha256",
        "max_new_tokens",
        "mtp_tokens",
        "submitted_requests",
        "rendered_request_sha256",
    )
    evidence["trials"] = [
        {
            "workload_id": workload["id"],
            "cache_policy": cache,
            "computed_prompt_tokens": 128,
            "cached_prompt_tokens": cached,
            **{k: workload[k] for k in fields},
        }
    ]
    with pytest.raises(ValueError, match="mislabeled|prefix reuse"):
        validate_evidence(evidence, reference)


def test_no_offline_promotion(reference):
    with pytest.raises(ValueError, match="pending"):
        validate_evidence(evidence_for(reference), reference, require_live=True)


def test_cpu_pass_cannot_be_represented_as_hardware_gate(reference):
    evidence = evidence_for(reference)
    evidence["gates"]["kernel_parity"] = {"status": "passed", "scope": "cpu"}
    with pytest.raises(ValueError, match="hardware evidence"):
        validate_evidence(evidence, reference)


def test_image_and_tool_pending_inputs_are_explicit(reference):
    assert all(w["conversation"] == {"payload_ref": "tool"} for w in reference["workloads"] if w["kind"] == "tool")
    assert all(w["image_sha256"] is None for w in reference["workloads"] if w["kind"] == "image")
    assert reference["pending"]


def test_authored_payloads_are_frozen_without_inventing_token_counts(reference):
    payloads = reference["request_payloads"]
    for workload in reference["workloads"]:
        payload = payloads[workload["request_payload_ref"]]
        assert workload["conversation_sha256"] == canonical_sha256(payload["conversation"])
        assert workload["rendered_request_sha256"] == canonical_sha256(payload["request"])
        assert payload["request"]["model"] is None
        assert payload["tokenizer"]["id"] is None
        assert workload["actual_prompt_tokens"] is None
        assert workload["token_ids"] is None
    assert "Note 200:" in payloads["23k"]["conversation"][-1]["content"]
    assert "Note 640:" in payloads["long"]["conversation"][-1]["content"]
    assert payloads["tool"]["request"]["tools"][0]["function"]["name"] == "read_temperature_fixture"
    assert payloads["tool"]["tool_response_fixture"]["temperatures_c"] == [81, 85, 94, 83]
    assert payloads["mixed"]["arrival_schedule"]["trigger"] == {"event": "generated_token", "index": 16}
    assert payloads["image_fresh"]["request"]["messages"][-1]["content"][-1]["image_url"]["url"] is None


@pytest.mark.parametrize("change", ["conversation", "rendering", "reference", "trial"])
def test_changed_rendered_inputs_invalidate_comparisons(reference, change):
    manifest = copy.deepcopy(reference)
    if change == "conversation":
        manifest["request_payloads"]["short"]["conversation"][0]["content"] = "changed"
    elif change == "rendering":
        manifest["request_payloads"]["short"]["request"]["temperature"] = 1
    elif change == "reference":
        manifest["workloads"][0]["request_payload_ref"] = "tool"
    else:
        evidence = evidence_for(manifest)
        workload = manifest["workloads"][0]
        evidence["trials"] = [
            {
                "workload_id": workload["id"],
                "cache_policy": "cold",
                "computed_prompt_tokens": 128,
                "cached_prompt_tokens": 0,
                **{
                    k: workload[k]
                    for k in (
                        "token_ids_sha256",
                        "conversation_sha256",
                        "image_sha256",
                        "max_new_tokens",
                        "mtp_tokens",
                        "submitted_requests",
                    )
                },
                "rendered_request_sha256": "0" * 64,
            }
        ]
        with pytest.raises(ValueError, match="rendered request"):
            validate_evidence(evidence, manifest)
        return
    with pytest.raises(ValueError):
        validate_manifest(seal_manifest(manifest))


def test_previous_append_only_reference_remains_valid():
    old = json.loads((ROOT / "artifacts/qwen38-streaming-upgrade/T1/reference.json").read_text())
    assert validate_manifest(old) is old


@pytest.mark.parametrize("value", [None, [], {}, {"schema_version": True}])
def test_malformed_inputs_fail_closed(reference, value):
    with pytest.raises(ValueError):
        validate_manifest(value)
    with pytest.raises(ValueError):
        validate_evidence(value, reference)


def test_null_pending_identity_key_is_still_required(reference):
    evidence = evidence_for(reference)
    del evidence["rank_receipts"][0]["checkpoint_sha256"]
    with pytest.raises(ValueError, match="baseline mismatch"):
        validate_evidence(evidence, reference)


def test_candidate_source_or_build_changes_invalidate_binding(reference):
    manifest = copy.deepcopy(reference)
    candidate = {
        "assets": [{"path": "source.py", "sha256": "1" * 64}],
        "binaries": [{"path": "kernel.bin", "sha256": "2" * 64}],
        "bridges": [{"path": "bridge.so", "sha256": "3" * 64}],
        "contract_sha256": "4" * 64,
        "native_components": ["streaming_experts"],
    }
    candidate["candidate_sha256"] = canonical_sha256(candidate)
    manifest["candidate"] = candidate
    manifest = seal_manifest(manifest)
    assert validate_manifest(manifest) is manifest
    evidence = evidence_for(manifest)
    assert validate_evidence(evidence, manifest) is evidence
    candidate["binaries"][0]["sha256"] = "5" * 64
    with pytest.raises(ValueError, match="candidate identity"):
        validate_manifest(seal_manifest(manifest))


def test_gate_receipt_cannot_omit_rank_or_manifest_binding(reference):
    evidence = evidence_for(reference)
    evidence["gates"]["real_weights"] = {"status": "passed", "scope": "hardware", "artifact_sha256": "1" * 64}
    with pytest.raises(ValueError, match="coverage mismatch"):
        validate_evidence(evidence, reference)
