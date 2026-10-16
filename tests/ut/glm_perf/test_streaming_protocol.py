# SPDX-License-Identifier: Apache-2.0
"""Offline protocol tests: no server mutation, NPU allocations or dependencies."""

import ast
import base64
import copy
import hashlib
import json

import pytest

from tools.glm_perf.streaming_protocol import (
    REMOTE_COLLECTOR,
    canonical_digest,
    collect_reference,
    evidence_schema,
    freeze_reference,
    make_workloads,
    native_case_ids,
    shadow_case_ids,
    summarize_trials,
    validate_evidence,
    verify_reference,
)

DIGEST = "a" * 64


def evidence(kind="native_gates"):
    schema = evidence_schema()
    result = {rule: True for rule in schema["x-complete-result-rules"][kind]}
    if kind == "shadow":
        result["comparisons"] = 344
    return {
        "schema_version": 1,
        "kind": kind,
        "state": "complete",
        "identity": {
            "contract_sha256": DIGEST,
            "reference_sha256": DIGEST,
            "source_sha256": DIGEST,
            "binaries": {"gate.bin": DIGEST, "down.bin": DIGEST},
            "version": 1100,
            "namespace": "glm_reconstruction_v1100",
        },
        "dispatch": {
            "bulk_a4_gate": "streaming",
            "bulk_a4_down": "streaming",
            "sparse": "qualified_reference",
            "a8": "qualified_reference",
            "decode_version": 984,
        },
        "coverage": schema["x-required-coverage"][kind],
        "artifacts": [{"path": "result.json", "sha256": DIGEST}],
        "errors": [],
        "result": result,
    }


def trials(reference=50.0, candidate=44.0):
    return [
        {
            "cycle": cycle,
            "position": pos,
            "variant": variant,
            "ttft_s": reference if variant == "reference" else candidate,
            "complete": True,
            "reused_at_admission": 0,
            "timing_method": "streamed_token_event",
            "prompt_sha256": canonical_digest([cycle, 43]),
        }
        for cycle in range(3)
        for pos, variant in enumerate(("reference", "candidate", "candidate", "reference"))
    ]


@pytest.mark.parametrize("kind", ["build", "native_gates", "shadow", "trials"])
def test_complete_evidence_requires_exact_binding(kind):
    report = evidence(kind)
    assert validate_evidence(report, kind=kind, bindings=copy.deepcopy(report["identity"])) is report
    with pytest.raises(ValueError, match="authoritative"):
        validate_evidence(report, bindings={"source_sha256": "b" * 64})


@pytest.mark.parametrize(
    "mutation",
    [
        lambda r: r.update(schema_version=True),
        lambda r: r.update(unrecognized=True),
        lambda r: r["identity"].update(namespace="glm_reconstruction_v1099"),
        lambda r: r["identity"].update(binaries={}),
        lambda r: r["identity"].update(source_sha256="not a digest"),
        lambda r: r["dispatch"].update(bulk_a4_down="qualified_reference"),
        lambda r: r["dispatch"].update(decode_version=1021),
        lambda r: r["coverage"].pop(),
        lambda r: r["coverage"].append(r["coverage"][0]),
        lambda r: r["result"].update(finite=False),
        lambda r: r["errors"].append("device timeout"),
        lambda r: r.update(artifacts=[]),
    ],
)
def test_partial_or_incompatible_reports_rejected(mutation):
    report = evidence()
    mutation(report)
    with pytest.raises(ValueError):
        validate_evidence(report)


@pytest.mark.parametrize("state", ["pending", "failed", "blocked"])
def test_incomplete_evidence_is_never_live_admission(state):
    report = evidence()
    report.update(state=state, coverage=[])
    if state != "pending":
        report["errors"] = ["required hardware validation unavailable"]
    assert validate_evidence(report, require_complete=False) is report
    with pytest.raises(ValueError, match="not complete"):
        validate_evidence(report)


def test_shadow_counts_do_not_replace_coverage():
    report = evidence("shadow")
    report["coverage"][-1] = "rank0_bank0_chunk999"
    with pytest.raises(ValueError, match="missing case"):
        validate_evidence(report)
    report = evidence("shadow")
    report["result"]["comparisons"] = 343
    with pytest.raises(ValueError, match="344"):
        validate_evidence(report)


def test_artifact_identity_verified_with_path_confinement(tmp_path):
    path = tmp_path / "result.json"
    path.write_bytes(b"real evidence")
    report = evidence()
    report["artifacts"][0]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    validate_evidence(report, artifact_root=tmp_path)
    path.write_bytes(b"changed evidence")
    with pytest.raises(ValueError, match="changed"):
        validate_evidence(report, artifact_root=tmp_path)
    report["artifacts"][0]["path"] = "../result.json"
    with pytest.raises(ValueError, match="outside"):
        validate_evidence(report, artifact_root=tmp_path)


def test_required_workload_inputs_and_lengths_are_frozen():
    protocol = make_workloads()
    rows = {row["id"]: row for row in protocol["workloads"]}
    for tokens in (640, 6400, 32768):
        row = rows[f"cold_{tokens}"]
        assert len(row["prompt_token_ids"]) == tokens
        assert canonical_digest(row["prompt_token_ids"]) == row["prompt_sha256"]
        assert row["max_tokens"] == 1
    assert rows["shadow_1920"]["prompt_tokens"] == 1920
    assert rows["decode_c4"]["concurrency"] == 4
    assert rows["decode_c1"]["max_tokens"] == rows["decode_c4"]["max_tokens"] == 128
    assert rows["warm_prefix_6400"]["cold"] is False
    for name in ("language", "tool"):
        assert canonical_digest(rows[name]["input"]) == rows[name]["rendered_input_sha256"]
        assert rows[name]["materialize_token_ids_before_trials"] is True
    assert protocol["long_context"]["existing_context_depths"] == [0, 6400, 32768]
    assert "visible-text" in protocol["timing"]["first_visible_text"]
    assert protocol["warmup"]["clear_prefix_after_warmup"] is True
    assert protocol["promotion"]["not_a_significance_test"] is True


def test_required_native_and_shadow_shapes_cover_complete_protocol():
    ids = native_case_ids()
    assert len(ids) == len(set(ids)) == 438
    for bits in (2, 3, 4):
        for activation in (4, 8):
            for k in (2048, 4096):
                for rows in (0, 1, 4, 5, 15, 16, 17, 30, 31, 32, 62, 63):
                    assert f"w{bits}_a{activation}_k{k}_rows{rows}_all_peer" in ids
    assert len(shadow_case_ids()) == 344


def test_repeatability_uses_medians_and_is_distinct_from_objective():
    summary = summarize_trials(trials(candidate=48.0))
    assert summary["repeatability_passed"] is True
    assert summary["objective_10pct_met"] is False
    assert summary["minimum_improvement"] == 0.02
    assert summary["repeat_spread"] == 0
    assert summary["variants"]["reference"]["median_s"] == 50


def test_repeat_spread_takes_worst_variant_linear_quartiles():
    samples = trials()
    candidates = [row for row in samples if row["variant"] == "candidate"]
    for row, duration in zip(candidates, [30, 32, 34, 36, 38, 40]):
        row["ttft_s"] = duration
    summary = summarize_trials(samples)
    assert summary["variants"]["candidate"]["p25_s"] == 32.5
    assert summary["variants"]["candidate"]["p75_s"] == 37.5
    assert summary["repeat_spread"] == pytest.approx(5 / 35)
    assert summary["minimum_improvement"] == pytest.approx(10 / 35)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda rows: rows.pop(),
        lambda rows: rows[0].update(complete=False),
        lambda rows: rows[0].update(reused_at_admission=640),
        lambda rows: rows[0].update(ttft_s=float("nan")),
        lambda rows: rows[0].update(ttft_s=0),
        lambda rows: rows[0].update(prompt_sha256="b" * 64),
        lambda rows: rows[0].update(timing_method="first_visible_text"),
        lambda rows: rows[0].update(timing_method="server_generated_token"),
        lambda rows: rows[0].update(position=3),
        lambda rows: rows[0].update(variant="candidate"),
    ],
)
def test_bad_measurements_cannot_be_promoted(mutation):
    rows = trials()
    mutation(rows)
    with pytest.raises(ValueError):
        summarize_trials(rows)


def test_canonical_hash_has_no_nan_and_is_independent_of_dict_order():
    assert canonical_digest({"a": 1, "b": 2}) == canonical_digest({"b": 2, "a": 1})
    with pytest.raises(ValueError):
        canonical_digest(float("inf"))


def test_remote_collector_contains_no_tensor_imports_or_mutation_queries():
    tree = ast.parse(REMOTE_COLLECTOR)
    imports = [
        alias.name for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom)) for alias in node.names
    ]
    assert not set(imports).intersection({"torch", "torch_npu", "vllm"})
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "query"
    ]
    assert {node.args[0].value for node in calls} == {
        "/health",
        "/is_paused",
        "/resident/maintenance",
        "/metrics",
        "/server_info",
        "/collective_rpc",
    }
    query_node = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "query")
    namespace = {"queries": [], "base": "http://localhost"}
    exec(compile(ast.Module(body=[query_node], type_ignores=[]), "collector_guard", "exec"), namespace)
    for path, payload in (("/pause", {}), ("/collective_rpc", {"method": "resident_apply", "args": []})):
        with pytest.raises(ValueError, match="read-only"):
            namespace["query"](path, payload)
    assert namespace["queries"] == []


@pytest.mark.parametrize("host", ["-oProxyCommand=oops", "host;touch /tmp/oops", "host\ncommand"])
def test_bad_ssh_host_rejected_before_subprocess(host, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("subprocess must not run")

    monkeypatch.setattr("subprocess.run", forbidden)
    with pytest.raises(ValueError, match="SSH host"):
        collect_reference(host)


def test_frozen_manifest_detects_changed_source_and_path_escape(tmp_path):
    source = tmp_path / "reference.py"
    source.write_bytes(b"qualified reference")
    reference = {
        "schema_version": 1,
        "frozen_files": [{"path": "reference.py", "sha256": hashlib.sha256(source.read_bytes()).hexdigest()}],
    }
    reference["reference_sha256"] = canonical_digest(reference)
    verify_reference(reference, tmp_path)
    source.write_bytes(b"edited reference")
    with pytest.raises(ValueError, match="changed"):
        verify_reference(reference, tmp_path)
    reference["schema_version"] = 2
    with pytest.raises(ValueError, match="digest"):
        verify_reference(reference, tmp_path)


def encoded(name, value):
    raw = value if isinstance(value, bytes) else json.dumps(value).encode()
    return {
        "remote_path": "/remote/" + name,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "base64": base64.b64encode(raw).decode(),
    }


def snapshot():
    source = encoded("reference.py", b"complete replacement source")
    provenance = {
        "_build": {"version": 1021, "namespace": "glm_reconstruction_v1021"},
        "_helpers": {},
        "glm_fused_gate_up.bin": {"binary_sha256": DIGEST},
    }
    workers = [
        {
            "rank": rank,
            "pid": 100 + rank,
            "generation": "unchanged",
            "candidate": "qualified_profiled_v1021",
            "digest": source["sha256"],
            "graphs_dirty": False,
            "native_failed": False,
            "weight_storage_digest": str(rank) * 64,
            "fp16_scale_storage": {
                "prefill_version": 1021,
                "decode_version": 984,
                "banks": 43,
                "layouts": ["fp16_storage_fp32_compute_v1"],
            },
            "native_launch_audit": {
                "operator": "glm_reconstruction_v1021.launch",
                "binaries": {"gate": {"path": "/remote/glm_fused_gate_up.bin", "sha256": DIGEST}},
            },
        }
        for rank in range(4)
    ]
    shadow = {
        "cases_per_rank": [
            {"rank": rank, "cases": [{"bank_pointer": 10000 * rank + bank, "bits": 2 + bank % 3} for bank in range(43)]}
            for rank in range(4)
        ]
    }
    argv = [
        "serve",
        "--max-model-len",
        "131072",
        "--max-num-batched-tokens",
        "1280",
        "--max-num-seqs",
        "4",
        "--tensor-parallel-size",
        "4",
        "--kv-cache-memory-bytes",
        "2415919104",
        "--gpu-memory-utilization",
        "0.92",
        "--speculative-config",
        '{"num_speculative_tokens":1}',
        "--compilation-config",
        '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[2,8]}',
        "--hf-overrides",
        "{}",
        "--served-model-name",
        "glm",
        "--mamba-cache-mode",
        "align",
        "--no-async-scheduling",
        "--enable-prefix-caching",
        "--enable-chunked-prefill",
    ]
    cp = {
        "native-layout.json": {
            "complete": True,
            "weight_scale_layout": "fp16_storage_fp32_compute_v1",
            "native_shards": [],
        },
        "model.safetensors.index.json": {"metadata": {"expert_scale_layout": "fp16_storage_fp32_compute_v1"}},
        "config.json": {},
    }
    cp_files = {key: encoded(key, value) for key, value in cp.items()}
    cp_files.update(
        {
            name: {"sha256": DIGEST, "bytes": 1, "remote_path": "/remote/" + name}
            for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja")
        }
    )
    return {
        "workers": workers,
        "processes": [{"pid": 42, "argv": argv}],
        "files": {
            "final-candidate-v1021.py": source,
            "build-down-group-v1021/provenance.json": encoded("provenance.json", provenance),
            "shadow-summary.json": encoded("shadow-summary.json", shadow),
            "shadow-worker-identities.json": encoded("identities.json", {"workers": workers}),
        },
        "binaries": {"glm_fused_gate_up.bin": DIGEST},
        "helper_hashes": {},
        "checkpoint_files": cp_files,
        "health": {"http_status": 200},
        "server_info": {"body": {}},
        "paused": {"body": {"is_paused": False}},
        "maintenance": {"body": {"maintenance": False}},
        "queries": [],
    }


def test_freeze_is_append_only_and_hashes_published_assets(tmp_path):
    output = tmp_path / "freeze"
    reference = freeze_reference(snapshot(), output, host="user@host")
    assert verify_reference(reference, output) is reference
    assert len(reference["bank_inventory"]) == 4
    assert (output / "evidence-schema.json").is_file()
    assert (output / "workloads.json").is_file()
    with pytest.raises(ValueError, match="overwrite"):
        freeze_reference(snapshot(), output, host="user@host")


@pytest.mark.parametrize(
    "mutation",
    [
        lambda s: s["workers"][0].update(digest="b" * 64),
        lambda s: s["workers"][0].update(graphs_dirty=True),
        lambda s: s["workers"][0].update(native_failed=True),
        lambda s: s["workers"][0]["fp16_scale_storage"].update(decode_version=1021),
        lambda s: s["workers"][0]["native_launch_audit"]["binaries"]["gate"].update(sha256="b" * 64),
        lambda s: s["binaries"].update({"glm_fused_gate_up.bin": "b" * 64}),
        lambda s: s["workers"][0].update(pid=999),
    ],
)
def test_freeze_rejects_changed_dispatch_storage_binary_before_writing(tmp_path, mutation):
    data = snapshot()
    # Historical identity records must not alias the newly read status.
    raw = json.loads(base64.b64decode(data["files"]["shadow-worker-identities.json"]["base64"]))
    assert raw["workers"][0]["pid"] == 100
    mutation(data)
    output = tmp_path / "freeze"
    with pytest.raises(ValueError):
        freeze_reference(data, output, host="user@host")
    assert not output.exists()
