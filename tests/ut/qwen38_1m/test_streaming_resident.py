# SPDX-License-Identifier: Apache-2.0
"""Synthetic all-rank evidence and fake RPC transactions; no hardware claims."""

import json
import subprocess
from dataclasses import asdict
from pathlib import Path

import pytest

from tools.qwen4exp import build_streaming as build
from tools.qwen4exp import streaming_resident as resident
from tools.qwen4exp.streaming_memory import RANK_COMPONENTS
from tools.qwen4exp.streaming_protocol import (
    REQUIRED_GATES,
    canonical_sha256,
    file_sha256,
    seal_manifest,
    source_sha256,
)
from tools.qwen4exp.streaming_schedule import SchedulePlan, SchedulePolicy

ROOT = Path(__file__).resolve().parents[3]


def write_json(path, value):
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")


@pytest.fixture
def fixture(tmp_path, request):
    """Create explicitly synthetic bytes/receipts, never compile/load a device binary."""
    reference = json.loads((ROOT / "artifacts/qwen38-streaming-upgrade/T1/reference-v2.json").read_text())
    git_root = tmp_path / "baseline"
    git_root.mkdir()
    (git_root / "baseline.py").write_text("# Synthetic baseline source\n")
    for args in (
        ["init", "-q"],
        ["add", "baseline.py"],
        ["-c", "user.name=Fixture", "-c", "user.email=fixture@example.test", "commit", "-qm", "fixture"],
    ):
        subprocess.run(["git", "-C", str(git_root), *args], check=True, capture_output=True)
    external_mode = getattr(request, "param", None)
    external_inventory = []
    gitlinks = []
    external_base = tmp_path / "external"
    if external_mode:
        external_repo = external_base / "lib"
        external_repo.mkdir(parents=True)
        (external_repo / "header.h").write_text("// Synthetic pinned external source\n")
        for args in (["init", "-q"], ["add", "header.h"]):
            subprocess.run(["git", "-C", str(external_repo), *args], check=True, capture_output=True)
        if external_mode == "external_nested":
            # A nested Git link is intentionally absent from the flat byte
            # inventory; admission must reject it instead of silently skipping.
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(external_repo),
                    "update-index",
                    "--add",
                    "--cacheinfo",
                    "160000",
                    subprocess.run(
                        ["git", "-C", str(git_root), "rev-parse", "HEAD"], check=True, capture_output=True, text=True
                    ).stdout.strip(),
                    "nested",
                ],
                check=True,
                capture_output=True,
            )
        subprocess.run(
            [
                "git",
                "-C",
                str(external_repo),
                "-c",
                "user.name=Fixture",
                "-c",
                "user.email=fixture@example.test",
                "commit",
                "-qm",
                "external fixture",
            ],
            check=True,
            capture_output=True,
        )
        external_commit = subprocess.run(
            ["git", "-C", str(external_repo), "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip()
        subprocess.run(
            ["git", "-C", str(git_root), "update-index", "--add", "--cacheinfo", "160000", external_commit, "lib"],
            check=True,
            capture_output=True,
        )
        subprocess.run(
            [
                "git",
                "-C",
                str(git_root),
                "-c",
                "user.name=Fixture",
                "-c",
                "user.email=fixture@example.test",
                "commit",
                "-qm",
                "external link",
            ],
            check=True,
            capture_output=True,
        )
        gitlinks = [{"path": "lib", "commit": external_commit, "content_status": "pending_external_bytes"}]
        external_files = [{"path": "header.h", "sha256": file_sha256(external_repo / "header.h")}]
        external_inventory = [
            {
                "path": "lib",
                "commit": external_commit,
                "files": external_files,
                "sha256": canonical_sha256(external_files),
            }
        ]
    commit = subprocess.run(
        ["git", "-C", str(git_root), "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    assets = [{"path": "baseline.py", "sha256": file_sha256(git_root / "baseline.py")}]
    reference["source"].update(
        commit=commit, assets=assets, gitlinks=gitlinks, source_sha256=source_sha256(assets, gitlinks)
    )
    data = tmp_path / "evidence"
    data.mkdir()
    for name in ("checkpoint.bin", "serving.bin", "image.png", "gate.json"):
        (data / name).write_bytes(b"synthetic host fixture only: " + name.encode())
    for name, filename in (("checkpoint", "checkpoint.bin"), ("serving_bundle", "serving.bin")):
        files = [{"path": filename, "sha256": file_sha256(data / filename)}]
        reference[name].update(files=files, sha256=canonical_sha256(files))
    for payload in reference["request_payloads"].values():
        payload["request"]["model"] = "synthetic_fixture"
        payload["tokenizer"].update(id="synthetic_fixture", revision="synthetic_revision")
        payload["rendered_request_sha256"] = canonical_sha256(payload["request"])
    for workload in reference["workloads"]:
        workload.update(
            token_ids=[1], token_ids_sha256=canonical_sha256([1]), actual_prompt_tokens=1, input_status="frozen"
        )
        workload["rendered_request_sha256"] = reference["request_payloads"][workload["request_payload_ref"]][
            "rendered_request_sha256"
        ]
        if workload["kind"] == "image":
            workload.update(image_content_path="image.png", image_sha256=file_sha256(data / "image.png"))
    reference = seal_manifest(reference)
    plan = SchedulePlan(
        SchedulePolicy(),
        1,
        reference["source"]["source_sha256"],
        reference["arithmetic_sha256"],
        reference["manifest_sha256"],
    )
    configuration = {"schedule": asdict(plan), "synthetic_fixture": True}
    base_path = tmp_path / "base.json"
    write_json(base_path, reference)
    bundle = tmp_path / "bundle"
    prepared = build.prepare_bundle(bundle, ROOT, 1, base_path, configuration)
    binaries = []
    (bundle / "binaries").mkdir()
    (bundle / "bridge").mkdir()
    for name, entries in build.KERNELS:
        path = bundle / "binaries" / f"{name}.bin"
        path.write_bytes(b"synthetic bytes; not executable: " + name.encode())
        binaries.append(
            {"path": path.relative_to(bundle).as_posix(), "sha256": file_sha256(path), "entrypoints": list(entries)}
        )
    bridge = bundle / "bridge/qwen_streaming_v1.so"
    bridge.write_bytes(b"synthetic bridge bytes, never loaded")
    candidate = {
        key: prepared[key]
        for key in (
            "namespace",
            "assets",
            "base_reference_sha256",
            "configuration_sha256",
            "configuration",
            "contract_sha256",
        )
    }
    candidate.update(
        binaries=sorted(binaries, key=lambda entry: entry["path"]),
        bridges=[{"path": bridge.relative_to(bundle).as_posix(), "sha256": file_sha256(bridge)}],
        native_components=[name for name, _ in build.KERNELS],
    )
    candidate["resources_sha256"] = resident.resource_inventory_sha256(candidate)
    (bundle / "build").mkdir()
    helper = bundle / "build/host-compile-sandbox"
    helper.write_text("synthetic containment helper; never executed\n")
    helper_source = next(
        entry for entry in prepared["assets"] if entry["path"] == "sources/tools/qwen4exp/host_compile_sandbox.cpp"
    )
    containment = build._containment_policy(
        bundle,
        {"cann": Path("/usr"), "npu": Path("/usr"), "torch_root": Path("/usr")},
        helper,
        6,
        json.dumps({"landlock_abi": 6, "probe": "query_only"}),
        helper_source["sha256"],
    )
    attestations = []
    for index in range(len(build.KERNELS) + 2):
        attestation = {
            "containment": "landlock_seccomp",
            "landlock_abi": 6,
            "restricted": True,
            "nonstandard_fds_closed": True,
            "driver_ioctls_and_sockets_denied": True,
        }
        stdout, stderr = bundle / f"build/command-{index}.stdout", bundle / f"build/command-{index}.stderr"
        stdout.write_text("synthetic fixture only\n")
        stderr.write_text(json.dumps(attestation) + "\n")
        command = [f"synthetic-command-{index}-never-executed"]
        attestations.append(
            {
                "command": command,
                "command_sha256": canonical_sha256(command),
                "prefix_sha256": canonical_sha256(containment["command_prefix"]),
                "attestation": attestation,
                "stdout": {"path": stdout.relative_to(bundle).as_posix(), "sha256": file_sha256(stdout)},
                "stderr": {"path": stderr.relative_to(bundle).as_posix(), "sha256": file_sha256(stderr)},
            }
        )
    compile_receipt = seal_manifest(
        {
            "kind": "host_only_compile_receipt",
            "target": "dav-2002",
            "prepared_manifest_sha256": prepared["manifest_sha256"],
            "resources_sha256": candidate["resources_sha256"],
            "npu_opened": False,
            "library_loaded": False,
            "kernels_launched": False,
            "server_contacted": False,
            "synthetic_fixture": True,
            "filesystem_containment": containment,
            "command_attestations": attestations,
        }
    )
    write_json(bundle / "build-receipt.json", compile_receipt)
    candidate["build_receipt"] = {"path": "build-receipt.json", "sha256": file_sha256(bundle / "build-receipt.json")}
    candidate["candidate_sha256"] = canonical_sha256(candidate)
    write_json(bundle / "candidate.json", candidate)
    reference["candidate"] = candidate
    reference = seal_manifest(reference)
    identities = {
        "manifest_sha256": reference["manifest_sha256"],
        "source_sha256": reference["source"]["source_sha256"],
        "arithmetic_sha256": reference["arithmetic_sha256"],
        "checkpoint_sha256": reference["checkpoint"]["sha256"],
        "serving_bundle_sha256": reference["serving_bundle"]["sha256"],
        "candidate_sha256": candidate["candidate_sha256"],
    }
    ranks = [
        dict(
            identities,
            rank=rank,
            pid=100 + rank,
            generation=0,
            execution_namespace="qwen_streaming_v1",
            weight_storage_digest=str(rank + 1) * 64,
        )
        for rank in range(4)
    ]
    artifact = {"artifact_path": "gate.json", "artifact_sha256": file_sha256(data / "gate.json")}
    gates = {
        name: dict(
            artifact,
            status="passed",
            scope="hardware",
            ranks=[0, 1, 2, 3],
            manifest_sha256=reference["manifest_sha256"],
        )
        for name in REQUIRED_GATES
    }
    trials = []
    for workload in reference["workloads"]:
        trial = {
            key: workload[key]
            for key in (
                "cache_policy",
                "token_ids_sha256",
                "conversation_sha256",
                "image_sha256",
                "max_new_tokens",
                "mtp_tokens",
                "submitted_requests",
                "rendered_request_sha256",
            )
        }
        warm = workload["cache_policy"] == "warm"
        trial.update(
            artifact,
            workload_id=workload["id"],
            computed_prompt_tokens=0 if warm else 1,
            cached_prompt_tokens=1 if warm else 0,
            hardware_validated=True,
        )
        trials.append(trial)
    evidence = dict(
        identities,
        schema_version=1,
        rank_receipts=ranks,
        gates=gates,
        trials=trials,
        engine_id="synthetic_engine",
        native_component_evidence={
            name: dict(
                artifact,
                status="passed",
                scope="hardware",
                ranks=[0, 1, 2, 3],
                candidate_sha256=candidate["candidate_sha256"],
            )
            for name in candidate["native_components"]
        },
        external_source_inventory=external_inventory,
    )
    plan_receipts = [
        dict(
            rank=rank,
            pid=100 + rank,
            generation=1,
            previous_generation=0,
            execution_namespace="qwen_streaming_v1",
            weight_storage_digest=str(rank + 1) * 64,
            plan_sha256=plan.sha256,
            candidate_sha256=candidate["candidate_sha256"],
        )
        for rank in range(4)
    ]
    components = dict.fromkeys(RANK_COMPONENTS, 1024)
    components.update(resident.minimum_workspaces(plan))
    memory = [
        dict(
            artifact,
            rank=rank,
            pid=100 + rank,
            generation=1,
            plan_sha256=plan.sha256,
            candidate_sha256=candidate["candidate_sha256"],
            weight_storage_digest=str(rank + 1) * 64,
            available_bytes=2**30,
            components=dict(components),
            backend_scratch_resolved=True,
            guard_bytes=4096,
        )
        for rank in range(4)
    ]
    if external_mode == "external_changed":
        (external_base / "lib/header.h").write_text("// Changed since pinned commit\n")
    return {
        "manifest": reference,
        "evidence": evidence,
        "kwargs": {
            "bundle_root": bundle,
            "reference_git_root": git_root,
            "artifact_roots": {
                "checkpoint": data,
                "serving_bundle": data,
                "hardware_evidence": data,
                "external_source": external_base,
            },
            "plan": plan,
            "plan_receipts": plan_receipts,
            "memory_receipts": memory,
            "configuration_sha256": candidate["configuration_sha256"],
            "resources_sha256": candidate["resources_sha256"],
        },
    }


def admission(fixture):
    return resident.admit(fixture["manifest"], fixture["evidence"], **fixture["kwargs"])


def payload(approved, owner="test"):
    return {
        "transaction_id": owner,
        "admission_sha256": approved.seal,
        "candidate_sha256": approved.candidate_sha256,
        "configuration_sha256": approved.configuration_sha256,
        "resources_sha256": approved.resources_sha256,
        "plan_sha256": approved.plan_sha256,
        "generation": approved.generation,
    }


class FakeRPC:
    """Independent four-worker installer and explicit authoritative fake lease."""

    def __init__(self, approved, journal):
        self.approved, self.journal = approved, journal
        self.calls, self.loaded, self.installed = [], [], [False] * 4
        self.fault = None
        self.scheduler = dict(
            engine_id=approved.engine_id,
            running=0,
            waiting=0,
            paused=False,
            maintenance_owner=None,
            outstanding_collective=False,
            thermal_hold=False,
            temperatures_c=[70] * 4,
        )
        self.workers = []
        for rank, identity in enumerate(approved.baseline_identities):
            snapshot = dict(
                zip(resident.IDENTITY_FIELDS, identity),
                graphs_dirty=False,
                native_failed=False,
                outstanding_collective=False,
                pending_execution=False,
            )

            def prepare(control, rank=rank):
                assert not self.loaded or rank not in self.loaded
                return resident.WorkerPreparation(approved, rank, rank)

            def apply(token):
                self.loaded.append(token)
                self.installed[token] = True
                if self.fault == "worker_apply" and token == 1:
                    raise RuntimeError("synthetic partial installer failure")

            def restore(token):
                if self.fault == "restore":
                    raise RuntimeError("synthetic restore failure")
                self.installed[token] = False

            self.workers.append(
                resident.WorkerTransaction(
                    snapshot=lambda snapshot=snapshot: dict(
                        snapshot, paused=self.scheduler["paused"], maintenance_owner=self.scheduler["maintenance_owner"]
                    ),
                    prepare=prepare,
                    apply=apply,
                    restore=restore,
                )
            )

    def request(self, path, control=None, method="POST"):
        self.calls.append((path, control))
        if path.startswith("/pause"):
            assert list(self.journal.glob("*-intent.json")), "pause preceded durable intent"
            assert "clear_cache=false" in path
            self.scheduler.update(paused=True, maintenance_owner=control["owner"])
            return {"status": "paused"}
        if path == "/resume":
            self.scheduler.update(paused=False, maintenance_owner=None)
            return {"status": "resumed"}
        assert path == "/collective_rpc"
        name, args = control["method"], control["args"]
        if self.fault == name + "_timeout":
            raise TimeoutError("synthetic unresolved RPC")
        if self.fault == name + "_outstanding":
            return {"outstanding_collective": True, "collective_completed": False, "results": []}
        results = []
        for worker in self.workers:
            try:
                if name == "streaming_status":
                    result = worker.status()
                elif name == "streaming_prepare":
                    result = worker.prepare(json.loads(args[0]))
                elif name == "streaming_apply":
                    result = worker.apply(args[0])
                elif name == "streaming_restore":
                    result = worker.restore(args[0])
                else:
                    raise AssertionError(name)
            except Exception as error:
                result = dict(worker.baseline, error=str(error))
            results.append(result)
        if self.fault == name + "_partial":
            results.pop()
        if self.fault == "stale" and name == "streaming_status":
            results[0]["generation"] += 1
        return {"outstanding_collective": False, "collective_completed": True, "results": results}


def controller(tmp_path, approved):
    journal = tmp_path / "journal"
    fake = FakeRPC(approved, journal)
    control = resident.ResidentController(
        fake, journal, "test", live=True, scheduler_state=lambda: dict(fake.scheduler)
    )
    return control, fake


def test_complete_synthetic_admission_and_immutable_decoded_receipts(fixture):
    approved = admission(fixture)
    assert approved.reference_sha256 == fixture["kwargs"]["plan"].reference_sha256
    copy_receipts = approved.plan_receipts
    copy_receipts[0]["pid"] = 999
    assert approved.plan_receipts[0]["pid"] == 100
    approved.require_execution(
        configuration_sha256=approved.configuration_sha256, resources_sha256=approved.resources_sha256
    )


@pytest.mark.parametrize("fixture", ["external_valid", "external_changed", "external_nested"], indirect=True)
def test_expanded_external_git_bytes_and_nested_coverage(fixture):
    record = fixture["evidence"]["external_source_inventory"][0]
    external_path = fixture["kwargs"]["artifact_roots"]["external_source"] / record["path"]
    header = external_path / "header.h"
    tree = subprocess.run(
        ["git", "-C", str(external_path), "ls-tree", "-r", record["commit"]], check=True, capture_output=True, text=True
    ).stdout
    if "nested" in tree:
        with pytest.raises(ValueError, match="nested external Git"):
            admission(fixture)
    elif file_sha256(header) != record["files"][0]["sha256"]:
        with pytest.raises(ValueError, match="actual admitted file bytes differ"):
            admission(fixture)
    else:
        approved = admission(fixture)
        assert (str(header), file_sha256(header)) in approved.file_bindings


def test_offline_default_calls_no_rpc(tmp_path, fixture):
    approved = admission(fixture)
    control, fake = controller(tmp_path, approved)
    control.live = False
    assert control.execute(approved, payload(approved))["rpc_calls"] == 0
    assert fake.calls == [] and not (tmp_path / "journal").exists()


def test_idle_success_intent_precedes_mutation_and_prepare_does_not_load(tmp_path, fixture):
    approved = admission(fixture)
    control, fake = controller(tmp_path, approved)
    assert control.execute(approved, payload(approved))["state"] == "COMPLETE"
    assert fake.installed == [True] * 4 and fake.loaded == [0, 1, 2, 3]
    assert not fake.scheduler["paused"]
    phases = [json.loads(path.read_text())["phase"] for path in sorted((tmp_path / "journal").glob("*.json"))]
    assert phases == ["intent", "paused", "apply-intent", "applied", "resume-intent", "complete"]


def test_direct_worker_rpc_cannot_bypass_owned_maintenance(tmp_path, fixture):
    approved = admission(fixture)
    control, fake = controller(tmp_path, approved)
    with pytest.raises(resident.HeldTransaction, match="maintenance lease"):
        fake.workers[0].prepare(payload(approved))
    assert fake.loaded == []
    control.execute(approved, payload(approved))
    with pytest.raises(resident.HeldTransaction, match="maintenance lease"):
        fake.workers[0].restore("test")
    assert fake.installed == [True] * 4


def test_containment_helper_bytes_remain_bound_after_admission(fixture):
    approved = admission(fixture)
    helper = fixture["kwargs"]["bundle_root"] / "build/host-compile-sandbox"
    helper.write_text("changed helper after admission")
    with pytest.raises(ValueError, match="bytes changed"):
        approved.require_execution(
            configuration_sha256=approved.configuration_sha256, resources_sha256=approved.resources_sha256
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("running", 1),
        ("waiting", 1),
        ("paused", True),
        ("maintenance_owner", "other"),
        ("thermal_hold", True),
        ("temperatures_c", [94, 70, 70, 70]),
        ("temperatures_c", [70, 70, 70]),
    ],
)
def test_nonidle_foreign_hold_and_thermal_missing_sensors_leave_server_untouched(tmp_path, fixture, field, value):
    approved = admission(fixture)
    control, fake = controller(tmp_path, approved)
    fake.scheduler[field] = value
    with pytest.raises(resident.HeldTransaction):
        control.execute(approved, payload(approved))
    assert fake.calls == []


@pytest.mark.parametrize("fault", ["streaming_status_partial", "stale"])
def test_preflight_missing_ranks_or_stale_generation_do_not_pause(tmp_path, fixture, fault):
    approved = admission(fixture)
    control, fake = controller(tmp_path, approved)
    fake.fault = fault
    with pytest.raises((resident.HeldTransaction, ValueError)):
        control.execute(approved, payload(approved))
    assert len(fake.calls) == 1 and not fake.scheduler["paused"]


@pytest.mark.parametrize(
    "fault", ["streaming_apply_timeout", "streaming_apply_outstanding", "streaming_prepare_timeout"]
)
def test_uncertain_collective_halts_without_followup_status_restore_or_resume(tmp_path, fixture, fault):
    approved = admission(fixture)
    control, fake = controller(tmp_path, approved)
    fake.fault = fault
    with pytest.raises(resident.HaltedTransaction):
        control.execute(approved, payload(approved))
    before = list(fake.calls)
    with pytest.raises(resident.HaltedTransaction):
        control.recover()
    assert fake.calls == before and fake.scheduler["paused"] and control.owns_pause


@pytest.mark.parametrize("fault", ["streaming_apply_partial", "worker_apply"])
def test_known_complete_partial_mutation_retains_hold_and_explicitly_recovers(tmp_path, fixture, fault):
    approved = admission(fixture)
    control, fake = controller(tmp_path, approved)
    fake.fault = fault
    with pytest.raises(resident.HeldTransaction):
        control.execute(approved, payload(approved))
    assert fake.scheduler["paused"] and control.state == "HELD"
    assert not any(path == "/resume" for path, _ in fake.calls)
    fake.fault = None
    control.recover()
    assert fake.installed == [False] * 4 and control.state == "COMPLETE"


def test_recovery_failure_halts_and_stops_all_further_rpc(tmp_path, fixture):
    approved = admission(fixture)
    control, fake = controller(tmp_path, approved)
    fake.fault = "streaming_apply_partial"
    with pytest.raises(resident.HeldTransaction):
        control.execute(approved, payload(approved))
    fake.fault = "restore"
    with pytest.raises(resident.HaltedTransaction):
        control.recover()
    before = list(fake.calls)
    with pytest.raises(resident.HaltedTransaction):
        control.recover()
    assert fake.calls == before and fake.scheduler["paused"]


def test_failure_before_apply_does_not_restore_worker_dispatch(tmp_path, fixture):
    approved = admission(fixture)
    control, fake = controller(tmp_path, approved)
    fake.fault = "streaming_prepare_partial"
    with pytest.raises(resident.HeldTransaction):
        control.execute(approved, payload(approved))
    assert not control.may_have_mutated
    fake.fault = None
    control.recover()
    assert not any(body and body.get("method") == "streaming_restore" for _, body in fake.calls)


@pytest.mark.parametrize(
    "field",
    [
        "source_bytes",
        "gate_bytes",
        "build_receipt",
        "memory_unknown",
        "memory_zero",
        "scratch",
        "rank_pid",
        "plan_generation",
        "gate_pending",
    ],
)
def test_admission_rejects_changed_bytes_pending_gates_or_unresolved_budgets(fixture, field):
    if field == "source_bytes":
        entry = fixture["manifest"]["candidate"]["assets"][0]
        (fixture["kwargs"]["bundle_root"] / entry["path"]).write_text("changed")
    elif field == "gate_bytes":
        (fixture["kwargs"]["artifact_roots"]["hardware_evidence"] / "gate.json").write_text("changed")
    elif field == "build_receipt":
        (fixture["kwargs"]["bundle_root"] / "build-receipt.json").write_text("changed")
    elif field == "memory_unknown":
        fixture["kwargs"]["memory_receipts"][0]["components"]["hccl"] = None
    elif field == "memory_zero":
        fixture["kwargs"]["memory_receipts"][0]["components"]["route_workspace"] = 0
    elif field == "scratch":
        fixture["kwargs"]["memory_receipts"][0]["backend_scratch_resolved"] = False
    elif field == "rank_pid":
        fixture["kwargs"]["plan_receipts"][0]["pid"] += 1
    elif field == "plan_generation":
        fixture["kwargs"]["plan_receipts"][0]["generation"] += 1
    else:
        fixture["evidence"]["gates"]["quality"]["status"] = "pending"
    with pytest.raises((ValueError, json.JSONDecodeError)):
        admission(fixture)


def test_execution_rehashes_actual_bytes_after_admission(fixture):
    approved = admission(fixture)
    path, _ = approved.file_bindings[-1]
    Path(path).write_text("changed after admission")
    with pytest.raises(ValueError, match="bytes changed"):
        approved.require_execution(
            configuration_sha256=approved.configuration_sha256, resources_sha256=approved.resources_sha256
        )


def test_payload_identity_mismatch_never_contacts_server(tmp_path, fixture):
    approved = admission(fixture)
    control, fake = controller(tmp_path, approved)
    wrong = payload(approved)
    wrong["resources_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="payload"):
        control.execute(approved, wrong)
    assert fake.calls == []


def test_out_of_band_completion_requires_actual_artifact_and_owned_rank_identity(tmp_path, fixture):
    approved = admission(fixture)
    control, fake = controller(tmp_path, approved)
    fake.fault = "streaming_apply_timeout"
    with pytest.raises(resident.HaltedTransaction):
        control.execute(approved, payload(approved))
    body = {
        "transaction_id": "test",
        "engine_id": approved.engine_id,
        "collectives_completed": True,
        "paused": True,
        "maintenance_owner": "test",
        "rank_receipts": [dict(zip(resident.IDENTITY_FIELDS, identity)) for identity in approved.baseline_identities],
    }
    path = tmp_path / "external.json"
    write_json(path, body)
    proof = dict(body, artifact_sha256=file_sha256(path))
    before = list(fake.calls)
    control.acknowledge_external_completion(proof, proof_path=path)
    assert fake.calls == before and control.state == "HELD"
    fake.fault = None
    # Timeout happened before fake worker installation; restore rejects absent
    # preparations only if they were never prepared. Here all prepared earlier.
    control.recover()
    assert control.state == "COMPLETE"


def test_existing_journal_intent_cannot_be_overwritten_before_pause(tmp_path, fixture):
    approved = admission(fixture)
    control, fake = controller(tmp_path, approved)
    control.journal_dir.mkdir()
    (control.journal_dir / "test-000-intent.json").write_text("original immutable intent")
    with pytest.raises(FileExistsError):
        control.execute(approved, payload(approved))
    assert not fake.scheduler["paused"]
    assert (control.journal_dir / "test-000-intent.json").read_text() == "original immutable intent"
