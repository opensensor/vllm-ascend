# SPDX-License-Identifier: Apache-2.0
"""Offline-first evidence admission and an injectable resident transaction.

There is no network/device adapter or CLI here. The caller must explicitly opt
into live execution and provide scheduler ownership probes plus an RPC client.
Saved hashes bind bytes and claims; they do not authenticate hardware results.
Uncertain collective completion is terminal until an external proof is supplied.
"""

import hashlib
import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from threading import RLock

import regex as re

from tools.qwen4exp.build_streaming import verify_bundle
from tools.qwen4exp.streaming_memory import rank_envelope, route_workspace
from tools.qwen4exp.streaming_protocol import (
    canonical_sha256,
    file_sha256,
    validate_evidence,
    validate_manifest,
)
from tools.qwen4exp.streaming_schedule import validate_rank_plan

RANKS = (0, 1, 2, 3)
HOLD_C = 94
RESUME_C = 85
IDENTITY_FIELDS = ("rank", "pid", "generation", "weight_storage_digest", "execution_namespace")


def _hash(value):
    return isinstance(value, str) and bool(re.fullmatch(r"[0-9a-f]{64}", value))


def resource_inventory_sha256(candidate):
    """Inventory identity without a self-referential candidate digest."""
    return canonical_sha256({key: candidate[key] for key in ("namespace", "binaries", "bridges")})


def minimum_workspaces(plan):
    """Conservative logical allocation bounds, never bus/performance estimates.

    A full output and routed input survive the complete call; local/shared/add,
    both local slots and out-of-place reduction outputs may coexist. The full
    routed MoE baseline is retained even when windows remove an intermediate.
    Additional builtin/operator scratch needs its own resolved budget receipt.
    """
    hidden, top_k, fp32_bytes = 2560, 10, 4
    chunk = min(plan.policy.chunk_tokens, plan.max_input_tokens)
    slots = plan.policy.max_inflight
    activation = (
        plan.max_input_tokens * hidden * fp32_bytes
        + (2 * slots + 3) * chunk * hidden * fp32_bytes
        + plan.max_input_tokens * top_k * (4 + 8)
    )
    return {"route_workspace": route_workspace(chunk, top_k)["total_bytes"], "activation_workspace": activation}


def _files(entries, root, bindings):
    root = Path(root).resolve(strict=True)
    if not entries:
        raise ValueError("actual file inventory is empty")
    paths = []
    for entry in entries:
        path = Path(entry["path"])
        if path.is_absolute() or ".." in path.parts or not _hash(entry.get("sha256")):
            raise ValueError("invalid admitted file path/hash")
        source = root / path
        actual = source.resolve(strict=True)
        if not actual.is_relative_to(root) or source.is_symlink() or not actual.is_file():
            raise ValueError("admitted file escapes its root or is not a regular file")
        digest = file_sha256(actual)
        if digest != entry["sha256"]:
            raise ValueError("actual admitted file bytes differ")
        prior = bindings.setdefault(str(actual), digest)
        if prior != digest:
            raise ValueError("same admitted path has conflicting hashes")
        paths.append(path.as_posix())
    if len(paths) != len(set(paths)):
        raise ValueError("duplicate admitted inventory path")


def _artifact(record, root, bindings):
    _files([{"path": record["artifact_path"], "sha256": record["artifact_sha256"]}], root, bindings)


def _worker_identity(receipt):
    if "error" in receipt or type(receipt.get("rank")) is not int or receipt["rank"] not in RANKS:
        raise ValueError("resident rank failed")
    if type(receipt.get("pid")) is not int or receipt["pid"] <= 0:
        raise ValueError("resident worker PID missing")
    if type(receipt.get("generation")) is not int or receipt["generation"] < 0:
        raise ValueError("resident generation missing")
    if not _hash(receipt.get("weight_storage_digest")) or not receipt.get("execution_namespace"):
        raise ValueError("resident storage/namespace ownership missing")
    return tuple(receipt[field] for field in IDENTITY_FIELDS)


def _rank_identities(receipts):
    if len(receipts) != 4 or sorted(r.get("rank", -1) for r in receipts) != list(RANKS):
        raise ValueError("exactly four resident rank identities required")
    return tuple(_worker_identity(receipt) for receipt in sorted(receipts, key=lambda r: r["rank"]))


@dataclass(frozen=True, init=False)
class Admission:
    """Created by admit only; includes actual-byte bindings rechecked on use."""

    manifest_sha256: str
    reference_sha256: str
    candidate_sha256: str
    plan_sha256: str
    configuration_sha256: str
    resources_sha256: str
    generation: int
    engine_id: str
    rank_plan_receipts: tuple
    rank_memory_envelopes: tuple
    baseline_identities: tuple
    file_bindings: tuple
    seal: str

    @property
    def plan_receipts(self):
        """Fresh decoded copies; callers cannot mutate admitted rank bindings."""
        return tuple(json.loads(receipt) for receipt in self.rank_plan_receipts)

    def _body(self):
        return {field: getattr(self, field) for field in self.__dataclass_fields__ if field != "seal"}

    def require_execution(self, *, configuration_sha256, resources_sha256, plan_sha256=None):
        if self.seal != canonical_sha256(self._body()):
            raise ValueError("admission identity corrupted")
        if configuration_sha256 != self.configuration_sha256 or resources_sha256 != self.resources_sha256:
            raise ValueError("execution config/resources differ from admission")
        if plan_sha256 is not None and plan_sha256 != self.plan_sha256:
            raise ValueError("execution plan differs from admission")
        for path, expected in self.file_bindings:
            actual = Path(path)
            if actual.is_symlink() or not actual.is_file() or file_sha256(actual) != expected:
                raise ValueError("admitted bytes changed before execution")
        return self


def admit(
    manifest,
    evidence,
    *,
    bundle_root,
    reference_git_root,
    artifact_roots,
    plan,
    plan_receipts,
    memory_receipts,
    configuration_sha256,
    resources_sha256,
):
    """Require complete T1 live gates and verify actual local artifact bytes.

    Candidate carries configuration/configuration_sha256, resources_sha256,
    base_reference_sha256 and the T1 candidate source/build inventories. External
    checkpoint and serving_bundle receipts carry files plus their canonical list
    SHA256. Gate/component/trial records carry artifact_path/artifact_sha256.
    artifact_roots supplies checkpoint, serving_bundle and hardware_evidence.
    Current offline manifests fail this API; no library is loaded here.
    """
    validate_manifest(manifest, reference_git_root, require_live=True)
    validate_evidence(evidence, manifest, require_live=True)
    candidate = manifest["candidate"]
    if verify_bundle(bundle_root, require_compiled=True) != candidate:
        raise ValueError("admitted candidate differs from actual compiled bundle")
    if (
        candidate.get("base_reference_sha256") != plan.reference_sha256
        or candidate.get("configuration_sha256") != configuration_sha256
        or canonical_sha256(candidate.get("configuration")) != configuration_sha256
        or candidate.get("resources_sha256") != resources_sha256
        or resource_inventory_sha256(candidate) != resources_sha256
    ):
        raise ValueError("candidate reference/configuration/resource identity mismatch")
    if (
        plan.source_sha256 != manifest["source"]["source_sha256"]
        or plan.arithmetic_sha256 != manifest["arithmetic_sha256"]
    ):
        raise ValueError("schedule baseline arithmetic/source mismatch")
    validate_rank_plan(plan, plan_receipts)
    identities = _rank_identities(evidence["rank_receipts"])
    by_rank = {record[0]: record for record in identities}
    for receipt in plan_receipts:
        old = by_rank[receipt["rank"]]
        if (
            receipt["pid"] != old[1]
            or receipt.get("previous_generation") != old[2]
            or receipt.get("weight_storage_digest") != old[3]
            or receipt["execution_namespace"] != old[4]
            or receipt["execution_namespace"] != candidate["namespace"]
            or receipt.get("candidate_sha256") != candidate["candidate_sha256"]
        ):
            raise ValueError("rank storage/generation/candidate ownership differs")
    if not isinstance(evidence.get("engine_id"), str) or not evidence["engine_id"]:
        raise ValueError("engine identity missing")
    if len(memory_receipts) != 4 or sorted(r.get("rank", -1) for r in memory_receipts) != list(RANKS):
        raise ValueError("complete per-rank memory admission required")
    envelopes = []
    for receipt in sorted(memory_receipts, key=lambda r: r["rank"]):
        if receipt.get("generation") != plan.generation or receipt.get("plan_sha256") != plan.sha256:
            raise ValueError("rank memory receipt is stale")
        old = by_rank[receipt["rank"]]
        if (
            receipt.get("pid") != old[1]
            or receipt.get("weight_storage_digest") != old[3]
            or receipt.get("candidate_sha256") != candidate["candidate_sha256"]
            or receipt.get("backend_scratch_resolved") is not True
        ):
            raise ValueError("rank memory ownership or backend scratch unresolved")
        bounds = minimum_workspaces(plan)
        if any(
            type(receipt["components"].get(key)) is not int or receipt["components"][key] < value
            for key, value in bounds.items()
        ):
            raise ValueError("candidate workspace bound exceeds declared rank budget")
        envelopes.append(
            rank_envelope(receipt["available_bytes"], receipt["components"], receipt.get("guard_bytes", 0))
        )
    bindings = {}
    links = manifest["source"].get("gitlinks", [])
    external = evidence.get("external_source_inventory", [])
    if {entry.get("path") for entry in external} != {link["path"] for link in links} or len(external) != len(links):
        raise ValueError("external reference source byte coverage missing")
    for link in links:
        record = next(entry for entry in external if entry["path"] == link["path"])
        if record.get("commit") != link["commit"] or record.get("sha256") != canonical_sha256(record.get("files")):
            raise ValueError("external source commit/inventory differs")
        external_root = Path(artifact_roots["external_source"]) / link["path"]
        _files(record["files"], external_root, bindings)
        tree = subprocess.run(
            ["git", "-C", str(external_root), "ls-tree", "-r", record["commit"]],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
        if any(line.split()[1] != "blob" for line in tree):
            raise ValueError("nested external Git source links require additional verified byte inventories")
        blobs = [line.split("\t", 1)[1] for line in tree if line.split()[1] == "blob"]
        if sorted(entry["path"] for entry in record["files"]) != sorted(blobs):
            raise ValueError("expanded external Git content inventory incomplete")
        for entry in record["files"]:
            actual_blob = subprocess.run(
                ["git", "-C", str(external_root), "show", f"{record['commit']}:{entry['path']}"],
                check=True,
                capture_output=True,
            ).stdout
            if hashlib.sha256(actual_blob).hexdigest() != entry["sha256"]:
                raise ValueError("external inventory does not match pinned Git blob bytes")
    for field in ("assets", "binaries", "bridges"):
        _files(candidate[field], bundle_root, bindings)
    _files(
        [
            {"path": name, "sha256": file_sha256(Path(bundle_root) / name)}
            for name in ("prepared.json", "candidate.json")
        ]
        + [candidate["build_receipt"]],
        bundle_root,
        bindings,
    )
    compile_receipt = json.loads((Path(bundle_root) / candidate["build_receipt"]["path"]).read_text())
    containment = compile_receipt.get("filesystem_containment")
    if not isinstance(containment, dict):
        raise ValueError("uncontained host compilation cannot admit a candidate")
    _files([containment["helper"]], bundle_root, bindings)
    policy = containment["policy"]
    if containment.get("policy_sha256") != canonical_sha256(policy):
        raise ValueError("filesystem containment policy identity differs")
    for attestation in compile_receipt["command_attestations"]:
        _files([attestation["stdout"], attestation["stderr"]], bundle_root, bindings)
    if compile_receipt.get("abi_evidence") is not None:
        _files([compile_receipt["abi_evidence"]], bundle_root, bindings)
    contracts = [entry for entry in candidate["assets"] if entry.get("role") == "streaming_contract"]
    if len(contracts) != 1:
        raise ValueError("exactly one actual streaming contract required")
    value = json.loads((Path(bundle_root) / contracts[0]["path"]).read_text())
    seal = value.pop("contract_sha256", None)
    if seal != canonical_sha256(value) or seal != candidate["contract_sha256"]:
        raise ValueError("actual streaming contract identity differs")
    for name in ("checkpoint", "serving_bundle"):
        receipt = manifest[name]
        if canonical_sha256(receipt.get("files")) != receipt["sha256"]:
            raise ValueError("external content inventory identity differs")
        _files(receipt["files"], artifact_roots[name], bindings)
    artifact_root = artifact_roots["hardware_evidence"]
    for receipt in memory_receipts:
        _artifact(receipt, artifact_root, bindings)
    for gate in evidence["gates"].values():
        _artifact(gate, artifact_root, bindings)
    for component in evidence["native_component_evidence"].values():
        _artifact(component, artifact_root, bindings)
    for trial in evidence["trials"]:
        _artifact(trial, artifact_root, bindings)
    for workload in manifest["workloads"]:
        if workload["kind"] == "image":
            _files(
                [{"path": workload["image_content_path"], "sha256": workload["image_sha256"]}], artifact_root, bindings
            )
    result = object.__new__(Admission)
    fields = {
        "manifest_sha256": manifest["manifest_sha256"],
        "reference_sha256": plan.reference_sha256,
        "candidate_sha256": candidate["candidate_sha256"],
        "plan_sha256": plan.sha256,
        "configuration_sha256": configuration_sha256,
        "resources_sha256": resources_sha256,
        "generation": plan.generation,
        "engine_id": evidence["engine_id"],
        "rank_plan_receipts": tuple(
            json.dumps(r, sort_keys=True) for r in sorted(plan_receipts, key=lambda r: r["rank"])
        ),
        "rank_memory_envelopes": tuple(json.dumps(r, sort_keys=True) for r in envelopes),
        "baseline_identities": identities,
        "file_bindings": tuple(sorted(bindings.items())),
    }
    for field, value in fields.items():
        object.__setattr__(result, field, value)
    object.__setattr__(result, "seal", canonical_sha256(fields))
    return result


class HeldTransaction(RuntimeError):
    """Workers may remain held; only explicit recovery can proceed."""


class HaltedTransaction(HeldTransaction):
    """Unresolved execution forbids any further RPC until external proof."""


@dataclass(frozen=True)
class WorkerPreparation:
    """Metadata-only preparation retains an undo token before any mutation."""

    admission: Admission
    apply_token: object
    restore_token: object


class WorkerTransaction:
    """Worker seam exposed by a future extension; no server/device imports.

    snapshot() supplies normalized integer generation and actual rank/PID,
    storage/namespace/graph health, and outstanding_collective=False. prepare()
    callback must only hash/validate metadata and return WorkerPreparation; it
    cannot load a library or kernel. apply(token) is the only loader/installer
    callback. restore(undo_token) restores already saved dispatch. The callback
    contract cannot prevent arbitrary user callback side effects; tests verify
    these responsibilities through independent fake worker implementations.
    """

    def __init__(self, *, snapshot, prepare, apply, restore):
        self.snapshot, self.prepare_callback = snapshot, prepare
        self.apply_callback, self.restore_callback = apply, restore
        self.lock = RLock()
        self.baseline = dict(snapshot())
        _worker_identity(self.baseline)
        self.generation = self.baseline["generation"]
        self.owner = None
        self.preparation = None
        self.may_have_mutated = False
        self.active = False
        self.poisoned = False

    def _physical_status(self):
        current = dict(self.snapshot())
        if any(
            current.get(field) != self.baseline.get(field)
            for field in ("rank", "pid", "weight_storage_digest", "execution_namespace")
        ):
            raise HeldTransaction("worker physical identity/storage changed")
        if current.get("graphs_dirty") is not False or current.get("native_failed") is not False:
            raise HeldTransaction("worker graph/native resources unhealthy")
        if current.get("outstanding_collective") is not False:
            raise HaltedTransaction("worker still owns an unresolved collective")
        if current.get("pending_execution") is not False:
            raise HeldTransaction("worker pending execution completion is unknown")
        return current

    def _owned_maintenance(self, owner):
        current = self._physical_status()
        if current.get("paused") is not True or current.get("maintenance_owner") != owner:
            raise HeldTransaction("worker mutation lacks the authoritative owned maintenance lease")

    def status(self):
        with self.lock:
            result = self._physical_status()
            result.update(generation=self.generation, transaction_id=self.owner, poisoned=self.poisoned)
            if self.active:
                result["candidate_sha256"] = self.preparation.admission.candidate_sha256
            return result

    def prepare(self, payload):
        with self.lock:
            self._owned_maintenance(payload.get("transaction_id"))
            if self.active or self.may_have_mutated or self.poisoned:
                raise HeldTransaction("worker session must finish/recover before another preparation")
            candidate = self.prepare_callback(payload)
            if not isinstance(candidate, WorkerPreparation) or not isinstance(candidate.admission, Admission):
                raise ValueError("worker preparation needs verified metadata-only admission")
            admission = candidate.admission
            admission.require_execution(
                configuration_sha256=payload.get("configuration_sha256"),
                resources_sha256=payload.get("resources_sha256"),
                plan_sha256=payload.get("plan_sha256"),
            )
            old = admission.baseline_identities[self.baseline["rank"]]
            if (
                tuple(self.baseline[field] for field in IDENTITY_FIELDS) != old
                or payload.get("generation") != admission.generation
            ):
                raise HeldTransaction("worker preparation baseline/generation differs")
            if (
                payload.get("admission_sha256") != admission.seal
                or payload.get("candidate_sha256") != admission.candidate_sha256
            ):
                raise HeldTransaction("worker preparation payload binding differs")
            if not isinstance(payload.get("transaction_id"), str) or not payload["transaction_id"]:
                raise ValueError("worker transaction owner missing")
            self.preparation, self.owner = candidate, payload["transaction_id"]
            result = self.status()
            result["prepared_sha256"] = admission.candidate_sha256
            return result

    def apply(self, transaction_id):
        with self.lock:
            self._owned_maintenance(transaction_id)
            if transaction_id != self.owner or self.preparation is None or self.poisoned:
                raise HeldTransaction("worker apply owner/preparation unresolved")
            if self.active:
                return self.status()
            admission = self.preparation.admission
            admission.require_execution(
                configuration_sha256=admission.configuration_sha256,
                resources_sha256=admission.resources_sha256,
            )
            self.may_have_mutated = True
            try:
                self.apply_callback(self.preparation.apply_token)
                self._physical_status()
            except BaseException:
                self.poisoned = True
                raise
            self.generation, self.active = admission.generation, True
            return self.status()

    def restore(self, transaction_id):
        with self.lock:
            self._owned_maintenance(transaction_id)
            if transaction_id != self.owner or self.preparation is None:
                raise HeldTransaction("worker restoration has no owned preparation")
            if self.may_have_mutated:
                try:
                    self.restore_callback(self.preparation.restore_token)
                    self._physical_status()
                except BaseException:
                    self.poisoned = True
                    raise
            self.generation = self.baseline["generation"]
            self.active, self.may_have_mutated, self.poisoned = False, False, False
            self.preparation, self.owner = None, None
            return self.status()


class ResidentController:
    """One invocation owns its pause and any mutation; no automatic retries.

    Client.request uses the existing ResidentClient signature. scheduler_state is
    an injected authoritative probe returning engine_id, running, waiting,
    paused, maintenance_owner, outstanding_collective, thermal_hold and four
    temperatures_c. Existing/foreign pauses cannot be appropriated. RPC replies
    retain raw partial/error acknowledgments rather than hiding them in retries.
    Worker seam: streaming_prepare(payload), streaming_apply(transaction_id),
    streaming_restore(transaction_id), and streaming_status(). No reset/recapture
    RPC is called; worker composition must preserve existing graphs and caches.
    """

    def __init__(self, client, journal_dir, transaction_id, *, live=False, scheduler_state=None):
        if not isinstance(transaction_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", transaction_id):
            raise ValueError("transaction ID must be a simple unique filename")
        self.client, self.journal_dir, self.transaction_id = client, Path(journal_dir), transaction_id
        self.live, self.scheduler_state = live, scheduler_state
        self.state = "NEW"
        self.owns_pause = False
        self.may_have_mutated = False
        self.sequence = 0
        self.admission = None

    def _record(self, phase, **value):
        self.journal_dir.mkdir(parents=True, exist_ok=True)
        path = self.journal_dir / f"{self.transaction_id}-{self.sequence:03d}-{phase}.json"
        content = {"transaction_id": self.transaction_id, "phase": phase, "state": self.state, **value}
        with path.open("x", encoding="utf-8") as stream:
            json.dump(content, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        directory = os.open(self.journal_dir, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        self.sequence += 1

    def _halt(self, message):
        self.state = "HALTED"
        raise HaltedTransaction(message)

    def _call(self, method, *args):
        if self.state == "HALTED":
            raise HaltedTransaction("unresolved transaction: no further RPC")
        try:
            response = self.client.request("/collective_rpc", {"method": method, "args": list(args)})
        except BaseException as error:
            self.state = "HALTED"
            raise HaltedTransaction(f"{method} completion unknown; no status/recovery/resume RPC") from error
        if (
            not isinstance(response, dict)
            or response.get("outstanding_collective") is not False
            or response.get("collective_completed") is not True
        ):
            self._halt("collective completion outstanding or unknown")
        receipts = response.get("results")
        if not isinstance(receipts, list):
            self._halt("collective acknowledgment completion unknown")
        return receipts

    def _scheduler(self):
        if self.state == "HALTED":
            raise HaltedTransaction("scheduler probe forbidden until external completion proof")
        value = self.scheduler_state()
        if value.get("outstanding_collective") is not False:
            self._halt("outstanding collective prevents maintenance")
        if value.get("engine_id") != self.admission.engine_id:
            raise HeldTransaction("engine identity changed")
        if any(type(value.get(key)) is not int or value[key] != 0 for key in ("running", "waiting")):
            raise HeldTransaction("active or queued requests: leave server untouched")
        temperatures = value.get("temperatures_c")
        if (
            value.get("thermal_hold") is not False
            or not isinstance(temperatures, list)
            or len(temperatures) != 4
            or any(type(t) not in (int, float) or not 0 <= t < HOLD_C for t in temperatures)
        ):
            raise HeldTransaction("thermal hold or missing/hot sensors prevents switching")
        if type(value.get("paused")) is not bool:
            raise HeldTransaction("scheduler pause ownership unknown")
        return value

    def _validate_replies(self, receipts, *, applied=False):
        identities = _rank_identities(receipts)
        baseline = self.admission.baseline_identities
        for current, previous in zip(identities, baseline):
            expected_generation = self.admission.generation if applied else previous[2]
            if current != (previous[0], previous[1], expected_generation, previous[3], previous[4]):
                raise HeldTransaction("rank PID/storage/namespace/generation changed")
        if any(r.get("graphs_dirty") is not False or r.get("native_failed") is not False for r in receipts):
            raise HeldTransaction("worker graphs/resource health unresolved")
        if any(r.get("pending_execution") is not False for r in receipts):
            raise HeldTransaction("worker still has an unresolved model execution")
        if applied and any(r.get("candidate_sha256") != self.admission.candidate_sha256 for r in receipts):
            raise HeldTransaction("ranks did not apply the same candidate")

    def execute(self, admission, payload):
        if self.state != "NEW" or not isinstance(admission, Admission):
            raise HeldTransaction("one fresh invocation and verified admission required")
        admission.require_execution(
            configuration_sha256=admission.configuration_sha256,
            resources_sha256=admission.resources_sha256,
            plan_sha256=admission.plan_sha256,
        )
        self.admission = admission
        expected = {
            "transaction_id": self.transaction_id,
            "admission_sha256": admission.seal,
            "candidate_sha256": admission.candidate_sha256,
            "configuration_sha256": admission.configuration_sha256,
            "resources_sha256": admission.resources_sha256,
            "plan_sha256": admission.plan_sha256,
            "generation": admission.generation,
        }
        if any(payload.get(key) != value for key, value in expected.items()):
            raise ValueError("controller payload differs from verified admission")
        if not self.live:
            self.state = "DRY_RUN"
            return {"state": self.state, "rpc_calls": 0, "admission_sha256": admission.seal}
        if not callable(self.scheduler_state):
            raise ValueError("live controller requires an authoritative scheduler/ownership probe")
        state = self._scheduler()
        if state["paused"] or state.get("maintenance_owner") is not None:
            raise HeldTransaction("existing pause/maintenance belongs to another invocation")
        self._validate_replies(self._call("streaming_status"))
        self._record(
            "intent", admission_sha256=admission.seal, candidate_sha256=admission.candidate_sha256, payload=payload
        )
        # Recheck after the append-only intent and before the first mutation.
        state = self._scheduler()
        if state["paused"] or state.get("maintenance_owner") is not None:
            raise HeldTransaction("maintenance changed before pause")
        self.owns_pause = True
        try:
            reply = self.client.request("/pause?mode=wait&clear_cache=false", {"owner": self.transaction_id})
        except BaseException as error:
            self.state = "HALTED"
            raise HaltedTransaction("pause completion unknown; no further RPC") from error
        if not isinstance(reply, dict) or reply.get("status") != "paused":
            self._halt("pause completion unknown")
        self.state = "PAUSED"
        self._record("paused")
        try:
            state = self._scheduler()
            if not state["paused"] or state.get("maintenance_owner") != self.transaction_id:
                raise HeldTransaction("pause was not exclusively acquired")
            prepared = self._call("streaming_prepare", json.dumps(payload, sort_keys=True))
            self._validate_replies(prepared)
            if any(r.get("prepared_sha256") != self.admission.candidate_sha256 for r in prepared):
                raise HeldTransaction("candidate preparation does not match admission")
            admission.require_execution(
                configuration_sha256=admission.configuration_sha256,
                resources_sha256=admission.resources_sha256,
            )
            self._record("apply-intent", generation=admission.generation)
            self.may_have_mutated = True
            self._validate_replies(self._call("streaming_apply", self.transaction_id), applied=True)
            self.state = "APPLIED"
            self._record("applied")
            self._resume()
        except HaltedTransaction:
            raise
        except BaseException as error:
            self.state = "HELD"
            raise HeldTransaction(
                "candidate switch incomplete; owned pause retained; explicit recovery required"
            ) from error
        return {"state": self.state, "candidate_sha256": admission.candidate_sha256}

    def _resume(self):
        state = self._scheduler()
        if not self.owns_pause or not state["paused"] or state.get("maintenance_owner") != self.transaction_id:
            raise HeldTransaction("resume would release a foreign pause")
        if any(t > RESUME_C for t in state["temperatures_c"]):
            raise HeldTransaction("all cores must cool to 85C before owned resume")
        self._record("resume-intent")
        try:
            reply = self.client.request("/resume", {"owner": self.transaction_id})
        except BaseException as error:
            self.state = "HALTED"
            raise HaltedTransaction("resume completion unknown") from error
        if not isinstance(reply, dict) or reply.get("status") != "resumed":
            self._halt("resume acknowledgment unknown")
        self.owns_pause = False
        self.state = "COMPLETE"
        self._record("complete")

    def recover(self):
        if self.state == "HALTED":
            raise HaltedTransaction("external completion proof required before recovery")
        if self.state != "HELD" or not self.owns_pause:
            raise HeldTransaction("no owned incomplete transaction to recover")
        state = self._scheduler()
        if not state["paused"] or state.get("maintenance_owner") != self.transaction_id:
            raise HeldTransaction("recovery would touch a foreign pause")
        if self.may_have_mutated:
            self._record("restore-intent")
            try:
                self._validate_replies(self._call("streaming_restore", self.transaction_id))
            except BaseException as error:
                self.state = "HALTED"
                raise HaltedTransaction("recovery failed; no further RPC") from error
        self.state = "RESTORED"
        self._record("restored")
        self._resume()

    def acknowledge_external_completion(self, proof, *, proof_path):
        """No RPC: consume operator-supplied completion/held-ownership proof.

        The caller must obtain this out of band, never by issuing another RPC
        through a possibly blocked executor. This prepares explicit recovery;
        it cannot resume or reapply a partially installed candidate.
        """
        if self.state != "HALTED" or self.admission is None:
            raise HeldTransaction("no unresolved admitted transaction")
        artifact = Path(proof_path)
        if artifact.is_symlink() or not artifact.is_file() or file_sha256(artifact) != proof.get("artifact_sha256"):
            raise HaltedTransaction("external completion artifact bytes missing or changed")
        body = {key: value for key, value in proof.items() if key != "artifact_sha256"}
        if json.loads(artifact.read_text()) != body:
            raise HaltedTransaction("external completion artifact contents differ")
        if (
            proof.get("transaction_id") != self.transaction_id
            or proof.get("engine_id") != self.admission.engine_id
            or proof.get("collectives_completed") is not True
            or proof.get("paused") is not True
            or proof.get("maintenance_owner") != self.transaction_id
            or not self.owns_pause
        ):
            raise HaltedTransaction("external proof does not establish completion and owned hold")
        current = _rank_identities(proof.get("rank_receipts", []))
        for now, old in zip(current, self.admission.baseline_identities):
            if now[0:2] != old[0:2] or now[3:] != old[3:] or now[2] not in (old[2], self.admission.generation):
                raise HaltedTransaction("external proof worker identity/storage changed")
        self._record("external-completion", proof=proof)
        self.state = "HELD"
