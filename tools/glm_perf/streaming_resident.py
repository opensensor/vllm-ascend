# SPDX-License-Identifier: Apache-2.0
"""Guarded, serial resident installation and complete GLM streaming trials.

Default CLI commands inspect or prepare a dry run. Hardware mutation requires
--execute and an owned idle maintenance window. Complete native evidence is the
default admission policy; explicit --experimental permits hashed pending
diagnostics without qualifying trials or promotion.
Unresolved collective receipts never trigger another collective or recovery.
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import fcntl
import hashlib
import json
import math
import statistics
import threading
import time
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import regex as re

from tools.glm_perf.resident_candidates.streaming_experts import compose_source
from tools.glm_perf.resident_control import Control
from tools.glm_perf.resident_harness import ResidentClient
from tools.glm_perf.streaming_protocol import (
    canonical_digest,
    file_digest,
    shadow_case_ids,
    summarize_trials,
    validate_evidence,
    verify_reference,
)

WORKERS = 4
SOURCE_FIELDS = ("rank", "pid", "weight_storage_digest")
MEMORY_COMPONENTS = frozenset(
    {"old_resources", "new_resources", "reference_scratch", "candidate_scratch", "outputs", "verification"}
)
PRIMARY_WORKLOAD = "cold_6400"
ABBA = ("reference", "candidate", "candidate", "reference")
RPC_TIMEOUT = 900


class UnresolvedOperation(RuntimeError):
    """The prior executor operation has no proven complete receipt set."""


class CompletedRPCError(RuntimeError):
    def __init__(self, method, receipts):
        self.method, self.receipts = method, receipts
        super().__init__(f"{method}: completed four-rank error receipts: {receipts}")


class StreamingTransport(ResidentClient):
    """Existing resident transport with text metrics and streamed inference."""

    def request(self, path, payload=None, method="POST"):
        if path != "/metrics":
            return super().request(path, payload, method)
        with urllib.request.urlopen(self.base_url + path, timeout=self.timeout) as response:
            return response.read().decode()

    def stream(self, path, payload, request_id, *, on_first_token=None):
        return stream_request(
            self.base_url, path, payload, request_id=request_id, timeout=self.timeout, on_first_token=on_first_token
        )


def token_bearing(event):
    """Exclude role-only/usage-only SSE events; count reasoning/tool deltas."""
    for choice in event.get("choices", []):
        if choice.get("text") or choice.get("token_ids") or (choice.get("logprobs") or {}).get("tokens"):
            return True
        delta = choice.get("delta", {})
        if any(delta.get(key) for key in ("content", "reasoning_content", "reasoning")):
            return True
        if any(call.get("function", {}).get("arguments") for call in delta.get("tool_calls", [])):
            return True
    return False


def consume_sse(lines, *, started, now=time.monotonic, request_id, on_first_token=None):
    first = last = visible = None
    first_event = usage = server_timing = None
    events, text, reasoning, tools, response_id = [], [], [], {}, None
    done = False
    for raw in lines:
        if not raw.startswith(b"data:"):
            continue
        data = raw[5:].strip()
        if data == b"[DONE]":
            done = True
            break
        event = json.loads(data)
        if event.get("error"):
            raise RuntimeError(f"inference error: {event['error']}")
        when = now()
        events.append(event)
        if event.get("id"):
            if response_id is not None and event["id"] != response_id:
                raise ValueError("SSE events changed request identity")
            response_id = event["id"]
        if event.get("usage"):
            usage = event["usage"]
        if token_bearing(event):
            if first is None:
                first, first_event = when, event
                if on_first_token is not None:
                    on_first_token()
            last = when
        for choice in event.get("choices", []):
            delta = choice.get("delta", {})
            piece = choice.get("text") or delta.get("content")
            if piece:
                text.append(piece)
                visible = when if visible is None else visible
            reasoning.append(delta.get("reasoning_content") or delta.get("reasoning") or "")
            for call in delta.get("tool_calls", []):
                slot = tools.setdefault(call.get("index", 0), {"name": "", "arguments": ""})
                function = call.get("function", {})
                slot["name"] += function.get("name", "")
                slot["arguments"] += function.get("arguments", "")
        timing = event.get("server_timing")
        if (
            isinstance(timing, dict)
            and isinstance(timing.get("request_id"), str)
            and timing["request_id"] in (request_id, response_id)
        ):
            admission, generated = timing.get("admission_time"), timing.get("first_generated_token_time")
            if (
                type(admission) in (int, float)
                and type(generated) in (int, float)
                and timing.get("clock")
                and math.isfinite(admission)
                and math.isfinite(generated)
                and 0 <= admission <= generated
            ):
                server_timing = {**timing, "ttft_s": generated - admission}
    finished = now()
    if not done or usage is None or first is None:
        raise ValueError("SSE request lacks DONE, usage or token-bearing output")
    completions = usage.get("completion_tokens")
    if type(completions) is not int or completions < 1:
        raise ValueError("invalid completed generation length")
    duration = last - first
    return {
        "request_id": request_id,
        "response_id": response_id,
        "complete": True,
        "usage": usage,
        "prompt_tokens": usage["prompt_tokens"],
        "completion_tokens": completions,
        "client_started_monotonic": started,
        "first_generated_event_monotonic": first,
        "last_generated_event_monotonic": last,
        "streamed_ttft_s": first - started,
        "ttft_s": server_timing["ttft_s"] if server_timing else first - started,
        "timing_method": "server_generated_token" if server_timing else "streamed_token_event",
        "server_timing": server_timing,
        "first_token_event": first_event,
        "first_visible_text_s": visible - started if visible is not None else None,
        "completion_latency_s": finished - started,
        "generation_tok_s": (completions - 1) / duration if completions > 1 and duration > 0 else None,
        "generation_rate_method": "usage count / first-to-last token-bearing SSE event; events may coalesce tokens",
        "text": "".join(text),
        "reasoning": "".join(reasoning),
        "tool_calls": list(tools.values()),
        "events": events,
    }


def stream_request(base_url, path, payload, *, request_id, timeout=RPC_TIMEOUT, on_first_token=None):
    request = urllib.request.Request(
        base_url.rstrip("/") + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "X-Request-Id": request_id},
    )
    started = time.monotonic()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return consume_sse(response, started=started, request_id=request_id, on_first_token=on_first_token)


@dataclasses.dataclass(frozen=True)
class Candidate:
    identity: dict[str, Any]
    dispatch: dict[str, Any]
    build_dir: Path
    gate_digest: str
    memory_admission: dict[str, Any]
    manifest: Any
    experimental: bool = False


def load_candidate(reference, gates_path, *, build_dir=None, manifest_factory=None, experimental=False):
    gates_path = Path(gates_path)
    gates = json.loads(gates_path.read_text())
    validate_evidence(
        gates,
        kind="native_gates",
        bindings={"reference_sha256": reference["reference_sha256"]},
        artifact_root=gates_path.parent,
        require_complete=not experimental,
    )
    if experimental and (
        gates["state"] != "pending"
        or not gates["coverage"]
        or gates["errors"]
        or gates["result"].get("experimental", True) is not True
        or gates["result"].get("diagnostic_passed", True) is not True
    ):
        raise ValueError("experimental install requires matching nonempty passed pending diagnostic evidence")
    result = gates["result"]
    root = Path(build_dir or result["build_dir"]).resolve(strict=True)
    build_path = Path(result["build_evidence"])
    if not build_path.is_absolute():
        build_path = (gates_path.parent / build_path).resolve(strict=True)
    build = json.loads(build_path.read_text())
    validate_evidence(build, kind="build", bindings=gates["identity"], artifact_root=build_path.parent)
    for name, digest in gates["identity"]["binaries"].items():
        path = (root / name).resolve(strict=True)
        if not path.is_relative_to(root) or file_digest(path) != digest:
            raise ValueError("candidate binary does not match gated identity")
    provenance = json.loads((root / "provenance.json").read_text())
    if (
        provenance["_build"].get("streaming_pipeline") is not True
        or provenance["_build"].get("version") != gates["identity"]["version"]
    ):
        raise ValueError("native gates do not select this complete streaming build")
    memory = result.get("resident_memory_admission", {})
    if (
        memory.get("passed") is not True
        or type(memory.get("required_free_bytes")) is not int
        or memory["required_free_bytes"] < 0
        or not MEMORY_COMPONENTS.issubset(memory.get("includes", []))
    ):
        raise ValueError("live installation needs whole-rank old/new/shadow memory admission")
    from tools.glm_perf.build_streaming import admit_candidate

    admitted = admit_candidate(root, native_evidence_path=None if experimental else gates_path, live=not experimental)
    if admitted != gates["identity"]:
        raise ValueError("admitted build identity differs from submitted gates")
    if experimental:
        if manifest_factory is None:
            from tools.glm_perf.fused_moe_control import experimental_manifest as manifest_factory
        manifest = manifest_factory(root, diagnostic_report=gates_path)
    else:
        legacy = Path(result["legacy_gate_report"])
        if not legacy.is_absolute():
            legacy = (gates_path.parent / legacy).resolve(strict=True)
        if manifest_factory is None:
            # Import torch-dependent checks only after complete admission.
            from tools.glm_perf.fused_moe_control import manifest as manifest_factory
        manifest = manifest_factory(root, legacy, streaming_gate_report=gates_path)
    manifest.verify_files()
    return Candidate(
        copy.deepcopy(gates["identity"]),
        copy.deepcopy(gates["dispatch"]),
        root,
        file_digest(gates_path),
        copy.deepcopy(memory),
        manifest,
        experimental=experimental,
    )


class GuardedResidentController:
    def __init__(
        self,
        client,
        reference,
        reference_dir,
        output_dir,
        *,
        candidate=None,
        expected_generation=None,
        lock_path=None,
        record=None,
    ):
        self.client, self.reference, self.reference_dir = client, reference, Path(reference_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.candidate = candidate
        self.expected_generation = expected_generation or reference["runtime"]["workers"][0]["generation"]
        self.lock_path = Path(lock_path or f"/tmp/glm-streaming-resident-{urlparse(client.base_url).port or 80}.lock")
        self.record_hook = record
        self.journal = {
            "state": "prepared",
            "events": [],
            "owned_maintenance": False,
            "paused": False,
            "outstanding": None,
            "changed_dispatch": False,
            "recovery_error": None,
        }
        self.owned = self.unresolved = self.changed = False
        self.native_attempted = False
        self.last_workers = None
        self.sources = {}
        self.journal["experimental"] = bool(candidate and candidate.experimental)

    def record(self, event, **values):
        # Receipt/sample arrays are immutable evidence files. Rewriting all
        # prior arrays on every RPC would create quadratic host I/O during
        # long ABBA runs; the durable journal stores their hashes and paths.
        original_values = values
        if len(json.dumps(values, separators=(",", ":"), allow_nan=False)) > 4096:
            directory = self.output_dir / "journal-data"
            directory.mkdir(exist_ok=True)
            path = directory / f"{len(self.journal['events']):05d}-{event}.json"
            _write_json(path, values)
            values = {"data_artifact": {"path": str(path.relative_to(self.output_dir)), "sha256": file_digest(path)}}
        self.journal["events"].append({"event": event, "time_monotonic": time.monotonic(), **values})
        temporary = self.output_dir / "controller-journal.json.tmp"
        temporary.write_text(json.dumps(self.journal, indent=2, allow_nan=False) + "\n")
        temporary.replace(self.output_dir / "controller-journal.json")
        if self.record_hook:
            self.record_hook(event, original_values)

    def _unknown(self, method, reason):
        self.unresolved = True
        self.journal["state"] = "recovery_failed"
        self.journal["recovery_error"] = (
            f"{method}: unresolved operation; no further collective/restore/resume: {reason}"
        )
        self.record("unresolved", method=method, reason=str(reason))
        raise UnresolvedOperation(self.journal["recovery_error"])

    def rpc(self, method, *args, expected=None, rankless=False):
        if self.unresolved:
            raise UnresolvedOperation("prior operation is unresolved; refusing another collective")
        self.journal["outstanding"] = {"method": method, "args_digest": canonical_digest(args)}
        self.record("rpc_intent", method=method)
        try:
            response = self.client.request(
                "/collective_rpc", {"method": method, "args": list(args), "timeout": self.client.timeout}
            )
        except Exception as error:
            self._unknown(method, repr(error))
        rows = response.get("results") if isinstance(response, dict) else None
        if not isinstance(rows, list) or len(rows) != WORKERS or not all(isinstance(row, dict) for row in rows):
            self._unknown(method, "missing/malformed rank receipts")
        if not (rankless and all("rank" not in row for row in rows)) and {row.get("rank") for row in rows} != set(
            range(WORKERS)
        ):
            self._unknown(method, "duplicate/missing rank identities")
        self.journal["outstanding"] = None
        self.record("rpc_completed", method=method, receipts=rows)
        if any("error" in row for row in rows):
            raise CompletedRPCError(method, rows)
        if expected is not None and any(any(row.get(key) != value for key, value in expected.items()) for row in rows):
            self._unknown(method, "stale or incompatible completed receipts")
        return rows

    def idle(self):
        metrics = self.client.request("/metrics", method="GET")
        lines = metrics.splitlines() if isinstance(metrics, str) else []
        values = [
            float(line.split()[-1])
            for line in lines
            if line.startswith(("vllm:num_requests_running{", "vllm:num_requests_waiting{"))
        ]
        if not values or any(not math.isfinite(value) or value != 0 for value in values):
            raise RuntimeError("active or unknown requests; abort before mutation")
        self.record("idle_verified")

    def check_workers(self, rows, *, initial=False, clean=True, allow_inconsistent=False):
        expected = {tuple(row[field] for field in SOURCE_FIELDS) for row in self.reference["runtime"]["workers"]}
        if {tuple(row.get(field) for field in SOURCE_FIELDS) for row in rows} != expected:
            raise RuntimeError("worker/checkpoint storage identity changed")
        if (
            not allow_inconsistent
            and len({(row.get("generation"), row.get("digest"), row.get("candidate")) for row in rows}) != 1
        ):
            raise RuntimeError("ranks disagree about installed generation")
        if initial and (
            any(row.get("generation") != self.expected_generation for row in rows)
            or any(row.get("digest") != self.reference["replacement_source"]["sha256"] for row in rows)
        ):
            raise RuntimeError("expected reference generation/source changed")
        if any(row.get("native_failed") for row in rows) or (
            clean and any(row.get("graphs_dirty") is not False for row in rows)
        ):
            raise RuntimeError("native registry failed or graph capture is incomplete")
        if any(row.get("fp16_scale_storage", {}).get("decode_version") != 984 for row in rows):
            raise RuntimeError("decode v984 changed")
        self.last_workers = rows
        return rows

    @contextmanager
    def lease(self):
        if urlparse(self.client.base_url).hostname not in ("127.0.0.1", "localhost", "::1"):
            raise ValueError("resident mutation must use loopback on the NPU host")
        with self.lock_path.open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            try:
                self.idle()
                if self.client.request("/resident/maintenance", method="GET").get("maintenance") is not False:
                    raise RuntimeError("another controller owns maintenance")
                if self.client.request("/is_paused", method="GET").get("is_paused") is not False:
                    raise RuntimeError("server was already paused; do not take ownership")
                self.check_workers(self.rpc("resident_status"), initial=True)
                self.record("maintenance_intent")
                receipt = self.client.request("/resident/maintenance", method="POST")
                if receipt.get("maintenance") is not True:
                    raise RuntimeError("maintenance acquisition was not acknowledged")
                self.owned = True
                self.journal["owned_maintenance"] = True
                self.record("maintenance_acquired")
                self.idle()  # Admission race: abort before pause/reset/cache clearing.
                yield self
            except BaseException as error:
                self.record("phase_error", error=repr(error))
                if self.owned and not self.unresolved and (self.changed or self.native_attempted):
                    try:
                        self.restore()
                    except BaseException as recovery:
                        self.journal["state"] = "recovery_failed"
                        self.journal["recovery_error"] = repr(recovery)
                        self.record("recovery_failed", error=repr(recovery))
                        raise RuntimeError(f"phase failed and safe recovery failed: {recovery}") from error
                raise
            finally:
                if self.owned and not self.unresolved and not self.journal["recovery_error"]:
                    try:
                        if self.journal["paused"]:
                            self.resume()
                        self.client.request("/resident/maintenance", method="DELETE")
                        self.owned = False
                        self.journal["owned_maintenance"] = False
                        self.record("maintenance_released")
                    except BaseException as error:
                        self.journal["state"] = "recovery_failed"
                        self.journal["recovery_error"] = repr(error)
                        self.record("maintenance_release_failed", error=repr(error))
                        raise
                fcntl.flock(lock, fcntl.LOCK_UN)

    def pause_clear(self):
        if not self.owned:
            raise RuntimeError("no owned maintenance lease")
        self.idle()
        self.record("pause_cache_clear_intent")
        try:
            receipt = self.client.request("/pause?mode=wait&clear_cache=true")
        except Exception as error:
            self._unknown("pause", repr(error))
        if receipt.get("status") != "paused":
            self._unknown("pause", "pause was not acknowledged")
        self.journal["paused"] = True
        self.record("paused")

    def resume(self):
        self.check_workers(self.rpc("resident_status"))
        self.record("resume_intent")
        try:
            self.client.request("/resume")
        except Exception as error:
            self._unknown("resume", repr(error))
        self.journal["paused"] = False
        self.record("resumed")

    def load(self):
        if self.candidate is None:
            raise ValueError("candidate requires complete native gates")
        self.pause_clear()
        before = self.check_workers(self.rpc("resident_status"))
        demand = self.candidate.memory_admission["required_free_bytes"]
        if any(row.get("packed_state", {}).get("memory", {}).get("free", -1) < demand for row in before):
            raise RuntimeError("observed rank headroom cannot admit gated old/new/shadow peak")
        manifest = self.candidate.manifest
        self.rpc("resident_native_prepare", manifest.payload, expected={"native_digest": manifest.digest})
        self.native_attempted = True
        self.record("native_load_intent", native_name=manifest.name, native_digest=manifest.digest)
        loaded = self.rpc("resident_native_load", manifest.digest, expected={"native_digest": manifest.digest})
        if any(row.get("validation", {}).get("passed") is not True for row in loaded):
            raise CompletedRPCError("resident_native_load_validation", loaded)
        self.check_workers(self.rpc("resident_status"))
        self.record("native_loaded", memory_admission=self.candidate.memory_admission)

    def source(self, variant, *, shadow=False, profile_dir=None):
        source = (self.reference_dir / self.reference["replacement_source"]["path"]).read_text()
        if variant == "reference":
            version = 1021
            root = Path(
                next(iter(self.reference["runtime"]["workers"][0]["native_launch_audit"]["binaries"].values()))["path"]
            ).parent
        else:
            if self.candidate is None:
                raise ValueError("candidate gates are required")
            version, root = self.candidate.identity["version"], self.candidate.build_dir
        inventory = {str(row["rank"]): row["banks"] for row in self.reference["bank_inventory"]}
        text = compose_source(
            source,
            source_sha256=self.reference["replacement_source"]["sha256"],
            version=version,
            build_dir=root,
            expected_inventory=inventory,
            shadow=shadow,
            profile_dir=profile_dir,
        )
        name = f"{variant}_v{version}" + ("_shadow" if shadow else "_profile" if profile_dir else "_ordinary")
        path = self.output_dir / f"{name}.py.txt"
        path.write_text(text)
        self.sources[name] = {"path": str(path), "sha256": hashlib.sha256(text.encode()).hexdigest()}
        return text, name

    def switch(self, source, name, *, recovering=False):
        self.pause_clear()
        before = self.check_workers(self.rpc("resident_status"), clean=False, allow_inconsistent=recovering)
        control = Control(uuid.uuid4().hex, candidate="streaming_" + name, source=source)
        self.record("prepare_source_intent", candidate=control.candidate, source_sha256=control.digest)
        prepared = self.rpc(
            "resident_prepare",
            json.dumps(dataclasses.asdict(control)),
            expected={"generation": control.generation, "digest": control.digest},
            rankless=True,
        )
        if len({tuple(row.get("targets", [])) for row in prepared}) != 1:
            raise RuntimeError("ranks prepared different replacement targets")
        self.changed = True
        self.journal["changed_dispatch"] = True
        self.record("dispatch_mutation_intent", candidate=control.candidate, generation=control.generation)
        self.rpc("resident_reset")
        expected = {
            "generation": control.generation,
            "digest": control.digest,
            "candidate": control.candidate,
            "mode": "graph",
        }
        self.rpc("resident_apply", control.generation, expected=expected)
        self.rpc("resident_apply", control.generation, expected=expected)
        self.rpc("resident_capture", expected={**expected, "graphs_dirty": False})
        rows = self.rpc("resident_reset", expected={**expected, "graphs_dirty": False})
        self.check_workers(rows)
        if {tuple(row[field] for field in SOURCE_FIELDS) for row in before} != {
            tuple(row[field] for field in SOURCE_FIELDS) for row in rows
        }:
            raise RuntimeError("swap changed loaded checkpoint storage")
        self.expected_generation = control.generation
        self.record("switched", workers=rows)
        self.resume()
        return rows

    def select(self, variant, *, shadow=False, profile_dir=None):
        source, name = self.source(variant, shadow=shadow, profile_dir=profile_dir)
        return self.switch(source, name)

    def restore(self):
        if self.unresolved:
            raise UnresolvedOperation("cannot restore while a collective is unresolved")
        self.idle()
        rows = self.rpc("resident_status")
        if any(row.get("native_failed") for row in rows):
            raise RuntimeError("append-only native registry is poisoned; cannot safely resume")
        source = (self.reference_dir / self.reference["replacement_source"]["path"]).read_text()
        if hashlib.sha256(source.encode()).hexdigest() != self.reference["replacement_source"]["sha256"]:
            raise ValueError("recovery reference source changed")
        if any(row.get("streaming", {}).get("profiler_active") for row in rows):
            self.pause_clear()
            self.rpc("profile", False, None)
        self.switch(source, "qualified_reference_v1021", recovering=True)
        self.changed = False
        self.journal["changed_dispatch"] = False
        self.record("reference_restored")

    def audit(self, rows, variant):
        expected = self.reference["binaries"] if variant == "reference" else self.candidate.identity["binaries"]
        version = 1021 if variant == "reference" else self.candidate.identity["version"]
        for row in rows:
            state = row.get("streaming", {})
            audit = state.get("native_launch_audit", {})
            if (
                state.get("version") != version
                or state.get("decode_version") != 984
                or audit.get("operator") != f"glm_reconstruction_v{version}.launch"
            ):
                raise ValueError("actual operator/dispatch differs from the requested candidate")
            if any(
                expected.get(Path(binary["path"]).name) != binary["sha256"]
                for binary in audit.get("binaries", {}).values()
            ) or not audit.get("binaries"):
                raise ValueError("actual binary hashes differ from native gates")
        return rows

    def request_workload(self, workload, *, variant, cold=True, concurrency=None):
        if cold:
            self.pause_clear()
            self.resume()
        self.idle()
        before = self.audit(self.rpc("resident_status"), variant)
        payload = copy.deepcopy(workload.get("input", workload.get("generation", {})))
        payload["model"] = self.reference["runtime"]["settings"]["model_name"]
        if "prompt_token_ids" in workload:
            payload.update(prompt=workload["prompt_token_ids"], max_tokens=workload["max_tokens"])
        count = concurrency or workload.get("concurrency", 1)

        def run(index):
            return self.client.stream(workload["endpoint"], payload, "glm-streaming-" + uuid.uuid4().hex)

        try:
            if count > 1:
                with ThreadPoolExecutor(max_workers=count) as pool:
                    samples = list(pool.map(run, range(count)))
            else:
                samples = [run(0)]
        except (TimeoutError, OSError) as error:
            self._unknown("inference", repr(error))
        self.idle()
        after = self.audit(self.rpc("resident_status"), variant)
        previous = {event["request_id"] for event in before[0]["streaming"].get("progress", [])}
        progress = [event for event in after[0]["streaming"].get("progress", []) if event["request_id"] not in previous]
        for sample in samples:
            ids = workload.get("prompt_token_ids")
            expected_count = len(ids) if ids is not None else workload.get("materialized_prompt_tokens")
            if expected_count is not None and sample["prompt_tokens"] != expected_count:
                raise ValueError("serving used a different rendered prompt length")
            if ids is not None and sample["completion_tokens"] != workload["max_tokens"]:
                raise ValueError("synthetic request did not complete its frozen generation length")
            matched = [
                event
                for event in progress
                if sample.get("response_id") and event["request_id"].startswith(sample["response_id"])
            ]
            if not matched:
                raise ValueError("cannot correlate completed request with CPU prompt-progress evidence")
            reused = matched[0]["reused_at_admission"]
            if cold and workload["id"].startswith("cold_") and reused != 0:
                raise ValueError("cold request reused a prefix")
            sample.update(
                variant=variant,
                workload=workload["id"],
                prompt_sha256=workload.get("prompt_sha256", workload.get("materialized_prompt_sha256")),
                reused_at_admission=reused,
                scheduled_progress=matched,
                completed_uncached_prompt_tokens=sample["prompt_tokens"] - reused if isinstance(reused, int) else None,
                completion_evidence="SSE DONE+usage; scheduled progress alone is not device completion",
            )
        self.record("workload_completed", workload=workload["id"], variant=variant, samples=samples, workers=after)
        return samples, after


def _write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def _report(controller, kind, *, coverage, result, errors=None):
    errors = errors or []
    raw_path = controller.output_dir / f"{kind}-raw.json"
    _write_json(raw_path, {"result": result, "journal": controller.journal, "sources": controller.sources})
    report = {
        "schema_version": 1,
        "kind": kind,
        "state": "failed" if errors else "complete",
        "identity": controller.candidate.identity,
        "dispatch": controller.candidate.dispatch,
        "coverage": sorted(set(coverage)),
        "artifacts": [
            {"path": raw_path.name, "sha256": file_digest(raw_path)},
            *[event["data_artifact"] for event in controller.journal["events"] if "data_artifact" in event],
            *[
                {"path": Path(source["path"]).name, "sha256": source["sha256"]}
                for source in controller.sources.values()
            ],
        ],
        "errors": errors,
        "result": result,
    }
    validate_evidence(report, kind=kind, require_complete=not errors, artifact_root=controller.output_dir)
    _write_json(controller.output_dir / ("shadow.json" if kind == "shadow" else "trials.json"), report)
    return report


def _audit_shadow_workers(controller, rows):
    cases, coverage = [], []
    for row in rows:
        rank, state = row["rank"], row["streaming"]
        wanted = next(item["banks"] for item in controller.reference["bank_inventory"] if item["rank"] == rank)
        seen = set()
        for case in state["cases"]:
            ordinal, tokens = case["bank"], case["tokens"]
            key = (ordinal, tokens)
            if not 0 <= ordinal < 43 or key in seen or tokens not in (640, 1280):
                raise ValueError("shadow has duplicate or unexpected bank/chunk cases")
            seen.add(key)
            if (
                case["bank_pointer"] != wanted[ordinal]["pointer"]
                or case["bits"] != wanted[ordinal]["bits"]
                or case["experts"] != 72
                or case["hidden"] != 4096
                or case["intermediate"] != 2048
                or case["finite"] is not True
                or case["bitwise_equal"] is not True
            ):
                raise ValueError("actual shadow bank/geometry/parity differs from reference")
            coverage.append(f"rank{rank}_bank{ordinal}_chunk{tokens}")
            cases.append({"rank": rank, **case})
        if seen != {(bank, tokens) for bank in range(43) for tokens in (640, 1280)} or state["mismatches"] != 0:
            raise ValueError("shadow lacks every required bank/chunk comparison")
        whole = state["whole_moe"]
        if len(whole) != 86 or any(case["finite"] is not True or case["bitwise_equal"] is not True for case in whole):
            raise ValueError("shared-expert/HCCL whole-MoE comparisons failed")
        audit = state["native_launch_audit"]
        counts = audit["counts"]
        for stage in ("pack_kernel", "route_input_kernel", "reduce_kernel"):
            if counts.get(stage) != 86:
                raise ValueError("actual shadow stage submissions do not cover all MoE inputs")
        for prefix in ("gate", "down"):
            if (
                sum(counts.get(key, 0) for key in (f"{prefix}_kernel", f"{prefix}_w3_kernel", f"{prefix}_w4_kernel"))
                != 86
            ):
                raise ValueError("actual gate/down launches do not match shadow coverage")
        if row.get("reconstruction", {}).get("fallback_dispatches", -1) != 0:
            raise ValueError("shadow used an unqualified fallback")
    if set(coverage) != set(shadow_case_ids()):
        raise ValueError("shadow rank coverage is incomplete")
    return cases, coverage


def shadow_workflow(controller, workloads):
    rows = {row["id"]: row for row in workloads["workloads"]}
    result, coverage = {}, []
    try:
        with controller.lease():
            controller.load()
            controller.select("candidate", shadow=True)
            sample, workers = controller.request_workload(rows["shadow_1920"], variant="candidate")
            if sample[0]["reused_at_admission"] != 0:
                raise ValueError("shadow input must be cold 1920 tokens")
            cases, coverage = _audit_shadow_workers(controller, workers)
            result.update(cases=cases, shadow_request=sample, shadow_workers=workers)
            # Retire shadow hooks/reference scratch only after the full request
            # has completed and before ordinary graph recapture/timing.
            controller.select("candidate")
            changed = copy.deepcopy(rows["decode_c1"])
            changed["prompt_token_ids"][-1] += 1
            changed["prompt_sha256"] = canonical_digest(changed["prompt_token_ids"])
            decode_first, _ = controller.request_workload(rows["decode_c1"], variant="candidate")
            decode_changed, _ = controller.request_workload(changed, variant="candidate")
            prime, _ = controller.request_workload(rows["shadow_1920"], variant="candidate")
            warm, _ = controller.request_workload(rows["shadow_1920"], variant="candidate", cold=False)
            if warm[0]["reused_at_admission"] <= 0:
                raise ValueError("qualified prefix/KDA checkpoint was not reused")
            result.update(decode_controls=decode_first + decode_changed, prefix_controls=prime + warm)
            controller.restore()
            result.update(
                bitwise_equal=True,
                finite=True,
                all_banks=True,
                actual_launch_audit=True,
                decode_replay=True,
                prefix_checkpoints=True,
                shared_expert_combine_hccl=True,
                comparisons=len(cases),
                final_workers=controller.last_workers,
                final_generation=controller.expected_generation,
            )
        return _report(controller, "shadow", coverage=coverage, result=result)
    except BaseException as error:
        _report(controller, "shadow", coverage=coverage, result=result, errors=[repr(error)])
        raise


def materialize_language_inputs(controller, workloads):
    frozen = copy.deepcopy(workloads)
    for workload in frozen["workloads"]:
        if not workload.get("materialize_token_ids_before_trials"):
            continue
        payload = workload["input"]
        tokenize = {
            "model": controller.reference["runtime"]["settings"]["model_name"],
            "messages": payload["messages"],
            "add_generation_prompt": True,
            "add_special_tokens": False,
        }
        for name in ("tools", "chat_template", "chat_template_kwargs", "continue_final_message"):
            if name in payload:
                tokenize[name] = payload[name]
        result = controller.client.request("/tokenize", tokenize)
        ids = result.get("tokens")
        if (
            not isinstance(ids, list)
            or not ids
            or not all(type(token) is int and token >= 0 for token in ids)
            or result.get("count") != len(ids)
        ):
            raise ValueError("renderer did not return complete token IDs")
        workload.update(
            materialized_token_ids=ids,
            materialized_prompt_tokens=len(ids),
            materialized_prompt_sha256=canonical_digest(ids),
            renderer_request=tokenize,
        )
    _write_json(controller.output_dir / "materialized-workloads.json", frozen)
    return frozen


def _quality(name, sample):
    if name == "tool":
        for call in sample.get("tool_calls", []):
            try:
                if call["name"] == "lookup_order" and json.loads(call["arguments"]) == {"order_id": "ORD-2048"}:
                    return True
            except (ValueError, KeyError):
                continue
        return False
    # A prose explanation may precede the requested JSON object.
    for match in re.finditer(r"\{[^{}]*\}", sample.get("text", "")):
        try:
            result = json.loads(match.group())
            if result.get("remaining_jobs") == 3 and result.get("worker_count") == 3:
                return True
        except ValueError:
            continue
    return False


def _prepare_variant(controller, variant, rows, protocol):
    controller.select(variant)
    for name in protocol["warmup"]["per_install"]:
        controller.request_workload(rows[name], variant=variant)
    controller.pause_clear()
    controller.resume()


def _mixed_workload(controller, rows, variant):
    controller.pause_clear()
    controller.resume()
    controller.idle()
    before = controller.audit(controller.rpc("resident_status"), variant)

    generated = threading.Event()

    def run(workload):
        payload = {
            **copy.deepcopy(workload["generation"]),
            "model": controller.reference["runtime"]["settings"]["model_name"],
            "prompt": workload["prompt_token_ids"],
            "max_tokens": workload["max_tokens"],
        }
        return controller.client.stream(
            workload["endpoint"],
            payload,
            "glm-streaming-" + uuid.uuid4().hex,
            on_first_token=generated.set if workload["id"] == "decode_c1" else None,
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        decode = pool.submit(run, rows["decode_c1"])
        deadline = time.monotonic() + 30
        observed = False
        while not decode.done() and time.monotonic() < deadline:
            metrics = controller.client.request("/metrics", method="GET")
            active = [
                float(line.split()[-1])
                for line in metrics.splitlines()
                if line.startswith("vllm:num_requests_running{")
            ]
            if generated.is_set() and active and any(value > 0 for value in active):
                observed = True
                break
            time.sleep(0.05)
        if not observed:
            decode.result()
            raise ValueError("mixed control did not observe a live decode request")
        prefill = pool.submit(run, rows["mixed"])
        try:
            samples = [decode.result(), prefill.result()]
        except (TimeoutError, OSError) as error:
            controller._unknown("mixed_inference", repr(error))
    controller.idle()
    after = controller.audit(controller.rpc("resident_status"), variant)
    previous = {event["request_id"] for event in before[0]["streaming"].get("progress", [])}
    progress = [event for event in after[0]["streaming"].get("progress", []) if event["request_id"] not in previous]
    for sample, workload in zip(samples, (rows["decode_c1"], rows["mixed"])):
        matched = [
            event
            for event in progress
            if sample["response_id"] and event["request_id"].startswith(sample["response_id"])
        ]
        if (
            not matched
            or sample["prompt_tokens"] != workload["prompt_tokens"]
            or sample["completion_tokens"] != workload["max_tokens"]
        ):
            raise ValueError("mixed control used incomplete or unidentified prompt work")
        sample.update(
            variant=variant,
            workload="mixed",
            role=workload["id"],
            prompt_sha256=workload["prompt_sha256"],
            reused_at_admission=matched[0]["reused_at_admission"],
            scheduled_progress=matched,
        )
    controller.record(
        "mixed_completed", variant=variant, samples=samples, observed_decode_before_prefill=True, workers=after
    )
    return samples


def trials_workflow(controller, workloads, shadow_report, *, cycles=3, promote=False):
    if controller.candidate.experimental:
        raise ValueError("experimental diagnostics cannot qualify complete trials or promotion")
    validate_evidence(shadow_report, kind="shadow", bindings=controller.candidate.identity)
    if cycles < 3:
        raise ValueError("primary protocol requires at least three ABBA cycles")
    result, coverage, samples = {}, [], []
    try:
        with controller.lease():
            controller.load()
            frozen = materialize_language_inputs(controller, workloads)
            rows = {row["id"]: row for row in frozen["workloads"]}
            primary = []
            for cycle in range(cycles):
                for position, variant in enumerate(ABBA):
                    _prepare_variant(controller, variant, rows, frozen)
                    completed, _ = controller.request_workload(rows[PRIMARY_WORKLOAD], variant=variant)
                    completed[0].update(cycle=cycle, position=position)
                    primary += completed
                    samples += completed
            repeatability = summarize_trials(primary)
            coverage.append(PRIMARY_WORKLOAD)
            secondary = {}
            for variant in ("reference", "candidate"):
                _prepare_variant(controller, variant, rows, frozen)
                variant_samples = []
                for name in ("cold_640", "cold_32768", "decode_c1", "decode_c4", "language", "tool"):
                    repeats = 3 if name in ("decode_c1", "decode_c4") else 1
                    for _ in range(repeats):
                        completed, _ = controller.request_workload(rows[name], variant=variant)
                        variant_samples += completed
                    if name not in coverage:
                        coverage.append(name)
                prime, _ = controller.request_workload(rows["warm_prefix_6400"], variant=variant)
                warm, _ = controller.request_workload(rows["warm_prefix_6400"], variant=variant, cold=False)
                if warm[0]["reused_at_admission"] <= 0:
                    raise ValueError("warm prefix control did not reuse qualified state")
                variant_samples += prime + warm
                for depth in frozen["long_context"]["existing_context_depths"]:
                    prompt = [261] + [43] * max(0, depth - 1)
                    if depth:
                        context = copy.deepcopy(rows["cold_640"])
                        context.update(
                            id=f"context_prime_{depth}",
                            prompt_token_ids=prompt,
                            prompt_tokens=len(prompt),
                            prompt_sha256=canonical_digest(prompt),
                        )
                        controller.request_workload(context, variant=variant)
                        combined = prompt + [43] * 639 + [361]
                    else:
                        combined = rows["cold_640"]["prompt_token_ids"]
                    append = copy.deepcopy(rows["cold_640"])
                    append.update(
                        id=f"context_depth_{depth}",
                        prompt_token_ids=combined,
                        prompt_tokens=len(combined),
                        prompt_sha256=canonical_digest(combined),
                    )
                    completed, _ = controller.request_workload(append, variant=variant, cold=not bool(depth))
                    variant_samples += completed
                for _ in range(3):
                    variant_samples += _mixed_workload(controller, rows, variant)
                secondary[variant] = variant_samples
                samples += variant_samples
            coverage += ["warm_prefix_6400", "mixed"]
            regressions = {}
            for name in ("decode_c1", "decode_c4", "mixed"):
                rates = {
                    variant: [
                        sample["generation_tok_s"]
                        for sample in secondary[variant]
                        if sample["workload"] == name and sample["generation_tok_s"] is not None
                    ]
                    for variant in secondary
                }
                if not all(rates.values()):
                    raise ValueError("decode/mixed control lacks generated-token throughput")
                regressions[name] = 1 - statistics.median(rates["candidate"]) / statistics.median(rates["reference"])
            quality = {
                variant: {
                    name: all(_quality(name, sample) for sample in secondary[variant] if sample["workload"] == name)
                    for name in ("language", "tool")
                }
                for variant in secondary
            }
            eligible = (
                repeatability["repeatability_passed"]
                and max(regressions.values()) <= 0.05
                and all(all(values.values()) for values in quality.values())
            )
            result.update(
                samples=samples,
                primary=primary,
                repeatability=repeatability,
                regressions=regressions,
                language_tool_quality=quality,
                promotion_eligible=eligible,
                objective_10pct_met=repeatability["objective_10pct_met"],
                requests_complete=True,
                launch_audit=True,
                identical_abba_inputs=True,
                shadow_removed=True,
                profiler_disabled=True,
                timing_semantics_recorded=True,
            )
            if promote and eligible:
                controller.select("candidate")
                controller.changed = False  # Explicit measured promotion; final graphs/status still checked.
                result["retained"] = "candidate"
            else:
                controller.restore()
                result["retained"] = "reference"
            result.update(final_workers=controller.last_workers, final_generation=controller.expected_generation)
        return _report(controller, "trials", coverage=coverage, result=result)
    except BaseException as error:
        result["samples"] = samples
        _report(controller, "trials", coverage=coverage, result=result, errors=[repr(error)])
        raise


def profile_workflow(controller, workloads):
    """Attribution is a separate instrumented request, never a timing sample."""
    rows = {row["id"]: row for row in workloads["workloads"]}
    with controller.lease():
        controller.load()
        controller.select("candidate", profile_dir=str((controller.output_dir / "profile").resolve()))
        controller.pause_clear()
        controller.rpc("profile", True, "streaming-cold-prefill")
        controller.resume()
        samples, _ = controller.request_workload(rows[PRIMARY_WORKLOAD], variant="candidate", cold=False)
        controller.pause_clear()
        controller.rpc("profile", False, None)
        controller.select("candidate")  # Remove instrumentation after profiler stop.
        controller.restore()
    report = {
        "scope": "attribution only; excluded from latency comparisons",
        "samples": samples,
        "journal": controller.journal,
        "profiler_stopped": True,
        "final_workers": controller.last_workers,
    }
    _write_json(controller.output_dir / "profile.json", report)
    return report


def main():
    """Validate evidence and run one explicitly selected resident phase."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", nargs="?", default="status", choices=("status", "shadow", "trials", "profile", "install", "restore")
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8001")
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--gates", type=Path)
    parser.add_argument("--build-dir", type=Path)
    parser.add_argument("--shadow-report", type=Path)
    parser.add_argument("--workloads", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--expected-generation")
    parser.add_argument("--cycles", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=RPC_TIMEOUT)
    parser.add_argument(
        "--experimental",
        action="store_true",
        help="User-authorized pending diagnostics; never full qualification/promotion",
    )
    parser.add_argument("--execute", action="store_true", help="Permit mutation after idle lease and native admission")
    parser.add_argument("--promote", action="store_true", help="Retain only if all measured promotion rules pass")
    args = parser.parse_args()
    client = StreamingTransport(args.base_url, expected_workers=WORKERS, timeout=args.timeout)
    if args.command == "status":
        print(
            json.dumps(
                {
                    "maintenance": client.request("/resident/maintenance", method="GET"),
                    "paused": client.request("/is_paused", method="GET"),
                    "workers": client.rpc("resident_status"),
                },
                indent=2,
            )
        )
        return
    if args.reference is None or args.output_dir is None:
        parser.error("phase commands require --reference and --output-dir")
    reference = verify_reference(json.loads(args.reference.read_text()), args.reference.parent)
    if args.command != "restore" and args.gates is None:
        parser.error("candidate phases require --gates (complete unless --experimental)")
    candidate = (
        load_candidate(reference, args.gates, build_dir=args.build_dir, experimental=args.experimental)
        if args.gates
        else None
    )
    workloads_path = args.workloads or args.reference.parent / "workloads.json"
    workloads = json.loads(workloads_path.read_text())
    shadow = None
    generation = args.expected_generation
    if args.experimental and (args.command == "trials" or args.promote):
        parser.error("experimental installation does not permit complete trials or promotion")
    if args.command == "trials":
        if args.shadow_report is None:
            parser.error("trials requires complete --shadow-report")
        shadow = json.loads(args.shadow_report.read_text())
        validate_evidence(shadow, kind="shadow", bindings=candidate.identity, artifact_root=args.shadow_report.parent)
        generation = generation or shadow["result"]["final_generation"]
    if not args.execute:
        print(
            json.dumps(
                {
                    "phase": args.command,
                    "dry_run": True,
                    "reference_sha256": reference["reference_sha256"],
                    "candidate": candidate.identity if candidate else None,
                    "workload_ids": [row["id"] for row in workloads["workloads"]],
                    "requires": "native admission, idle owned lease, exact generation and whole-rank memory proof",
                }
            )
        )
        return
    controller = GuardedResidentController(
        client, reference, args.reference.parent, args.output_dir, candidate=candidate, expected_generation=generation
    )
    if args.command == "shadow":
        report = shadow_workflow(controller, workloads)
    elif args.command == "trials":
        report = trials_workflow(controller, workloads, shadow, cycles=args.cycles, promote=args.promote)
    elif args.command == "profile":
        report = profile_workflow(controller, workloads)
    else:
        with controller.lease():
            if args.command == "restore":
                controller.restore()
            else:
                controller.load()
                controller.select("candidate")
                controller.changed = False
        report = {"phase": args.command, "final_workers": controller.last_workers, "journal": controller.journal}
    print(
        json.dumps(
            {
                "phase": args.command,
                "state": report.get("state", "complete"),
                "output_dir": str(args.output_dir),
                "experimental": bool(candidate and candidate.experimental),
                "native_qualification": "pending" if candidate and candidate.experimental else "complete",
            }
        )
    )


if __name__ == "__main__":
    main()
