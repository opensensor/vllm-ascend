# SPDX-License-Identifier: Apache-2.0
"""CPU fault injection and complete workflow tests for resident streaming."""

import copy
import hashlib
import json
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
import regex as re

from tools.glm_perf.resident_control import Control
from tools.glm_perf.streaming_resident import GuardedResidentController, UnresolvedOperation, profile_workflow

FROZEN_SOURCE = """def replacements(native_resources):
    return {"preserved":native_resources}

import sys,types
_progress_module=object()
VERSION=1021
# The old tail must not be executed by a streaming successor.
"""


def make_reference():
    workers = [
        {
            "rank": rank,
            "pid": 100 + rank,
            "weight_storage_digest": str(rank) * 64,
            "generation": "a" * 32,
            "digest": hashlib.sha256(FROZEN_SOURCE.encode()).hexdigest(),
            "candidate": "qualified_v1021",
            "graphs_dirty": False,
            "native_failed": False,
            "mode": "graph",
            "fp16_scale_storage": {"decode_version": 984, "prefill_version": 1021},
        }
        for rank in range(4)
    ]
    return {
        "reference_sha256": "a" * 64,
        "runtime": {"workers": workers, "settings": {"model_name": "glm"}},
        "replacement_source": {"path": "reference.py", "sha256": workers[0]["digest"]},
        "bank_inventory": [
            {"rank": rank, "banks": [{"pointer": 10000 * rank + i, "bits": 2 + i % 3} for i in range(43)]}
            for rank in range(4)
        ],
    }


class FakeClient:
    base_url = "http://127.0.0.1:8001"
    timeout = 30

    def __init__(self, reference, *, fault=None, active=False, admission_race=False):
        self.workers = copy.deepcopy(reference["runtime"]["workers"])
        self.calls = []
        self.fault = fault
        self.active = active
        self.admission_race = admission_race
        self.paused = self.maintenance = False
        self.pending = None
        self.before_rpc = None
        self.failures = 0

    def request(self, path, payload=None, method="POST"):
        operation = payload["method"] if path == "/collective_rpc" else path
        self.calls.append((operation, method))
        if path == "/metrics":
            return (
                f'vllm:num_requests_running{{engine="0"}} {int(self.active)}\n'
                'vllm:num_requests_waiting{engine="0"} 0\n'
            )
        if path == "/resident/maintenance":
            if method == "POST":
                self.maintenance = True
                self.active = self.admission_race
            elif method == "DELETE":
                self.maintenance = False
            return {"maintenance": self.maintenance}
        if path == "/is_paused":
            return {"is_paused": self.paused}
        if path.startswith("/pause"):
            assert not self.active
            self.paused = True
            return {"status": "paused"}
        if path == "/resume":
            self.paused = False
            return {"status": "resumed"}
        if path != "/collective_rpc":
            raise AssertionError(path)
        if self.before_rpc:
            self.before_rpc(operation)
        if self.fault == (operation, "timeout") and not self.failures:
            self.failures += 1
            raise TimeoutError("operation still outstanding")
        if operation == "resident_status":
            return {"results": copy.deepcopy(self.workers)}
        assert self.paused
        if operation == "resident_prepare":
            self.pending = Control.from_dict(json.loads(payload["args"][0]))
            return {
                "results": [
                    {"generation": self.pending.generation, "digest": self.pending.digest, "targets": ["preserved"]}
                    for _ in range(4)
                ]
            }
        if operation == "resident_apply":
            for row in self.workers:
                row.update(
                    generation=self.pending.generation,
                    digest=self.pending.digest,
                    candidate=self.pending.candidate,
                    graphs_dirty=True,
                    mode="graph",
                )
            if self.fault == (operation, "completed_partial") and not self.failures:
                self.failures += 1
                self.workers[2]["generation"] = "a" * 32
                rows = copy.deepcopy(self.workers)
                rows[2] = {"rank": 2, "pid": 102, "error": "completed peer error"}
                return {"results": rows}
        if operation == "resident_capture":
            for row in self.workers:
                row["graphs_dirty"] = False
        rows = copy.deepcopy(self.workers)
        if self.fault == (operation, "missing") and not self.failures:
            self.failures += 1
            rows.pop()
        if self.fault == (operation, "stale") and not self.failures:
            self.failures += 1
            for row in rows:
                row["generation"] = "b" * 32
        return {"results": rows}


def controller(tmp_path, *, fault=None, active=False, admission_race=False):
    reference = make_reference()
    directory = tmp_path / "reference"
    directory.mkdir()
    (directory / "reference.py").write_text(FROZEN_SOURCE)
    client = FakeClient(reference, fault=fault, active=active, admission_race=admission_race)
    guard = GuardedResidentController(
        client, reference, directory, tmp_path / "output", lock_path=tmp_path / "lease.lock"
    )
    return guard, client


def operations(client):
    return [name for name, _ in client.calls]


def test_read_only_lease_has_no_pause_clear_swap_or_resume(tmp_path):
    guard, client = controller(tmp_path)
    with guard.lease():
        assert guard.owned and client.maintenance
        guard.check_workers(guard.rpc("resident_status"))
    assert not client.maintenance and not client.paused
    assert "resident_apply" not in operations(client)
    assert "/resume" not in operations(client)
    assert not any(name.startswith("/pause") for name in operations(client))


def test_active_request_aborts_without_maintenance_or_collective(tmp_path):
    guard, client = controller(tmp_path, active=True)
    with pytest.raises(RuntimeError, match="active"), guard.lease():
        pytest.fail("must not enter")
    assert operations(client) == ["/metrics"]


def test_admission_race_releases_only_owned_maintenance_without_pause(tmp_path):
    guard, client = controller(tmp_path, admission_race=True)
    with pytest.raises(RuntimeError, match="active"), guard.lease():
        pytest.fail("must not enter")
    assert not client.maintenance
    assert ("/resident/maintenance", "DELETE") in client.calls
    assert not any(name.startswith("/pause") for name in operations(client))
    assert "/resume" not in operations(client)


def test_another_controller_maintenance_is_never_removed(tmp_path):
    guard, client = controller(tmp_path)
    client.maintenance = True
    with pytest.raises(RuntimeError, match="another controller"), guard.lease():
        pytest.fail("must not enter")
    assert client.maintenance
    assert ("/resident/maintenance", "DELETE") not in client.calls


def test_preexisting_pause_is_not_resumed(tmp_path):
    guard, client = controller(tmp_path)
    client.paused = True
    with pytest.raises(RuntimeError, match="already paused"), guard.lease():
        pytest.fail("must not enter")
    assert client.paused and "/resume" not in operations(client)


def test_stale_parent_generation_aborts_before_acquiring_maintenance(tmp_path):
    guard, client = controller(tmp_path)
    for row in client.workers:
        row["generation"] = "c" * 32
    with pytest.raises(RuntimeError, match="reference generation"), guard.lease():
        pytest.fail("must not enter")
    assert ("/resident/maintenance", "POST") not in client.calls


@pytest.mark.parametrize("kind", ["timeout", "missing", "stale"])
def test_uncertain_apply_never_sends_another_collective_or_restore_resume(tmp_path, kind):
    guard, client = controller(tmp_path, fault=("resident_apply", kind))
    with pytest.raises(UnresolvedOperation), guard.lease():
        guard.switch("candidate replacement", "candidate_v1102")
    calls = operations(client)
    assert calls[-1] == "resident_apply"
    assert calls.count("resident_prepare") == 1
    assert client.maintenance and client.paused
    assert guard.journal["state"] == "recovery_failed"
    assert guard.journal["outstanding"] is not None or guard.unresolved
    assert "/resume" not in calls
    assert ("/resident/maintenance", "DELETE") not in client.calls


def test_mutation_intent_is_durable_before_rpc_submission(tmp_path):
    guard, client = controller(tmp_path)

    def before(method):
        journal = json.loads((guard.output_dir / "controller-journal.json").read_text())
        assert journal["events"][-1]["event"] == "rpc_intent"
        assert journal["outstanding"]["method"] == method
        if method == "resident_apply":
            assert any(event["event"] == "dispatch_mutation_intent" for event in journal["events"])

    client.before_rpc = before
    with guard.lease():
        guard.switch("candidate replacement", "candidate_v1102")
        guard.restore()
    assert not client.maintenance and not client.paused


def test_completed_four_rank_partial_apply_can_restore_reference(tmp_path):
    guard, client = controller(tmp_path, fault=("resident_apply", "completed_partial"))
    with pytest.raises(RuntimeError, match="completed four-rank error"), guard.lease():
        guard.switch("candidate replacement", "candidate_v1102")
    assert not guard.unresolved
    assert operations(client).count("resident_prepare") == 2
    assert all(row["digest"] == guard.reference["replacement_source"]["sha256"] for row in client.workers)
    assert all(row["graphs_dirty"] is False for row in client.workers)
    assert not client.maintenance and not client.paused


def test_two_local_controllers_cannot_acquire_same_lease(tmp_path):
    guard, client = controller(tmp_path)
    other = GuardedResidentController(
        client, guard.reference, guard.reference_dir, tmp_path / "other", lock_path=guard.lock_path
    )
    with guard.lease(), pytest.raises(BlockingIOError), other.lease():
        pytest.fail("must not enter second lease")
    assert operations(client).count("resident_status") == 1


def sse(*events, done=True):
    frames = [b"data: " + json.dumps(event).encode() + b"\n" for event in events]
    if done:
        frames.append(b"data: [DONE]\n")
    return frames


def test_sse_reasoning_token_precedes_visible_text_and_ignores_role():
    from tools.glm_perf.streaming_resident import consume_sse

    clock = iter([1.0, 3.0, 4.0, 6.0, 7.0])
    result = consume_sse(
        sse(
            {"id": "req", "choices": [{"delta": {"role": "assistant"}, "logprobs": None}]},
            {"id": "req", "choices": [{"delta": {"reasoning_content": "thinking"}}]},
            {"id": "req", "choices": [{"delta": {"content": "answer"}}]},
            {"choices": [], "usage": {"prompt_tokens": 5, "completion_tokens": 3}},
        ),
        started=0.0,
        now=lambda: next(clock),
        request_id="req",
    )
    assert result["ttft_s"] == 3
    assert result["first_visible_text_s"] == 4
    assert result["reasoning"] == "thinking"
    assert result["generation_tok_s"] == 2
    assert result["completion_latency_s"] == 7
    assert result["timing_method"] == "streamed_token_event"


def test_sse_correlated_server_ttft_is_preferred_without_mixing_clocks():
    from tools.glm_perf.streaming_resident import consume_sse

    result = consume_sse(
        sse(
            {
                "id": "req",
                "choices": [{"text": "a"}],
                "server_timing": {
                    "request_id": "req",
                    "clock": "server_monotonic",
                    "admission_time": 1000,
                    "first_generated_token_time": 1002,
                },
                "usage": {"prompt_tokens": 3, "completion_tokens": 1},
            }
        ),
        started=50,
        now=lambda: 60,
        request_id="req",
    )
    assert result["ttft_s"] == 2
    assert result["streamed_ttft_s"] == 10
    assert result["timing_method"] == "server_generated_token"
    assert result["generation_tok_s"] is None


@pytest.mark.parametrize(
    "timing",
    [
        {"request_id": "other", "clock": "server", "admission_time": 100, "first_generated_token_time": 101},
        {"clock": "server", "admission_time": 100, "first_generated_token_time": 101},
        {"request_id": "req", "clock": "server", "admission_time": 100, "first_generated_token_time": float("inf")},
    ],
)
def test_sse_uncorrelated_or_nonfinite_server_timing_not_used(timing):
    from tools.glm_perf.streaming_resident import consume_sse

    result = consume_sse(
        sse(
            {
                "id": "req",
                "choices": [{"text": "a"}],
                "server_timing": timing,
                "usage": {"prompt_tokens": 3, "completion_tokens": 1},
            }
        ),
        started=0,
        now=lambda: 5,
        request_id="req",
    )
    assert result["ttft_s"] == 5 and result["timing_method"] == "streamed_token_event"


def test_tool_arguments_are_tokens_even_without_visible_text():
    from tools.glm_perf.streaming_resident import consume_sse

    result = consume_sse(
        sse(
            {
                "id": "req",
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "function": {"name": "lookup_order", "arguments": '{"order_id":"ORD-2048"}'},
                                }
                            ]
                        }
                    }
                ],
                "usage": {"prompt_tokens": 3, "completion_tokens": 4},
            }
        ),
        started=0,
        now=lambda: 5,
        request_id="req",
    )
    assert result["first_visible_text_s"] is None
    assert result["ttft_s"] == 5
    assert result["tool_calls"] == [{"name": "lookup_order", "arguments": '{"order_id":"ORD-2048"}'}]


@pytest.mark.parametrize(
    "events,done",
    [
        ([{"id": "req", "choices": [{"text": "a"}]}], True),
        ([{"usage": {"prompt_tokens": 1, "completion_tokens": 0}, "choices": []}], True),
        ([{"choices": [{"text": "a"}], "usage": {"prompt_tokens": 1, "completion_tokens": 1}}], False),
    ],
)
def test_incomplete_stream_is_not_a_latency_sample(events, done):
    from tools.glm_perf.streaming_resident import consume_sse

    with pytest.raises(ValueError):
        consume_sse(sse(*events, done=done), started=0, now=lambda: 5, request_id="req")


def test_source_composition_excludes_old_selector_audit_and_preserves_prefix():
    from tools.glm_perf.resident_candidates.streaming_experts import compose_source

    source = FROZEN_SOURCE + '\nraise AssertionError("old queue tail executed")\n'
    digest = hashlib.sha256(source.encode()).hexdigest()
    combined = compose_source(
        source, source_sha256=digest, version=1102, build_dir=Path("/build"), expected_inventory={}
    )
    namespace = {}
    exec(compile(combined, "<candidate>", "exec"), namespace)
    captured = {}

    def extend(parent, resources, **options):
        captured.update(options)
        return parent(resources)

    namespace["_streaming_helper"].extend_replacements = extend
    untouched = {"reconstruction_v984": {"fused_int4a4": object()}, "reconstruction_v1001": {"sinkhorn": object()}}
    assert namespace["replacements"](untouched) == {"preserved": untouched}
    assert captured["version"] == 1102
    assert "VERSION" not in namespace
    with pytest.raises(ValueError, match="changed"):
        compose_source(source, source_sha256="b" * 64, version=1102, build_dir=Path("/build"), expected_inventory={})


def rich_reference():
    reference = make_reference()
    files = [
        "glm_fused_pack.bin",
        "glm_fused_route_input.bin",
        "glm_fused_gate_up.bin",
        "glm_fused_down.bin",
        "glm_fused_gate_up_w4.bin",
        "glm_fused_down_w4.bin",
        "glm_fused_reduce.bin",
    ]
    reference["binaries"] = {filename: "b" * 64 for filename in files}
    for worker in reference["runtime"]["workers"]:
        worker["native_launch_audit"] = {
            "binaries": {"pack": {"path": "/reference/build/glm_fused_pack.bin", "sha256": "b" * 64}}
        }
        worker["packed_state"] = {"memory": {"free": 1000000}}
    return reference


class WorkflowClient(FakeClient):
    def __init__(self, reference):
        super().__init__(reference)
        self.reference = reference
        self.tokens = [11, 22]
        self.prefixes = set()
        self.progress = []
        self.version = 1021
        self.shadow = False
        self.loads = 0
        self.inference = []
        self.active_count = 0

    def request(self, path, payload=None, method="POST"):
        if path == "/tokenize":
            self.calls.append(("/tokenize", method))
            return {"tokens": self.tokens, "count": len(self.tokens)}
        if path == "/metrics":
            self.calls.append(("/metrics", method))
            return (
                f'vllm:num_requests_running{{engine="0"}} {self.active_count}\n'
                'vllm:num_requests_waiting{engine="0"} 0\n'
            )
        if path.startswith("/pause"):
            self.prefixes.clear()
        if path == "/collective_rpc" and payload["method"] in ("resident_native_prepare", "resident_native_load"):
            operation = payload["method"]
            self.calls.append((operation, method))
            if operation == "resident_native_load":
                self.loads += 1
            return {
                "results": [
                    {"rank": rank, "pid": 100 + rank, "native_digest": "d" * 64, "validation": {"passed": True}}
                    for rank in range(4)
                ]
            }
        result = super().request(path, payload, method)
        if path == "/collective_rpc" and payload["method"] == "resident_apply":
            versions = re.findall(r"version=(\d+)", self.pending.source)
            self.version = int(versions[-1]) if versions else 1021
            self.shadow = "shadow=True" in self.pending.source
            self.progress = []
            for row in self.workers:
                binaries = (
                    self.reference["binaries"]
                    if self.version == 1021
                    else {name: "c" * 64 for name in self.reference["binaries"]}
                )
                row["streaming"] = {
                    "version": self.version,
                    "decode_version": 984,
                    "shadow": self.shadow,
                    "cases": [],
                    "whole_moe": [],
                    "mismatches": 0,
                    "progress": [],
                    "profiler_active": False,
                    "native_launch_audit": {
                        "operator": f"glm_reconstruction_v{self.version}.launch",
                        "counts": {},
                        "binaries": {name: {"path": "/build/" + name, "sha256": sha} for name, sha in binaries.items()},
                    },
                }
                row["reconstruction"] = {"fallback_dispatches": 0}
            # Return fresh receipts after installing the fake source.
            return {"results": copy.deepcopy(self.workers)}
        return result

    def stream(self, path, payload, request_id, *, on_first_token=None):
        import threading
        import time

        self.active_count += 1
        try:
            prompt = payload.get("prompt", self.tokens)
            count = len(prompt)
            output = payload.get("max_tokens", 1) if "prompt" in payload else 4
            key = tuple(prompt)
            reused = min(count - 1, 1280) if key in self.prefixes else 0
            self.prefixes.add(key)
            response_id = "cmpl-" + request_id
            cursor = reused
            while cursor < count:
                chunk = min(1280, count - cursor)
                cursor += chunk
                self.progress.append(
                    {
                        "request_id": response_id + "-0",
                        "prompt_tokens": count,
                        "reused_at_admission": reused,
                        "prompt_chunk": chunk,
                        "scheduled_through": cursor,
                        "remaining_to_schedule": count - cursor,
                        "forward_returned": True,
                    }
                )
            if on_first_token is not None:
                on_first_token()
            if output == 128 and threading.current_thread() is not threading.main_thread():
                time.sleep(0.02)  # Make mixed-service overlap observable to the real controller loop.
            for row in self.workers:
                state = row["streaming"]
                state["progress"] = copy.deepcopy(self.progress)
                counts = state["native_launch_audit"]["counts"]
                for stage in ("pack_kernel", "route_input_kernel", "reduce_kernel", "gate_kernel", "down_kernel"):
                    counts[stage] = (
                        counts.get(stage, 0) + 86 if self.shadow and count == 1920 else counts.get(stage, 0) + 1
                    )
                if self.shadow and count == 1920:
                    inventory = self.reference["bank_inventory"][row["rank"]]["banks"]
                    state["cases"] = [
                        {
                            "bank": bank,
                            "bank_pointer": info["pointer"],
                            "bits": info["bits"],
                            "tokens": chunk,
                            "experts": 72,
                            "hidden": 4096,
                            "intermediate": 2048,
                            "finite": True,
                            "bitwise_equal": True,
                        }
                        for bank, info in enumerate(inventory)
                        for chunk in (1280, 640)
                    ]
                    state["whole_moe"] = [
                        {"tokens": chunk, "finite": True, "bitwise_equal": True}
                        for _ in range(43)
                        for chunk in (1280, 640)
                    ]
            latency = 52 if self.version == 1021 else 45
            text = '{"remaining_jobs":3,"worker_count":3}' if "messages" in payload else "x"
            tools = [{"name": "lookup_order", "arguments": '{"order_id":"ORD-2048"}'}] if payload.get("tools") else []
            sample = {
                "request_id": request_id,
                "response_id": response_id,
                "complete": True,
                "prompt_tokens": count,
                "completion_tokens": output,
                "usage": {"prompt_tokens": count, "completion_tokens": output},
                "ttft_s": latency,
                "streamed_ttft_s": latency,
                "timing_method": "streamed_token_event",
                "first_visible_text_s": latency,
                "completion_latency_s": latency + 10,
                "generation_tok_s": 10 if output > 1 else None,
                "text": text,
                "reasoning": "",
                "tool_calls": tools,
                "events": [],
            }
            self.inference.append(
                {"version": self.version, "prompt": list(prompt), "payload": copy.deepcopy(payload), "sample": sample}
            )
            return sample
        finally:
            self.active_count -= 1


def workflow_controller(tmp_path):
    from tools.glm_perf.streaming_resident import Candidate

    reference = rich_reference()
    directory = tmp_path / "reference"
    directory.mkdir()
    (directory / "reference.py").write_text(FROZEN_SOURCE)
    client = WorkflowClient(reference)
    manifest = SimpleNamespace(payload="{}", digest="d" * 64, name="reconstruction_v1102")
    identity = {
        "version": 1102,
        "namespace": "glm_reconstruction_v1102",
        "contract_sha256": "a" * 64,
        "reference_sha256": "a" * 64,
        "source_sha256": "a" * 64,
        "binaries": {name: "c" * 64 for name in reference["binaries"]},
    }
    dispatch = {
        "bulk_a4_gate": "streaming",
        "bulk_a4_down": "streaming",
        "sparse": "qualified_reference",
        "a8": "qualified_reference",
        "decode_version": 984,
    }
    candidate = Candidate(identity, dispatch, Path("/build"), "f" * 64, {"required_free_bytes": 10}, manifest)
    guard = GuardedResidentController(
        client, reference, directory, tmp_path / "output", candidate=candidate, lock_path=tmp_path / "lease.lock"
    )
    return guard, client


def test_complete_shadow_workflow_uses_all_banks_and_removes_hooks(tmp_path):
    from tools.glm_perf.streaming_protocol import make_workloads
    from tools.glm_perf.streaming_resident import shadow_workflow

    guard, client = workflow_controller(tmp_path)
    report = shadow_workflow(guard, make_workloads())
    assert report["state"] == "complete"
    assert len(report["coverage"]) == 344 and report["result"]["comparisons"] == 344
    assert report["result"]["shared_expert_combine_hccl"] is True
    assert not client.maintenance and not client.paused
    assert all(row["digest"] == guard.reference["replacement_source"]["sha256"] for row in client.workers)
    assert (guard.output_dir / "shadow.json").is_file()


def test_complete_trial_workflow_runs_abba_and_every_control_before_restore(tmp_path):
    from tools.glm_perf.streaming_protocol import evidence_schema, make_workloads
    from tools.glm_perf.streaming_resident import trials_workflow

    guard, client = workflow_controller(tmp_path)
    shadow = {
        "schema_version": 1,
        "kind": "shadow",
        "state": "complete",
        "identity": guard.candidate.identity,
        "dispatch": guard.candidate.dispatch,
        "coverage": evidence_schema()["x-required-coverage"]["shadow"],
        "artifacts": [{"path": "proof.json", "sha256": "a" * 64}],
        "errors": [],
        "result": {
            **{rule: True for rule in evidence_schema()["x-complete-result-rules"]["shadow"]},
            "comparisons": 344,
        },
    }
    report = trials_workflow(guard, make_workloads(), shadow)
    assert report["state"] == "complete"
    primary = report["result"]["primary"]
    assert len(primary) == 12
    assert [row["variant"] for row in primary] == ["reference", "candidate", "candidate", "reference"] * 3
    for cycle in range(3):
        assert len({row["prompt_sha256"] for row in primary if row["cycle"] == cycle}) == 1
    assert set(report["coverage"]) == set(evidence_schema()["x-required-coverage"]["trials"])
    assert report["result"]["promotion_eligible"] is True
    assert report["result"]["retained"] == "reference"
    assert not client.maintenance and not client.paused
    tokenize_position = operations(client).index("/tokenize")
    first_apply_position = operations(client).index("resident_apply")
    assert tokenize_position < first_apply_position


def test_recovery_failure_keeps_lease_and_pause_instead_of_resuming(tmp_path):
    guard, client = controller(tmp_path)
    original = client.request
    failures = 0

    def request(path, payload=None, method="POST"):
        nonlocal failures
        if path == "/collective_rpc" and payload["method"] == "resident_capture":
            failures += 1
            client.calls.append(("resident_capture", method))
            return {"results": [{"rank": rank, "pid": 100 + rank, "error": "capture failed"} for rank in range(4)]}
        return original(path, payload, method)

    client.request = request
    with pytest.raises(RuntimeError, match="safe recovery failed"), guard.lease():
        guard.switch("candidate replacement", "candidate_v1102")
    assert failures == 2
    assert client.maintenance and client.paused
    assert guard.journal["recovery_error"]
    assert ("/resident/maintenance", "DELETE") not in client.calls
    assert "/resume" not in operations(client)


def test_preacquisition_timeout_does_not_claim_or_mutate_server(tmp_path):
    guard, client = controller(tmp_path, fault=("resident_status", "timeout"))
    with pytest.raises(UnresolvedOperation), guard.lease():
        pytest.fail("must not acquire")
    assert not client.maintenance and not client.paused
    assert operations(client)[-1] == "resident_status"
    assert ("/resident/maintenance", "POST") not in client.calls


def test_memory_failure_prevents_native_prepare_or_load(tmp_path):
    guard, client = workflow_controller(tmp_path)
    for row in client.workers:
        row["packed_state"]["memory"]["free"] = 1
    with pytest.raises(RuntimeError, match="headroom"), guard.lease():
        guard.load()
    assert "resident_native_prepare" not in operations(client)
    assert "resident_native_load" not in operations(client)
    assert not client.maintenance and not client.paused


def test_completed_native_registry_failure_cannot_resume_or_delete_maintenance(tmp_path):
    guard, client = workflow_controller(tmp_path)
    original = client.request

    def request(path, payload=None, method="POST"):
        if path == "/collective_rpc" and payload["method"] == "resident_native_load":
            client.calls.append(("resident_native_load", method))
            for row in client.workers:
                row["native_failed"] = True
            return {
                "results": [
                    {"rank": rank, "pid": 100 + rank, "error": "irreversible native registration failed"}
                    for rank in range(4)
                ]
            }
        return original(path, payload, method)

    client.request = request
    with pytest.raises(RuntimeError, match="safe recovery failed"), guard.lease():
        guard.load()
    assert client.maintenance and client.paused
    assert "resident_prepare" not in operations(client)
    assert ("/resident/maintenance", "DELETE") not in client.calls
    assert "/resume" not in operations(client)


def test_shadow_worker_compares_native_and_whole_moe_without_rerunning_attention():
    import torch

    from tools.glm_perf.resident_candidates.streaming_experts import ShadowExpert, wrap_moe_forward

    codes = torch.zeros((72, 8, 4), dtype=torch.int8)
    arguments = (codes, torch.ones(1), codes, torch.ones(1), torch.ones(640, 8), torch.zeros(640, 8), 0)
    calls = {"current": 0, "reference": 0, "whole": 0}

    class Native:
        def __init__(self, name, factor=0.5):
            self.name, self.factor = name, factor

        def __call__(self, x, *args):
            calls[self.name] += 1
            return x.float() * self.factor

        def geometry(self, *args):
            return SimpleNamespace(gate_bits=4, experts=72, hidden=8, intermediate=4)

    state = {"cases": [], "whole_moe": [], "mismatches": 0}
    proxy = ShadowExpert(Native("current"), Native("reference"), state, {codes.data_ptr(): {"ordinal": 0}})

    def forward(self, hidden, router):
        calls["whole"] += 1
        return proxy(hidden, *arguments) + torch.ones_like(hidden, dtype=torch.float32)

    wrapped = wrap_moe_forward(forward, proxy, state)
    result = wrapped(None, torch.ones(640, 8, dtype=torch.float16), torch.zeros(640, 72))
    assert torch.equal(result, torch.full((640, 8), 1.5))
    assert len(state["cases"]) == len(state["whole_moe"]) == 1
    assert state["mismatches"] == 0
    assert calls == {"current": 1, "reference": 2, "whole": 2}
    assert proxy.reference_mode is False
    # A rank-local mismatch is recorded after both collective schedules finish;
    # it must not raise in forward and strand the next rank inside HCCL.
    proxy.reference.factor = 0.75
    wrapped(None, torch.ones(640, 8, dtype=torch.float16), torch.zeros(640, 72))
    assert state["mismatches"] == 2
    assert len(state["whole_moe"]) == 2
    assert calls["whole"] == 4


def test_default_cli_status_never_mutates(tmp_path, monkeypatch, capsys):
    import sys

    import tools.glm_perf.streaming_resident as resident

    guard, client = controller(tmp_path)
    client.rpc = lambda method: client.request("/collective_rpc", {"method": method, "args": []})["results"]
    monkeypatch.setattr(resident, "StreamingTransport", lambda *args, **kwargs: client)
    monkeypatch.setattr(sys, "argv", ["streaming_resident"])
    resident.main()
    assert '"workers"' in capsys.readouterr().out
    assert operations(client) == ["/resident/maintenance", "/is_paused", "resident_status"]
    assert not client.maintenance and not client.paused


def test_bad_shadow_geometry_is_rejected_even_if_case_count_matches(tmp_path):
    from tools.glm_perf.streaming_resident import _audit_shadow_workers

    guard, client = workflow_controller(tmp_path)
    # Build real-shaped fake report using the same executable workflow fixture.
    client.pending = Control("b" * 32, candidate="test", source="version=1102 shadow=True")
    client.paused = True
    client.request("/collective_rpc", {"method": "resident_apply", "args": []})
    client.stream("/v1/completions", {"prompt": [43] * 1920, "max_tokens": 1}, "req")
    rows = copy.deepcopy(client.workers)
    rows[0]["streaming"]["cases"][0]["bits"] = 4
    with pytest.raises(ValueError, match="bank/geometry"):
        _audit_shadow_workers(guard, rows)


def admission_fixture(tmp_path, monkeypatch, *, experimental=False):
    import tools.glm_perf.build_streaming as build_streaming
    from tools.glm_perf.streaming_protocol import evidence_schema

    root = tmp_path / "build"
    root.mkdir()
    binary = root / "glm_fused_gate_up.bin"
    binary.write_bytes(b"versioned native binary")
    identity = {
        "version": 1102,
        "namespace": "glm_reconstruction_v1102",
        "contract_sha256": "a" * 64,
        "reference_sha256": "a" * 64,
        "source_sha256": "a" * 64,
        "binaries": {binary.name: hashlib.sha256(binary.read_bytes()).hexdigest()},
    }
    dispatch = {
        "bulk_a4_gate": "streaming",
        "bulk_a4_down": "streaming",
        "sparse": "qualified_reference",
        "a8": "qualified_reference",
        "decode_version": 984,
    }
    proof = root / "build-proof.json"
    proof.write_bytes(b"memory and frozen-source proof")
    build = {
        "schema_version": 1,
        "kind": "build",
        "state": "complete",
        "identity": identity,
        "dispatch": dispatch,
        "coverage": evidence_schema()["x-required-coverage"]["build"],
        "artifacts": [{"path": proof.name, "sha256": hashlib.sha256(proof.read_bytes()).hexdigest()}],
        "errors": [],
        "result": {rule: True for rule in evidence_schema()["x-complete-result-rules"]["build"]},
    }
    (root / "build.json").write_text(json.dumps(build))
    (root / "provenance.json").write_text(json.dumps({"_build": {"version": 1102, "streaming_pipeline": True}}))
    native_proof = tmp_path / "native-proof.json"
    native_proof.write_bytes(b"actual tiny diagnostic records")
    memory = {
        "passed": True,
        "required_free_bytes": 42,
        "includes": [
            "old_resources",
            "new_resources",
            "reference_scratch",
            "candidate_scratch",
            "outputs",
            "verification",
        ],
    }
    legacy = tmp_path / "legacy.json"
    legacy.write_text("{}")
    gates = {
        "schema_version": 1,
        "kind": "native_gates",
        "state": "pending" if experimental else "complete",
        "identity": identity,
        "dispatch": dispatch,
        "coverage": ["w4_a4_k4096_rows5_balanced"]
        if experimental
        else evidence_schema()["x-required-coverage"]["native_gates"],
        "artifacts": [{"path": native_proof.name, "sha256": hashlib.sha256(native_proof.read_bytes()).hexdigest()}],
        "errors": [],
        "result": {
            **{rule: True for rule in evidence_schema()["x-complete-result-rules"]["native_gates"]},
            "build_dir": str(root),
            "build_evidence": str(root / "build.json"),
            "legacy_gate_report": str(legacy),
            "resident_memory_admission": memory,
            "experimental": experimental,
            "diagnostic_passed": True,
        },
    }
    gates_path = tmp_path / "gates.json"
    gates_path.write_text(json.dumps(gates))
    calls = []
    monkeypatch.setattr(
        build_streaming, "admit_candidate", lambda *args, **kwargs: calls.append(("admit", kwargs)) or identity
    )

    def manifest(*args, **kwargs):
        calls.append(("manifest", kwargs))
        return SimpleNamespace(verify_files=lambda: calls.append(("verify", {})))

    return gates_path, manifest, calls


def test_live_candidate_load_requires_admission_before_native_manifest(tmp_path, monkeypatch):
    from tools.glm_perf.streaming_resident import load_candidate

    path, manifest, calls = admission_fixture(tmp_path, monkeypatch)
    candidate = load_candidate(make_reference(), path, manifest_factory=manifest)
    assert not candidate.experimental
    assert [name for name, _ in calls] == ["admit", "manifest", "verify"]
    assert calls[0][1] == {"native_evidence_path": path, "live": True}
    assert calls[1][1] == {"streaming_gate_report": path}


def test_experimental_loader_keeps_pending_diagnostics_explicit(tmp_path, monkeypatch):
    from tools.glm_perf.streaming_resident import load_candidate

    path, manifest, calls = admission_fixture(tmp_path, monkeypatch, experimental=True)
    with pytest.raises(ValueError, match="not complete"):
        load_candidate(make_reference(), path, manifest_factory=manifest)
    assert not calls
    candidate = load_candidate(make_reference(), path, manifest_factory=manifest, experimental=True)
    assert candidate.experimental
    assert calls[0][1] == {"native_evidence_path": None, "live": False}
    assert calls[1][1] == {"diagnostic_report": path}


def test_experimental_loader_accepts_legacy_diagnostic_metadata(tmp_path, monkeypatch):
    from tools.glm_perf.streaming_resident import load_candidate

    path, manifest, calls = admission_fixture(tmp_path, monkeypatch, experimental=True)
    gates = json.loads(path.read_text())
    del gates["result"]["experimental"]
    del gates["result"]["diagnostic_passed"]
    path.write_text(json.dumps(gates))
    candidate = load_candidate(make_reference(), path, manifest_factory=manifest, experimental=True)
    assert candidate.experimental
    assert calls[1][1] == {"diagnostic_report": path}


@pytest.mark.parametrize(
    "change", ["diagnostic_failed", "empty_coverage", "binary_changed", "insufficient_memory", "wrong_reference"]
)
def test_experimental_admission_failure_prevents_native_manifest(tmp_path, monkeypatch, change):
    from tools.glm_perf.streaming_resident import load_candidate

    path, manifest, calls = admission_fixture(tmp_path, monkeypatch, experimental=True)
    gates = json.loads(path.read_text())
    if change == "diagnostic_failed":
        gates["result"]["diagnostic_passed"] = False
    if change == "empty_coverage":
        gates["coverage"] = []
    if change == "binary_changed":
        (Path(gates["result"]["build_dir"]) / "glm_fused_gate_up.bin").write_bytes(b"different binary")
    if change == "insufficient_memory":
        gates["result"]["resident_memory_admission"]["passed"] = False
    if change == "wrong_reference":
        gates["identity"]["reference_sha256"] = "b" * 64
    path.write_text(json.dumps(gates))
    with pytest.raises(ValueError):
        load_candidate(make_reference(), path, manifest_factory=manifest, experimental=True)
    assert not calls


def test_experimental_diagnostics_cannot_qualify_trials_or_promotion(tmp_path):
    import dataclasses

    from tools.glm_perf.streaming_protocol import make_workloads
    from tools.glm_perf.streaming_resident import trials_workflow

    guard, client = workflow_controller(tmp_path)
    guard.candidate = dataclasses.replace(guard.candidate, experimental=True)
    with pytest.raises(ValueError, match="cannot qualify"):
        trials_workflow(guard, make_workloads(), {})
    assert not client.calls


def test_sealed_composition_runs_when_old_worker_package_lacks_new_dependencies(monkeypatch):
    import builtins
    import sys

    from tools.glm_perf.resident_candidates.streaming_experts import compose_source

    source = compose_source(
        FROZEN_SOURCE,
        source_sha256=hashlib.sha256(FROZEN_SOURCE.encode()).hexdigest(),
        version=1102,
        build_dir=Path("/build"),
        expected_inventory={},
    )
    original_import = builtins.__import__
    attempts = []

    def restricted(name, *args, **kwargs):
        if name in ("tools.glm_perf.native_launch_audit", "tools.glm_perf.prefill_progress"):
            attempts.append(name)
            raise ModuleNotFoundError(name)
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", restricted)
    namespace = {}
    exec(compile(source, "<sealed-candidate>", "exec"), namespace)
    helper = namespace["_streaming_helper"]
    assert helper.NativeLaunchAudit is namespace["_streaming_audit"].NativeLaunchAudit
    assert helper.PrefillProgress is namespace["_streaming_progress"].PrefillProgress
    assert sys.modules[namespace["_streaming_progress"].__name__] is namespace["_streaming_progress"]
    assert helper.PrefillProgress().requests == {}
    assert not attempts


def test_profile_export_path_is_absolute_for_workers_with_different_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    output = Path("controller-output")
    output.mkdir()
    selections = []
    controller = SimpleNamespace(
        output_dir=output,
        lease=lambda: nullcontext(),
        load=lambda: None,
        select=lambda variant, **kwargs: selections.append((variant, kwargs)),
        pause_clear=lambda: None,
        rpc=lambda *args: None,
        resume=lambda: None,
        request_workload=lambda *args, **kwargs: ([], []),
        restore=lambda: None,
        journal={},
        last_workers=[],
    )
    profile_workflow(controller, {"workloads": [{"id": "cold_6400"}]})
    assert selections[0] == ("candidate", {"profile_dir": str(tmp_path / output / "profile")})
    assert selections[1] == ("candidate", {})
