# SPDX-License-Identifier: Apache-2.0
"""Freeze read-only reference identities and validate streaming evidence offline.

This module uses only the standard library. A freeze reads host files and pure
status RPCs; it cannot install a resource, pause serving or allocate NPU tensors.
Evidence validation is a format/admission check, not proof that hardware passed.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import statistics
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
SCALE_LAYOUT = "fp16_storage_fp32_compute_v1"
BASELINE_VERSION = 1021
DECODE_VERSION = 984
REFERENCE_ROOT = "/srv/ai/artifacts/glm-scale-layout-20261009"
CHECKPOINT = "/srv/ai/models/GLM-5.3-Flash-native-int4-fp16-scales-20261008"
EVIDENCE_KINDS = ("build", "native_gates", "shadow", "trials")
EVIDENCE_STATES = ("pending", "complete", "failed", "blocked")
SHA256_PATTERN = r"^[0-9a-f]{64}$"
IDENTITY_FIELDS = ("contract_sha256", "reference_sha256", "source_sha256", "binaries", "version", "namespace")


def canonical_digest(value: Any) -> str:
    """SHA256 of compact UTF-8 JSON with sorted keys and no nonfinite numbers."""
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def file_digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _sha(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def native_case_ids() -> list[str]:
    """Required geometries; replay/lifetime assertions accompany each case."""
    return [
        f"w{bits}_a{activation}_k{k}_rows{rows}_{routes}"
        for bits in (2, 3, 4)
        for activation in (4, 8)
        for k in (2048, 4096)
        for rows in (0, 1, 4, 5, 15, 16, 17, 30, 31, 32, 62, 63)
        for routes in ("balanced", "skewed", "all_peer")
    ] + [f"w{bits}_a4_chunk{tokens}_topk8_e72" for bits in (2, 3, 4) for tokens in (640, 1280)]


def shadow_case_ids() -> list[str]:
    return [f"rank{rank}_bank{bank}_chunk{tokens}" for rank in range(4) for bank in range(43) for tokens in (1280, 640)]


def evidence_schema() -> dict[str, Any]:
    """Published JSON Schema plus explicit per-kind coverage/semantic rules."""
    sha = {"type": "string", "pattern": SHA256_PATTERN}
    artifact = {
        "type": "object",
        "required": ["path", "sha256"],
        "properties": {"path": {"type": "string", "minLength": 1}, "sha256": sha},
        "additionalProperties": False,
    }
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "GLM streaming evidence v1",
        "type": "object",
        "required": [
            "schema_version",
            "kind",
            "state",
            "identity",
            "dispatch",
            "coverage",
            "artifacts",
            "errors",
            "result",
        ],
        "properties": {
            "schema_version": {"const": SCHEMA_VERSION},
            "kind": {"enum": list(EVIDENCE_KINDS)},
            "state": {"enum": list(EVIDENCE_STATES)},
            "identity": {
                "type": "object",
                "required": list(IDENTITY_FIELDS),
                "properties": {
                    **{key: sha for key in IDENTITY_FIELDS[:3]},
                    "binaries": {"type": "object", "minProperties": 1, "additionalProperties": sha},
                    "version": {"type": "integer", "minimum": 1},
                    "namespace": {"type": "string", "pattern": r"^glm_reconstruction_v[1-9][0-9]*$"},
                },
                "additionalProperties": False,
            },
            "dispatch": {
                "type": "object",
                "required": ["bulk_a4_gate", "bulk_a4_down", "sparse", "a8", "decode_version"],
                "properties": {
                    "bulk_a4_gate": {"const": "streaming"},
                    "bulk_a4_down": {"const": "streaming"},
                    "sparse": {"enum": ["streaming", "qualified_reference"]},
                    "a8": {"enum": ["streaming", "qualified_reference"]},
                    "decode_version": {"const": DECODE_VERSION},
                },
                "additionalProperties": False,
            },
            "coverage": {"type": "array", "uniqueItems": True, "items": {"type": "string", "minLength": 1}},
            "artifacts": {"type": "array", "minItems": 1, "items": artifact},
            "errors": {"type": "array", "items": {"type": "string", "minLength": 1}},
            "result": {"type": "object"},
        },
        "additionalProperties": False,
        "x-required-coverage": {
            "build": ["gate_up", "down", "operand_producer", "product_consumer", "compact_epilogue", "memory_bounds"],
            "native_gates": native_case_ids(),
            "shadow": shadow_case_ids(),
            "trials": [
                "cold_640",
                "cold_6400",
                "cold_32768",
                "warm_prefix_6400",
                "decode_c1",
                "decode_c4",
                "mixed",
                "language",
                "tool",
            ],
        },
        "x-complete-result-rules": {
            "build": ["memory_proof_passed", "complete_pipeline"],
            "native_gates": [
                "bitwise_equal",
                "finite",
                "changed_replay",
                "changed_inputs_scales_weights_routes",
                "drain_reuse",
                "watchdog_passed",
                "padding_defined",
                "both_projections_streaming",
            ],
            "shadow": [
                "bitwise_equal",
                "finite",
                "all_banks",
                "actual_launch_audit",
                "decode_replay",
                "prefix_checkpoints",
                "shared_expert_combine_hccl",
            ],
            "trials": [
                "requests_complete",
                "launch_audit",
                "identical_abba_inputs",
                "shadow_removed",
                "profiler_disabled",
                "timing_semantics_recorded",
            ],
        },
    }


def validate_evidence(
    report: dict[str, Any],
    *,
    kind: str | None = None,
    bindings: dict[str, Any] | None = None,
    required_cases: list[str] | None = None,
    require_complete: bool = True,
    artifact_root: Path | None = None,
) -> dict[str, Any]:
    """Reject stale, partial or structurally incompatible evidence.

    ``bindings`` is the authoritative candidate identity, not fields copied from
    the submitted report. Passing an artifact root additionally verifies bytes.
    Native gates remain impossible to admit until every frozen case is present.
    """
    schema = evidence_schema()
    if not isinstance(report, dict) or set(report) != set(schema["required"]):
        raise ValueError("evidence fields do not match schema v1")
    if type(report["schema_version"]) is not int or report["schema_version"] != SCHEMA_VERSION:
        raise ValueError("unsupported evidence schema")
    actual_kind = report["kind"]
    if actual_kind not in EVIDENCE_KINDS or (kind is not None and actual_kind != kind):
        raise ValueError("wrong evidence kind")
    if report["state"] not in EVIDENCE_STATES or (require_complete and report["state"] != "complete"):
        raise ValueError("evidence is not complete")
    identity = report["identity"]
    if not isinstance(identity, dict) or set(identity) != set(IDENTITY_FIELDS):
        raise ValueError("incomplete evidence identity")
    if not all(_sha(identity[key]) for key in IDENTITY_FIELDS[:3]):
        raise ValueError("invalid identity digest")
    binaries = identity["binaries"]
    if (
        not isinstance(binaries, dict)
        or not binaries
        or not all(isinstance(k, str) and k and _sha(v) for k, v in binaries.items())
    ):
        raise ValueError("invalid binary digest map")
    if (
        type(identity["version"]) is not int
        or identity["version"] < 1
        or identity["namespace"] != f"glm_reconstruction_v{identity['version']}"
    ):
        raise ValueError("version/namespace mismatch")
    if bindings is not None and any(identity.get(key) != value for key, value in bindings.items()):
        raise ValueError("evidence does not match authoritative bindings")
    dispatch = report["dispatch"]
    if not isinstance(dispatch, dict) or set(dispatch) != {
        "bulk_a4_gate",
        "bulk_a4_down",
        "sparse",
        "a8",
        "decode_version",
    }:
        raise ValueError("incomplete dispatch coverage")
    if (
        dispatch["bulk_a4_gate"] != "streaming"
        or dispatch["bulk_a4_down"] != "streaming"
        or type(dispatch["decode_version"]) is not int
        or dispatch["decode_version"] != DECODE_VERSION
    ):
        raise ValueError("both A4 projections must stream; decode must stay qualified")
    if dispatch["sparse"] not in ("streaming", "qualified_reference") or dispatch["a8"] not in (
        "streaming",
        "qualified_reference",
    ):
        raise ValueError("unsupported fallback dispatch")
    coverage = report["coverage"]
    if (
        not isinstance(coverage, list)
        or not all(isinstance(case, str) and case for case in coverage)
        or len(set(coverage)) != len(coverage)
    ):
        raise ValueError("coverage must contain unique case IDs")
    errors, result, artifacts = report["errors"], report["result"], report["artifacts"]
    if (
        not isinstance(errors, list)
        or not all(isinstance(error, str) and error for error in errors)
        or not isinstance(result, dict)
    ):
        raise ValueError("invalid result/error fields")
    if not isinstance(artifacts, list) or not artifacts:
        raise ValueError("evidence needs artifact locations")
    seen = set()
    for artifact in artifacts:
        if (
            not isinstance(artifact, dict)
            or set(artifact) != {"path", "sha256"}
            or not isinstance(artifact["path"], str)
            or not artifact["path"]
            or not _sha(artifact["sha256"])
        ):
            raise ValueError("invalid artifact record")
        if artifact["path"] in seen:
            raise ValueError("duplicate artifact location")
        seen.add(artifact["path"])
        if artifact_root is not None:
            path = (artifact_root / artifact["path"]).resolve()
            if (
                not path.is_relative_to(artifact_root.resolve())
                or not path.is_file()
                or file_digest(path) != artifact["sha256"]
            ):
                raise ValueError("artifact is missing, changed or outside the evidence root")
    if report["state"] == "complete":
        required = schema["x-required-coverage"][actual_kind] if required_cases is None else required_cases
        if errors or not set(required).issubset(coverage):
            raise ValueError("complete evidence has errors or missing case coverage")
        if any(result.get(rule) is not True for rule in schema["x-complete-result-rules"][actual_kind]):
            raise ValueError("complete evidence is missing required validation results")
        if actual_kind == "shadow" and result.get("comparisons") != 344:
            raise ValueError("shadow must contain all 344 actual-input comparisons")
    elif report["state"] in ("failed", "blocked") and not errors:
        raise ValueError("failed/blocked evidence must explain its state")
    return report


def _token_workload(name: str, tokens: int, output: int, *, concurrency: int = 1, cold: bool = True) -> dict[str, Any]:
    ids = [261] + [43] * (tokens - 2) + [361]
    return {
        "id": name,
        "endpoint": "/v1/completions",
        "prompt_token_ids": ids,
        "prompt_sha256": canonical_digest(ids),
        "prompt_tokens": tokens,
        "max_tokens": output,
        "concurrency": concurrency,
        "cold": cold,
        "generation": {"temperature": 0, "ignore_eos": True, "stream": True, "stream_options": {"include_usage": True}},
    }


def make_workloads() -> dict[str, Any]:
    """Build the frozen synthetic, language/tool and mixed-request protocol."""
    workloads = [_token_workload(f"cold_{n}", n, 1) for n in (640, 6400, 32768)]
    workloads += [
        _token_workload("shadow_1920", 1920, 1),
        _token_workload("warm_prefix_6400", 6400, 1, cold=False),
        _token_workload("decode_c1", 128, 128),
        _token_workload("decode_c4", 128, 128, concurrency=4),
        _token_workload("mixed", 6400, 1),
    ]
    language = {
        "messages": [
            {
                "role": "user",
                "content": (
                    "A queue holds 9 jobs. Three workers each finish 2 jobs. Explain how many jobs remain, "
                    "then return a JSON object with remaining_jobs and worker_count."
                ),
            }
        ],
        "max_tokens": 192,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    tool = {
        "messages": [
            {"role": "user", "content": "Call lookup_order with order_id ORD-2048. Do not invent the order status."}
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "lookup_order",
                    "description": "Fetch one order by its exact ID.",
                    "parameters": {
                        "type": "object",
                        "properties": {"order_id": {"type": "string"}},
                        "required": ["order_id"],
                        "additionalProperties": False,
                    },
                },
            }
        ],
        "tool_choice": "auto",
        "max_tokens": 192,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    workloads += [
        {
            "id": name,
            "endpoint": "/v1/chat/completions",
            "input": payload,
            "rendered_input_sha256": canonical_digest(payload),
            "cold": True,
            "materialize_token_ids_before_trials": True,
        }
        for name, payload in (("language", language), ("tool", tool))
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "workloads": workloads,
        "abba": {
            "order": ["reference", "candidate", "candidate", "reference"],
            "minimum_cycles": 3,
            "primary": "cold_6400",
            "same_token_ids_within_cycle": True,
            "separate_cycles_may_use_distinct_frozen_nonce": True,
            "clear_prefix_before_every_cold_request": True,
            "require_reused_at_admission_zero": True,
        },
        "warmup": {
            "timed": False,
            "per_install": ["decode_c1", "decode_c4", "cold_640", "cold_6400"],
            "same_order_and_inputs_for_both_variants": True,
            "decode_graph_sizes": [2, 8],
            "clear_prefix_after_warmup": True,
            "report_first_use_separately": True,
        },
        "warm_prefix": {
            "prime_identical_input_untimed": True,
            "report_actual_reused_tokens": True,
            "do_not_label_as_cold": True,
        },
        "long_context": {
            "existing_context_depths": [0, 6400, 32768],
            "append_tokens": 640,
            "prime_context_untimed": True,
            "freeze_concatenated_token_ids": True,
            "record_reused_and_completed_tokens": True,
        },
        "mixed": {
            "start_decode_c1_first": True,
            "prefill_workload": "cold_6400",
            "decode_workload": "decode_c1",
            "require_observed_overlap": True,
            "record_decode_intertoken_latency_and_prefill_chunks": True,
        },
        "input_rules": {
            "freeze_tokenizer_and_chat_template_hashes": True,
            "language_tool_tokenization": (
                "Render/tokenize once before all variants; save final IDs, count, template/renderer "
                "settings and hashes. Refuse timing if either variant changes IDs."
            ),
            "generation_length": (
                "Synthetic controls must complete requested length; record finish reason/usage for "
                "language/tool controls."
            ),
            "quality": (
                "Finite actual-input parity is the arithmetic gate. Natural language must answer 3 "
                "jobs/3 workers; tool output must parse to lookup_order({order_id: ORD-2048}). Record "
                "baseline variation; exact first-token equality alone is not admission."
            ),
        },
        "timing": {
            "clock": "monotonic durations; do not subtract server timestamps from client timestamps",
            "preferred_ttft": (
                "Request-correlated server first-generated-token time minus that request's server "
                "admission time, when both timestamps are available and comparable."
            ),
            "fallback_ttft": (
                "Client request start to first token-bearing SSE event; label streamed_ttft_s. A "
                "role-only, usage-only, empty delta or DONE event is not a generated token."
            ),
            "token_evidence": (
                "Explicit token IDs/logprob records when present; otherwise any nonempty "
                "content/reasoning_content/text or tool-call argument delta constitutes the first "
                "token-bearing event. Save the exact identifying SSE event and rule."
            ),
            "first_visible_text": (
                "Record separately; reasoning or tool arguments can precede content. Do not replace "
                "token TTFT with visible-text latency."
            ),
            "completion_latency": "Client start to DONE and complete usage; failed/cancelled requests are not samples.",
            "decode_tokens_per_second": (
                "(completion_tokens - 1)/(last_generated_token_time - first_generated_token_time), with"
                " clock and event-boundary method recorded; n<=1 has no decode rate."
            ),
            "token_progress": (
                "scheduled_through is scheduler assignment, not device completion; sum completed chunks"
                " separately from admissions and reused prefixes."
            ),
        },
        "promotion": {
            "objective_cold_ttft_improvement": 0.10,
            "repeat_spread": "max((p75-p25)/median) over reference and candidate",
            "quantiles": "linear interpolation at (n-1)*p",
            "minimum_improvement": "strictly greater than max(0.02, 2*repeat_spread)",
            "not_a_significance_test": True,
            "max_repeatable_decode_or_mixed_regression": 0.05,
            "require_complete_correctness_gates": True,
            "inconclusive_retains_reference": True,
        },
    }


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    point = (len(ordered) - 1) * fraction
    left = math.floor(point)
    right = math.ceil(point)
    return ordered[left] + (ordered[right] - ordered[left]) * (point - left)


def summarize_trials(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize primary cold ABBA trials; mixed/quality gates remain separate."""
    groups = {"reference": [], "candidate": []}
    cycles: dict[int, list[dict[str, Any]]] = {}
    for sample in samples:
        if (
            sample.get("variant") not in groups
            or type(sample.get("cycle")) is not int
            or type(sample.get("position")) is not int
        ):
            raise ValueError("invalid ABBA sample identity")
        duration = sample.get("ttft_s")
        if type(duration) not in (int, float) or not math.isfinite(duration) or duration <= 0:
            raise ValueError("invalid TTFT")
        if (
            sample.get("complete") is not True
            or sample.get("reused_at_admission") != 0
            or sample.get("timing_method") not in ("server_generated_token", "streamed_token_event")
            or not _sha(sample.get("prompt_sha256"))
        ):
            raise ValueError("trial must be completed cold work with explicit token timing")
        groups[sample["variant"]].append(float(duration))
        cycles.setdefault(sample["cycle"], []).append(sample)
    if len(cycles) < 3:
        raise ValueError("at least three complete ABBA cycles are required")
    for cycle in cycles.values():
        rows = sorted(cycle, key=lambda row: row["position"])
        if [row["position"] for row in rows] != [0, 1, 2, 3] or [row["variant"] for row in rows] != [
            "reference",
            "candidate",
            "candidate",
            "reference",
        ]:
            raise ValueError("incomplete or misordered ABBA cycle")
        if len({row["prompt_sha256"] for row in rows}) != 1 or len({row["timing_method"] for row in rows}) != 1:
            raise ValueError("ABBA inputs/timing method differ within a cycle")
    if len({sample["timing_method"] for sample in samples}) != 1:
        raise ValueError("cannot mix server and client TTFT")
    summary = {
        variant: {
            "samples": values,
            "median_s": statistics.median(values),
            "p25_s": _percentile(values, 0.25),
            "p75_s": _percentile(values, 0.75),
        }
        for variant, values in groups.items()
    }
    spread = max((row["p75_s"] - row["p25_s"]) / row["median_s"] for row in summary.values())
    improvement = 1 - summary["candidate"]["median_s"] / summary["reference"]["median_s"]
    threshold = max(0.02, 2 * spread)
    return {
        "variants": summary,
        "repeat_spread": spread,
        "improvement": improvement,
        "minimum_improvement": threshold,
        "repeatability_passed": improvement > threshold,
        "objective_10pct_met": improvement >= 0.10,
        "cycles": len(cycles),
        "scope": "repeatability rule only; separate correctness/decode/mixed gates required",
    }


# Executed by remote stdlib Python. Reads local files, /proc and pure status RPCs.
REMOTE_COLLECTOR = r"""
import base64, hashlib, json, urllib.request
from pathlib import Path

root = Path("/srv/ai/artifacts/glm-scale-layout-20261009")
checkpoint = Path("/srv/ai/models/GLM-5.3-Flash-native-int4-fp16-scales-20261008")
base = "http://127.0.0.1:8001"
queries = []


def query(path, payload=None):
    method = "POST" if payload is not None else "GET"
    if method == "POST" and (
        path != "/collective_rpc" or payload.get("method") != "resident_status" or payload.get("args") != []
    ):
        raise ValueError("non-read-only query prohibited")
    queries.append({"path": path, "method": method, "rpc": payload.get("method") if payload else None})
    request = urllib.request.Request(
        base + path,
        data=json.dumps(payload).encode() if payload else None,
        headers={"Content-Type": "application/json"},
        method=method,
    )
    with urllib.request.urlopen(request, timeout=35) as response:
        raw = response.read().decode()
        return {"http_status": response.status, "body": json.loads(raw) if raw and path != "/metrics" else raw}


def status():
    return query("/collective_rpc", {"method": "resident_status", "args": [], "timeout": 30})["body"]["results"]


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


rows = status()
keys = (
    "rank",
    "pid",
    "generation",
    "candidate",
    "digest",
    "graphs_dirty",
    "weight_storage_digest",
    "native_failed",
    "fp16_scale_storage",
    "chunk_sweep",
    "native_launch_audit",
    "scale_queue",
    "prepared_weight_layout",
)
workers = [{key: row.get(key) for key in keys} for row in rows]
processes = []
listen_inodes = set()
for table in ("tcp", "tcp6"):
    for line in (Path("/proc/net") / table).read_text().splitlines()[1:]:
        columns = line.split()
        if int(columns[1].split(":")[1], 16) == 8001 and columns[3] == "0A":
            listen_inodes.add("socket:[" + columns[9] + "]")
for directory in Path("/proc").iterdir():
    if not directory.name.isdigit():
        continue
    try:
        argv = [x.decode() for x in (directory / "cmdline").read_bytes().split(b"\0") if x]
        if "--port" in argv and argv[argv.index("--port") + 1] == "8001" and "vllm.entrypoints.cli.main" in argv:
            sockets = {str(path.readlink()) for path in (directory / "fd").iterdir()}
            if sockets.intersection(listen_inodes):
                processes.append({"pid": int(directory.name), "argv": argv, "cwd": str((directory / "cwd").resolve())})
    except (OSError, ValueError, UnicodeError):
        pass
files = {}
for name in (
    "final-candidate-v1021.py",
    "build-down-group-v1021/provenance.json",
    "gates-down-group-v1021.json",
    "shadow-summary.json",
):
    path = root / name
    files[name] = {
        "remote_path": str(path),
        "sha256": digest(path),
        "base64": base64.b64encode(path.read_bytes()).decode(),
    }
shadow_full_path = root / "shadow-serving-results.json"
shadow_full = json.loads(shadow_full_path.read_text())
historical_workers = [
    {key: row[key] for key in ("rank", "pid", "weight_storage_digest")} for row in shadow_full["final_workers"]
]
historical_record = {
    "workers": historical_workers,
    "source_path": str(shadow_full_path),
    "source_sha256": digest(shadow_full_path),
}
raw = json.dumps(historical_record, sort_keys=True).encode()
files["shadow-worker-identities.json"] = {
    "remote_path": str(shadow_full_path) + "#final_workers",
    "sha256": hashlib.sha256(raw).hexdigest(),
    "base64": base64.b64encode(raw).decode(),
}
cp_files = {}
for name in (
    "config.json",
    "model.safetensors.index.json",
    "native-layout.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
):
    path = checkpoint / name
    cp_files[name] = {"remote_path": str(path), "sha256": digest(path), "bytes": path.stat().st_size}
    if name in ("config.json", "model.safetensors.index.json", "native-layout.json"):
        cp_files[name]["base64"] = base64.b64encode(path.read_bytes()).decode()
build = root / "build-down-group-v1021"
binaries = {path.name: digest(path) for path in sorted(build.iterdir()) if path.suffix in (".bin", ".so")}
helper_hashes = {
    str(path.relative_to(build)): digest(path)
    for path in sorted((build / "glm_reconstruction_v1021_helpers").rglob("*.py"))
}
health = query("/health")
paused = query("/is_paused")
maintenance = query("/resident/maintenance")
metrics = query("/metrics")
server_info = query("/server_info")
after = status()
fields = ("rank", "pid", "generation", "digest", "weight_storage_digest")
if [{key: r.get(key) for key in fields} for r in rows] != [{key: r.get(key) for key in fields} for r in after]:
    raise RuntimeError("runtime identity changed during reference freeze")
print(
    json.dumps(
        {
            "workers": workers,
            "processes": processes,
            "files": files,
            "checkpoint_files": cp_files,
            "binaries": binaries,
            "helper_hashes": helper_hashes,
            "health": health,
            "paused": paused,
            "maintenance": maintenance,
            "metrics": metrics,
            "server_info": server_info,
            "queries": queries,
        }
    )
)
"""


def collect_reference(host: str) -> dict[str, Any]:
    allowed = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_.@-"
    if not host or any(character not in allowed for character in host) or host.startswith("-"):
        raise ValueError("invalid SSH host")
    completed = subprocess.run(
        ["ssh", "-F", "/dev/null", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", host, "python3 -"],
        input=REMOTE_COLLECTOR,
        text=True,
        capture_output=True,
        timeout=120,
        check=True,
    )
    return json.loads(completed.stdout)


def _runtime_settings(argv: list[str]) -> dict[str, Any]:
    def value(flag: str) -> str:
        return argv[argv.index(flag) + 1]

    return {
        "context": int(value("--max-model-len")),
        "batch_tokens": int(value("--max-num-batched-tokens")),
        "max_seqs": int(value("--max-num-seqs")),
        "tp": int(value("--tensor-parallel-size")),
        "kv_cache_bytes": int(value("--kv-cache-memory-bytes")),
        "gpu_memory_utilization": float(value("--gpu-memory-utilization")),
        "mtp": json.loads(value("--speculative-config")),
        "compilation": json.loads(value("--compilation-config")),
        "hf_overrides": json.loads(value("--hf-overrides")),
        "model_name": value("--served-model-name"),
        "prefix_cache": "--enable-prefix-caching" in argv,
        "chunked_prefill": "--enable-chunked-prefill" in argv,
        "async_scheduling": "--no-async-scheduling" not in argv,
        "mamba_cache_mode": value("--mamba-cache-mode"),
    }


def freeze_reference(snapshot: dict[str, Any], output_dir: Path, *, host: str) -> dict[str, Any]:
    """Validate the snapshot before publishing an append-only local reference."""
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("reference directory is not empty; refusing to overwrite frozen evidence")
    rows = snapshot["workers"]
    if len(rows) != 4 or {row["rank"] for row in rows} != set(range(4)) or len(snapshot["processes"]) != 1:
        raise ValueError("expected four unique ranks and one API process")
    if len({(row["generation"], row["digest"], row["candidate"]) for row in rows}) != 1:
        raise ValueError("reference dispatch is inconsistent across ranks")
    files = snapshot["files"]
    source_record = files["final-candidate-v1021.py"]
    source_bytes = base64.b64decode(source_record["base64"], validate=True)
    if any(
        row["digest"] != hashlib.sha256(source_bytes).hexdigest()
        or row["graphs_dirty"] is not False
        or row["native_failed"] is not False
        for row in rows
    ):
        raise ValueError("reference source does not match healthy captured dispatch")
    provenance = json.loads(base64.b64decode(files["build-down-group-v1021/provenance.json"]["base64"]))
    if (
        provenance["_build"].get("version") != BASELINE_VERSION
        or provenance["_build"].get("namespace") != "glm_reconstruction_v1021"
    ):
        raise ValueError("wrong reference build")
    for filename, entry in provenance.items():
        if isinstance(entry, dict) and "binary_sha256" in entry:
            binary_name = {
                "glm_w4a8_pack.cpp": "glm_w4a8_pack.bin",
                "glm_lowbit_a8_pipeline.cpp": "glm_w4a8_matmul.bin",
                "reconstruction_bridge.cpp": "glm_reconstruction_bridge_v1021.so",
            }.get(filename, filename)
            if snapshot["binaries"].get(binary_name) != entry["binary_sha256"]:
                raise ValueError("compiled binary differs from provenance")
    for filename, expected in provenance["_helpers"].items():
        if snapshot["helper_hashes"].get("glm_reconstruction_v1021_helpers/" + filename) != expected:
            raise ValueError("frozen build helper differs from provenance")
    for row in rows:
        scale = row["fp16_scale_storage"]
        if (
            scale["prefill_version"] != BASELINE_VERSION
            or scale["decode_version"] != DECODE_VERSION
            or scale["banks"] != 43
            or scale["layouts"] != [SCALE_LAYOUT]
        ):
            raise ValueError("wrong reference scale/bank dispatch")
        audit = row["native_launch_audit"]
        if audit["operator"] != "glm_reconstruction_v1021.launch" or not audit["binaries"]:
            raise ValueError("missing actual native launch identity")
        for entry in audit["binaries"].values():
            if snapshot["binaries"].get(Path(entry["path"]).name) != entry["sha256"]:
                raise ValueError("actual native launch binary identity mismatch")
    checkpoint_files = snapshot["checkpoint_files"]
    manifest = json.loads(base64.b64decode(checkpoint_files["native-layout.json"]["base64"]))
    index = json.loads(base64.b64decode(checkpoint_files["model.safetensors.index.json"]["base64"]))
    if (
        manifest["complete"] is not True
        or manifest["weight_scale_layout"] != SCALE_LAYOUT
        or index["metadata"]["expert_scale_layout"] != SCALE_LAYOUT
    ):
        raise ValueError("checkpoint does not satisfy the FP16 scale contract")
    shadow = json.loads(base64.b64decode(files["shadow-summary.json"]["base64"]))
    historical = json.loads(base64.b64decode(files["shadow-worker-identities.json"]["base64"]))
    prior = {row["rank"]: row for row in historical["workers"]}
    inventories = []
    for item in shadow["cases_per_rank"]:
        rank = item["rank"]
        current = next(row for row in rows if row["rank"] == rank)
        if any(current[field] != prior[rank][field] for field in ("pid", "weight_storage_digest")):
            raise ValueError("historical bank inventory no longer matches loaded storage")
        banks = {case["bank_pointer"]: case["bits"] for case in item["cases"]}
        if len(banks) != 43 or set(banks.values()) != {2, 3, 4}:
            raise ValueError("historical bank inventory is incomplete")
        inventories.append(
            {
                "rank": rank,
                "source": "historical shadow inventory bound to unchanged current PID/storage digest",
                "banks": [{"pointer": pointer, "bits": bits} for pointer, bits in banks.items()],
                "experts_per_bank": 72,
            }
        )
    settings = _runtime_settings(snapshot["processes"][0]["argv"])
    if (
        settings["context"],
        settings["batch_tokens"],
        settings["max_seqs"],
        settings["tp"],
        settings["kv_cache_bytes"],
    ) != (131072, 1280, 4, 4, 2415919104):
        raise ValueError("runtime settings changed from the planned fixed comparison")
    if settings["mtp"].get("num_speculative_tokens") != 1 or settings["compilation"] != {
        "cudagraph_mode": "FULL_DECODE_ONLY",
        "cudagraph_capture_sizes": [2, 8],
    }:
        raise ValueError("reference graph/speculation settings changed")
    if snapshot["health"]["http_status"] != 200:
        raise ValueError("reference health is not ready")
    decoded = {}
    for name, entry in files.items():
        raw = base64.b64decode(entry["base64"], validate=True)
        if hashlib.sha256(raw).hexdigest() != entry["sha256"]:
            raise ValueError("downloaded reference file hash mismatch")
        decoded[name.replace("/", "__")] = raw
    for name, entry in checkpoint_files.items():
        if "base64" in entry:
            raw = base64.b64decode(entry["base64"], validate=True)
            if hashlib.sha256(raw).hexdigest() != entry["sha256"]:
                raise ValueError("downloaded checkpoint descriptor hash mismatch")
            decoded["checkpoint__" + name] = raw
    reference = {
        "schema_version": SCHEMA_VERSION,
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "npu_host": host,
        "version": BASELINE_VERSION,
        "namespace": "glm_reconstruction_v1021",
        "decode_version": DECODE_VERSION,
        "replacement_source": {
            "path": "final-candidate-v1021.py",
            "sha256": source_record["sha256"],
            "remote_path": source_record["remote_path"],
        },
        "build_options": provenance["_build"],
        "binaries": snapshot["binaries"],
        "source_provenance": provenance,
        "runtime": {
            "workers": rows,
            "api_process": snapshot["processes"][0],
            "settings": settings,
            "server_info": snapshot["server_info"]["body"],
            "paused": snapshot["paused"]["body"],
            "maintenance": snapshot["maintenance"]["body"],
        },
        "checkpoint": {
            "path": CHECKPOINT,
            "scale_layout": SCALE_LAYOUT,
            "descriptors": {
                key: {field: value for field, value in entry.items() if field != "base64"}
                for key, entry in checkpoint_files.items()
            },
            "native_shards": manifest["native_shards"],
            "scale_shards": manifest.get("scale_shards", []),
            "shard_hash_status": (
                "declared by hashed checkpoint manifest; weight shard contents not reread during freeze"
            ),
        },
        "bank_inventory": inventories,
        "historical_bank_identity_evidence": historical,
        "identity_rules": {
            "same_process_storage": ["rank", "pid", "weight_storage_digest"],
            "same_dispatch": ["generation", "digest"],
            "same_checkpoint": (
                "descriptor and manifest hashes; declared shard hashes authenticate export identity "
                "only, not a new full-shard verification"
            ),
            "no_device_reads": True,
        },
        "read_only_queries": snapshot["queries"],
        "frozen_files": [{"path": name, "sha256": hashlib.sha256(raw).hexdigest()} for name, raw in decoded.items()],
    }
    workloads = make_workloads()
    workloads["tokenizer_hashes"] = {
        name: checkpoint_files[name]["sha256"]
        for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja")
    }
    decoded["workloads.json"] = (json.dumps(workloads, indent=2, allow_nan=False) + "\n").encode()
    decoded["evidence-schema.json"] = (json.dumps(evidence_schema(), indent=2) + "\n").encode()
    reference["frozen_files"] += [
        {"path": name, "sha256": hashlib.sha256(decoded[name]).hexdigest()}
        for name in ("workloads.json", "evidence-schema.json")
    ]
    reference["reference_sha256"] = canonical_digest(reference)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, raw in decoded.items():
        (output_dir / name).write_bytes(raw)
    (output_dir / "reference.json").write_text(json.dumps(reference, indent=2, allow_nan=False) + "\n")
    verify_reference(reference, output_dir)
    return reference


def verify_reference(reference: dict[str, Any], directory: Path) -> dict[str, Any]:
    unsealed = {key: value for key, value in reference.items() if key != "reference_sha256"}
    if reference.get("reference_sha256") != canonical_digest(unsealed):
        raise ValueError("reference digest is stale")
    for item in reference["frozen_files"]:
        path = (directory / item["path"]).resolve()
        if not path.is_relative_to(directory.resolve()) or not path.is_file() or file_digest(path) != item["sha256"]:
            raise ValueError("frozen reference file is missing or changed")
    return reference


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--npu-host")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.freeze:
        if not args.npu_host:
            parser.error("--freeze requires --npu-host")
        reference = freeze_reference(collect_reference(args.npu_host), args.output_dir, host=args.npu_host)
    else:
        reference = verify_reference(json.loads((args.output_dir / "reference.json").read_text()), args.output_dir)
    print(
        json.dumps(
            {
                "reference_sha256": reference["reference_sha256"],
                "version": reference["version"],
                "ranks": len(reference["runtime"]["workers"]),
                "output_dir": str(args.output_dir),
                "mode": "read_only_freeze" if args.freeze else "offline_verify",
            }
        )
    )


if __name__ == "__main__":
    main()
