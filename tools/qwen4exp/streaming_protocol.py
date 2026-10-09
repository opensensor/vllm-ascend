# SPDX-License-Identifier: Apache-2.0
"""Host-only, immutable reference and evidence protocol for Qwen streaming.

Hashes identify bytes, not hardware qualification. Preparation reads committed
Git blobs and archived JSON only. It never contacts a server, loads a library,
imports a device runtime, or treats a historical launcher as current state.
"""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path, PurePosixPath

SCHEMA_VERSION = 1
EXPECTED_RANKS = (0, 1, 2, 3)
REQUIRED_GATES = (
    "kernel_parity",
    "real_weights",
    "quality",
    "image",
    "cache_cow",
    "ep",
    "mtp",
    "graph_replay",
    "sustained_thermal",
    "service_performance",
)
SOURCE_ROOTS = (
    "vllm_ascend/models/qwen4_exp/",
    "vllm_ascend/_310p/",
    "vllm_ascend/core/",
    "vllm_ascend/patch/",
    "vllm_ascend/worker/",
    "vllm_ascend/ops/",
    "vllm_ascend/envs.py",
    "csrc/",
    "tools/qwen4exp/",
    "tools/glm_perf/resident_native.py",
)
ARCHIVES = (
    "artifacts/qwen38-prefix-npu-20261008/runtime-provenance.json",
    "artifacts/qwen38-thermal-incident-20261008/deferred-profile.json",
    "artifacts/qwen38-thermal-incident-20261008/incident.json",
    "artifacts/qwen38-transfer-next-20261009/host-build-provenance.json",
    "artifacts/qwen38-thermal-incident-20261008/start-fast-image-paced.sh",
)
CASE_DEFINITIONS = (
    ("short", "text", "cold", 128, 32),
    ("23k", "text", "cold", 23410, 32),
    ("long", "text", "cold", 65536, 128),
    ("short_warm", "text", "warm", 128, 32),
    ("23k_warm", "text", "warm", 23410, 32),
    ("long_warm", "text", "warm", 65536, 128),
    ("tool", "tool", "cold", None, 256),
    ("tool_warm", "tool", "warm", None, 256),
    ("image_fresh", "image", "cold", None, 128),
    ("image_cached", "image", "warm", None, 128),
    ("mixed", "mixed", "cold", None, 256),
)


def canonical_sha256(value):
    """SHA256 of UTF-8 JSON with sorted keys, no whitespace or NaN/Infinity."""
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def file_sha256(path):
    """Hash a local file without importing or loading its contents as code."""
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def source_sha256(assets, gitlinks=()):
    """Bind source bytes and pinned external Git commits without inventing hashes."""
    return canonical_sha256({"assets": assets, "gitlinks": list(gitlinks)})


def _hash(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _git(root, *args):
    return subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True).stdout


def seal_manifest(value):
    """Return a copy sealed over all fields except its own manifest_sha256."""
    result = dict(value)
    result.pop("manifest_sha256", None)
    result["manifest_sha256"] = canonical_sha256(result)
    return result


def request_payloads():
    """Concrete deterministic fixtures; token counts require the real tokenizer.

    These are authored benchmark inputs, not recovered production conversations.
    Long text sizes are selected in paragraphs, never labeled measured tokens.
    Requests are templates until verified model/tokenizer IDs and image bytes
    are supplied. Payloads are stored once and referenced by all C/MTP variants.
    """
    paragraph = (
        "The serving worker owns packed expert weights and a bounded set of activation buffers. "
        "A producer may reuse a buffer only after its consumer has completed. "
        "Queue time, first generated token, decode gaps and thermal hold time are separate measurements. "
        "Stable routing preserves expert order and exact zero outputs for routes owned by another rank."
    )
    payloads = {}
    for name, kind, cache, target, generation in CASE_DEFINITIONS:
        if cache == "warm":
            continue
        tools = None
        if name == "short":
            content = (
                "Explain why an activation buffer must not be overwritten while a consumer still uses it. "
                "Use two sentences."
            )
        elif name in ("23k", "long", "mixed"):
            paragraphs = {"23k": 200, "long": 640, "mixed": 120}[name]
            content = (
                "Review the following numbered engineering notes. "
                "Identify three memory ownership rules and explain them briefly.\n\n"
            )
            content += "\n\n".join(f"Note {index + 1}: {paragraph}" for index in range(paragraphs))
        elif kind == "tool":
            content = (
                "Read the four-core temperature fixture using read_temperature_fixture, "
                "then state whether any core requires a hold at 94 C."
            )
            tools = [
                {
                    "type": "function",
                    "function": {
                        "name": "read_temperature_fixture",
                        "description": (
                            "Return deterministic benchmark fixture temperatures; this does not access hardware."
                        ),
                        "parameters": {
                            "type": "object",
                            "properties": {"fixture_id": {"type": "string", "enum": ["qwen-streaming-four-core"]}},
                            "required": ["fixture_id"],
                            "additionalProperties": False,
                        },
                    },
                }
            ]
        else:
            content = [
                {
                    "type": "text",
                    "text": (
                        "Read the labels in this fresh screenshot and summarize the visible process in two sentences."
                    ),
                },
                {"type": "image_url", "image_url": {"url": None}},
            ]
        conversation = [
            {"role": "system", "content": "Answer clearly. Treat the following as a self-contained benchmark fixture."},
            {"role": "user", "content": content},
        ]
        request = {"model": None, "messages": conversation, "max_tokens": generation, "temperature": 0, "stream": True}
        if tools is not None:
            request["tools"] = tools
            request["tool_choice"] = "auto"
        payloads[name] = {
            "conversation": conversation,
            "conversation_sha256": canonical_sha256(conversation),
            "request": request,
            "rendered_request_sha256": canonical_sha256(request),
            "rendering": "canonical JSON; model ID unresolved; image URL unresolved for image fixture",
            "tokenizer": {"id": None, "revision": None, "status": "pending_verified_tokenizer"},
            "model_id": None,
            "token_count_status": "target_only; tokenization pending",
            "target_prompt_tokens": target,
            "fixture_provenance": "authored deterministic input; not an archived production conversation",
        }
        if kind == "tool":
            payloads[name]["tool_response_fixture"] = {
                "fixture_id": "qwen-streaming-four-core",
                "temperatures_c": [81, 85, 94, 83],
            }
        if kind == "mixed":
            payloads[name]["arrival_schedule"] = {
                "initial_role": "decode",
                "arrival_role": "prefill",
                "trigger": {"event": "generated_token", "index": 16},
                "arrival_payload_ref": "23k",
                "submitted_request_cap": 4,
                "active_request_cap": 3,
                "trigger_failure": "inconclusive; do not substitute wall-clock delay",
            }
    return payloads


def workload_definitions():
    """Unresolved payloads are explicitly pending; generation limits are fixed.

    C4 means four submitted requests under the fixed three-active-slot reference,
    including queue time. It does not silently increase capacity. MTP variation
    has a corresponding graph shape and must be compared to the same MTP mode.
    """
    result = []
    payloads = request_payloads()
    warm_base = {
        "short_warm": "short",
        "23k_warm": "23k",
        "long_warm": "long",
        "tool_warm": "tool",
        "image_cached": "image_fresh",
    }
    for name, kind, cache, tokens, generation in CASE_DEFINITIONS:
        payload_ref = warm_base.get(name, name)
        payload = payloads[payload_ref]
        for submitted in range(1, 5):
            for mtp in range(3):
                suffix = f"C{submitted}-MTP{mtp}"
                result.append(
                    {
                        "id": f"{name}-{suffix}",
                        "kind": kind,
                        "cache_policy": cache,
                        "submitted_requests": submitted,
                        "max_active_requests": 3,
                        "mtp_tokens": mtp,
                        "graph_capture_sizes": [mtp + 1, 3 * (mtp + 1)],
                        "target_prompt_tokens": tokens,
                        "actual_prompt_tokens": None,
                        "max_new_tokens": generation,
                        "token_ids": None,
                        "token_ids_sha256": None,
                        "conversation": {"payload_ref": payload_ref},
                        "conversation_sha256": payload["conversation_sha256"],
                        "request_payload_ref": payload_ref,
                        "rendered_request_sha256": payload["rendered_request_sha256"],
                        "image_sha256": None,
                        "image_content_path": None,
                        "input_status": "pending_image_and_tokenization" if kind == "image" else "pending_tokenization",
                        "warm_control": f"{warm_base[name]}-{suffix}" if name in warm_base else None,
                        "warm_requirement": "identical token IDs and image bytes" if cache == "warm" else None,
                        "mixed_schedule": "prefill arrival during an ongoing decode" if kind == "mixed" else None,
                    }
                )
    return result


def freeze_reference(root, revision="2a2a4e416"):
    """Freeze committed source bytes plus historical records, never live state.

    Dirty/untracked files are excluded by construction. Each included asset has
    its committed blob SHA256; changing source bytes invalidates this identity.
    Actual checkpoint and coherent serving OPP hashes are unknown in the local
    archive and remain pending, regardless of staged native build hashes.
    """
    root = Path(root)
    commit = _git(root, "rev-parse", "--verify", f"{revision}^{{commit}}").decode().strip()
    tree = _git(root, "ls-tree", "-r", commit).decode().splitlines()
    records = [(entry.split("\t", 1)[0].split(), entry.split("\t", 1)[1]) for entry in tree]
    paths = [path for _, path in records]
    assets = []
    gitlinks = []
    for metadata, path in records:
        if any(path == prefix or path.startswith(prefix) for prefix in SOURCE_ROOTS):
            if metadata[1] == "commit":
                gitlinks.append({"path": path, "commit": metadata[2], "content_status": "pending_external_bytes"})
                continue
            content = _git(root, "show", f"{commit}:{path}")
            assets.append({"path": path, "sha256": hashlib.sha256(content).hexdigest()})
    archives = []
    for path in ARCHIVES:
        if path not in paths:
            archives.append({"path": path, "status": "missing", "sha256": None})
            continue
        content = _git(root, "show", f"{commit}:{path}")
        archives.append(
            {
                "path": path,
                "status": "archived",
                "sha256": hashlib.sha256(content).hexdigest(),
                "record": json.loads(content) if path.endswith(".json") else None,
            }
        )
    arithmetic = {
        "expert_checkpoint": "W4A16 G128 packed resident bank",
        "execution": "cube_310_int4_a8",
        "activation_quantization": "int8_per_group",
        "group_size": 128,
        "grouped_activation": "cann_builtin_fp16",
        "grouped_finalize": "cann_v2",
        "projection_boundary": "float16",
        "route_accumulator": "float32",
        "state_dtype": "float32",
        "shared_expert_execution": "tp_sharded",
        "mtp_expert_arithmetic": "baseline W8A16; no implicit W8A8 substitution",
        "operation_order": "preserve G128 correction, group addition and stable route order",
    }
    profile = {
        "tp": 4,
        "ep": 4,
        "max_model_len": 262144,
        "max_num_seqs": 3,
        "max_num_batched_tokens": 2560,
        "kv_cache_fraction": 0.70,
        "gpu_memory_utilization": 0.965,
        "mtp_tokens": 2,
        "graph_capture_sizes": [3, 9],
        "graph_mode": "FULL_DECODE_ONLY",
        "async_scheduling": False,
        "mamba_cache_mode": "align",
        "prefix_caching": True,
        "image_processor": {"image_limit": 1, "video_limit": 0, "min_pixels": 65536, "max_pixels": 1048576},
        "tool_parser": "qwen3_xml",
        "reasoning_parser": "qwen3",
    }
    candidate = next((a["record"] for a in archives if a["path"] == ARCHIVES[3] and a["status"] == "archived"), None)
    value = {
        "schema_version": SCHEMA_VERSION,
        "kind": "qwen_streaming_reference",
        "qualification": "offline_preparation",
        "payload_revision": 2,
        "hardware_validated": False,
        "offline_reference": "complete",
        "live_reference": "incomplete",
        "source": {
            "commit": commit,
            "assets": assets,
            "gitlinks": gitlinks,
            "source_sha256": source_sha256(assets, gitlinks),
            "snapshot": "committed_git_blobs; excludes dirty and untracked overlays",
        },
        "arithmetic": arithmetic,
        "arithmetic_sha256": canonical_sha256(arithmetic),
        "comparison_capacity": {
            k: profile[k]
            for k in ("tp", "ep", "max_model_len", "max_num_seqs", "kv_cache_fraction", "gpu_memory_utilization")
        },
        "historical_runtime": {
            "status": "thermally_unqualified",
            "profile": profile,
            "provenance_status": "partial archived evidence; not current runtime",
            "runtime_root": "/srv/ai/src/qwen38-prefix-bounded-runtime-20261008",
        },
        "live_runtime": {"status": "unknown", "queried": False},
        "checkpoint": {
            "path_hint": "/srv/ai/models/Qwen3.8-Flash-Next-W4A16-G128-300i",
            "sha256": None,
            "status": "pending_local_or_authorized_receipt",
        },
        "serving_bundle": {"sha256": None, "status": "pending_coherent_opp_bridge_binary_receipt"},
        "candidate": None,
        "archives": archives,
        "staged_native_build": {
            "receipt": candidate,
            "hardware_validated": False,
            "status": "host_compiled_only" if candidate else "missing",
        },
        "thermal_policy": {"hold_c": 94, "resume_all_cores_c": 85, "cutoff_c": 96, "missing_sensor": "hold"},
        "workloads": workload_definitions(),
        "request_payloads": request_payloads(),
        "pending": [
            "checkpoint content identity",
            "coherent serving OPP/bridge/binary hashes",
            "exact workload token IDs and verified model/tokenizer IDs",
            "fresh image content hash",
            "all-rank runtime provenance and real-weight hardware gates",
        ],
    }
    result = seal_manifest(value)
    validate_manifest(result)
    return result


def validate_manifest(value, root=None, *, require_live=False):
    """Validate internal identity; optionally verify committed source in Git.

    Returns the manifest on success, raises ValueError on inconsistent/pending
    admission. Offline preparation is valid with explicit pending payloads; it
    is not a promotion receipt. ``root`` never reads dirty source as a baseline.
    """
    try:
        return _validate_manifest(value, root, require_live=require_live)
    except (KeyError, TypeError, AttributeError, IndexError) as error:
        raise ValueError("malformed reference manifest") from error


def _validate_manifest(value, root, *, require_live):
    if (
        type(value.get("schema_version")) is not int
        or value["schema_version"] != SCHEMA_VERSION
        or value.get("kind") != "qwen_streaming_reference"
    ):
        raise ValueError("unsupported reference schema")
    if value.get("manifest_sha256") != seal_manifest(value)["manifest_sha256"]:
        raise ValueError("manifest digest mismatch")
    source = value["source"]
    assets = source["assets"]
    paths = [a["path"] for a in assets]
    if not assets or paths != sorted(set(paths)):
        raise ValueError("source assets must be nonempty, sorted and unique")
    for asset in assets:
        path = PurePosixPath(asset["path"])
        if path.is_absolute() or ".." in path.parts or not _hash(asset["sha256"]):
            raise ValueError("invalid source asset path or hash")
        if root is not None:
            actual = hashlib.sha256(_git(root, "show", f"{source['commit']}:{asset['path']}")).hexdigest()
            if actual != asset["sha256"]:
                raise ValueError("incompatible committed source asset")
    gitlinks = source.get("gitlinks", [])
    for gitlink in gitlinks:
        if len(gitlink["commit"]) != 40 or gitlink["content_status"] != "pending_external_bytes":
            raise ValueError("invalid external source commit")
        if root is not None:
            record = _git(root, "ls-tree", source["commit"], "--", gitlink["path"]).decode()
            if f"commit {gitlink['commit']}\t{gitlink['path']}" not in record:
                raise ValueError("external source commit mismatch")
    if source["source_sha256"] != source_sha256(assets, gitlinks):
        raise ValueError("source identity mismatch")
    if value["arithmetic_sha256"] != canonical_sha256(value["arithmetic"]):
        raise ValueError("numerical baseline identity mismatch")
    capacity = value["comparison_capacity"]
    profile = value["historical_runtime"]["profile"]
    if any(profile[k] != v for k, v in capacity.items()):
        raise ValueError("capacity differs from frozen reference")
    if value["historical_runtime"]["status"] != "thermally_unqualified":
        raise ValueError("historical thermal failure must not be labeled qualified")
    if value["live_runtime"] != {"status": "unknown", "queried": False}:
        raise ValueError("offline reference must not claim live runtime knowledge")
    workloads = value["workloads"]
    payload_revision = value.get("payload_revision", 1)
    payloads = value.get("request_payloads", {})
    if payload_revision == 2:
        if set(payloads) != set(request_payloads()):
            raise ValueError("frozen rendered request coverage incomplete")
        for payload in payloads.values():
            if payload["conversation_sha256"] != canonical_sha256(payload["conversation"]):
                raise ValueError("rendered conversation digest mismatch")
            if payload["request"]["messages"] != payload["conversation"]:
                raise ValueError("rendered request conversation mismatch")
            if payload["rendered_request_sha256"] != canonical_sha256(payload["request"]):
                raise ValueError("rendered request digest mismatch")
            if require_live and (
                not payload["request"].get("model")
                or not payload["tokenizer"].get("id")
                or not payload["tokenizer"].get("revision")
            ):
                raise ValueError("verified request model/tokenizer identity pending")
    elif payload_revision != 1:
        raise ValueError("unsupported rendered payload revision")
    templates = {w["id"]: w for w in workload_definitions()}
    expected = set(templates)
    if len(workloads) != len(expected) or {w["id"] for w in workloads} != expected:
        raise ValueError("incomplete or duplicate workload coverage")
    lookup = {w["id"]: w for w in workloads}
    for workload in workloads:
        template = templates[workload["id"]]
        for field in (
            "kind",
            "cache_policy",
            "submitted_requests",
            "max_active_requests",
            "mtp_tokens",
            "graph_capture_sizes",
            "target_prompt_tokens",
            "max_new_tokens",
            "warm_control",
            "warm_requirement",
            "mixed_schedule",
        ):
            if workload[field] != template[field]:
                raise ValueError("workload contract differs from frozen protocol")
        tokens = workload["token_ids"]
        if tokens is not None:
            if not isinstance(tokens, list) or not tokens or any(type(t) is not int or t < 0 for t in tokens):
                raise ValueError("token IDs must be nonnegative integer lists")
            if workload["token_ids_sha256"] != canonical_sha256(tokens):
                raise ValueError("token payload digest mismatch")
            if workload.get("actual_prompt_tokens") != len(tokens):
                raise ValueError("actual prompt token count differs from frozen token IDs")
        elif workload["token_ids_sha256"] is not None:
            raise ValueError("token digest without payload")
        if workload["max_active_requests"] != capacity["max_num_seqs"]:
            raise ValueError("workload silently changes capacity")
        if payload_revision == 2:
            payload = payloads.get(workload["request_payload_ref"])
            if payload is None or workload["conversation"] != {"payload_ref": workload["request_payload_ref"]}:
                raise ValueError("workload rendered payload reference missing")
            if (
                workload["conversation_sha256"] != payload["conversation_sha256"]
                or workload["rendered_request_sha256"] != payload["rendered_request_sha256"]
            ):
                raise ValueError("workload rendered request/conversation hash mismatch")
            if workload["max_new_tokens"] != payload["request"]["max_tokens"]:
                raise ValueError("rendered request generation limit differs")
        elif workload["conversation_sha256"] != (
            canonical_sha256(workload["conversation"]) if workload["conversation"] is not None else None
        ):
            raise ValueError("conversation digest mismatch")
        warm = workload["warm_control"]
        if workload["cache_policy"] == "warm":
            if warm not in lookup or lookup[warm]["cache_policy"] != "cold":
                raise ValueError("warm workload lacks identical cold control")
            control = lookup[warm]
            for field in (
                "token_ids",
                "conversation_sha256",
                "image_sha256",
                "max_new_tokens",
                "mtp_tokens",
                "submitted_requests",
            ):
                if workload[field] != control[field]:
                    raise ValueError("warm control payload/settings differ")
            if payload_revision == 2 and workload["rendered_request_sha256"] != control["rendered_request_sha256"]:
                raise ValueError("warm rendered request differs from cold control")
        elif workload["cache_policy"] != "cold" or warm is not None:
            raise ValueError("invalid cold/warm labeling")
        if require_live and (tokens is None or workload["input_status"] != "frozen"):
            raise ValueError("workload input evidence pending")
        if require_live and workload["kind"] in ("tool", "image", "mixed") and workload["conversation"] is None:
            raise ValueError("real conversation input pending")
        if require_live and workload["kind"] == "image" and not _hash(workload["image_sha256"]):
            raise ValueError("image content evidence pending")
    candidate = value.get("candidate")
    if candidate is not None:
        # T8 adds this independently sealed binding after candidate implementation.
        # The baseline source identity remains unchanged and is still required.
        candidate_body = {k: v for k, v in candidate.items() if k != "candidate_sha256"}
        if candidate.get("candidate_sha256") != canonical_sha256(candidate_body):
            raise ValueError("candidate identity mismatch")
        for field in ("assets", "binaries", "bridges"):
            entries = candidate.get(field, [])
            if not entries or len({entry["path"] for entry in entries}) != len(entries):
                raise ValueError("candidate assets/build artifacts missing or duplicate")
            if any(not _hash(entry.get("sha256")) for entry in entries):
                raise ValueError("candidate asset hash missing")
        if not _hash(candidate.get("contract_sha256")) or not candidate.get("native_components"):
            raise ValueError("candidate contract/components missing")
    if require_live:
        for field in ("checkpoint", "serving_bundle"):
            if not _hash(value[field]["sha256"]):
                raise ValueError(f"{field} identity evidence pending")
        if candidate is None:
            raise ValueError("candidate source/build binding pending")
    return value


def validate_evidence(evidence, manifest, *, require_live=False):
    """Validate saved all-rank evidence bound to one reference and arithmetic.

    Required keys: schema_version, manifest_sha256, source_sha256,
    arithmetic_sha256, checkpoint_sha256, serving_bundle_sha256, rank_receipts,
    trials, gates. Each rank receipt carries the same six identity fields plus
    integer rank/pid and execution_namespace. Gates need hardware scope, a
    content-addressed evidence artifact and complete ranks to admit live. This
    checks provenance structure; it does not authenticate claims or run gates.
    """
    try:
        return _validate_evidence(evidence, manifest, require_live=require_live)
    except (KeyError, TypeError, AttributeError, IndexError) as error:
        raise ValueError("malformed evidence receipt") from error


def _validate_evidence(evidence, manifest, *, require_live):
    validate_manifest(manifest, require_live=require_live)
    if type(evidence.get("schema_version")) is not int or evidence["schema_version"] != SCHEMA_VERSION:
        raise ValueError("unsupported evidence schema")
    identities = {
        "manifest_sha256": manifest["manifest_sha256"],
        "source_sha256": manifest["source"]["source_sha256"],
        "arithmetic_sha256": manifest["arithmetic_sha256"],
        "checkpoint_sha256": manifest["checkpoint"]["sha256"],
        "serving_bundle_sha256": manifest["serving_bundle"]["sha256"],
        "candidate_sha256": manifest["candidate"]["candidate_sha256"] if manifest.get("candidate") else None,
    }
    for key, expected in identities.items():
        if key not in evidence or evidence[key] != expected:
            raise ValueError(f"evidence {key} differs from reference")
    receipts = evidence.get("rank_receipts", [])
    ranks = [r.get("rank") for r in receipts]
    if any(type(r) is not int for r in ranks) or sorted(ranks) != list(EXPECTED_RANKS):
        raise ValueError("missing, duplicate or invalid rank receipts")
    for receipt in receipts:
        if type(receipt.get("pid")) is not int or receipt["pid"] <= 0 or not receipt.get("execution_namespace"):
            raise ValueError("rank execution identity missing")
        if any(key not in receipt or receipt[key] != expected for key, expected in identities.items()):
            raise ValueError("rank source/numerical baseline mismatch")
    if len({r["execution_namespace"] for r in receipts}) != 1:
        raise ValueError("rank execution namespaces differ")
    workloads = {w["id"]: w for w in manifest["workloads"]}
    for trial in evidence.get("trials", []):
        workload = workloads.get(trial.get("workload_id"))
        if workload is None or trial.get("cache_policy") != workload["cache_policy"]:
            raise ValueError("trial workload or cache label mismatch")
        for field in (
            "token_ids_sha256",
            "conversation_sha256",
            "image_sha256",
            "max_new_tokens",
            "mtp_tokens",
            "submitted_requests",
        ):
            if field not in trial or trial[field] != workload[field]:
                raise ValueError("trial payload or numerical mode mismatch")
        if (
            manifest.get("payload_revision", 1) == 2
            and trial.get("rendered_request_sha256") != workload["rendered_request_sha256"]
        ):
            raise ValueError("trial rendered request mismatch")
        computed, cached = trial.get("computed_prompt_tokens"), trial.get("cached_prompt_tokens")
        if type(computed) is not int or type(cached) is not int or min(computed, cached) < 0:
            raise ValueError("trial prompt accounting missing")
        if workload["token_ids"] is not None and computed + cached != len(workload["token_ids"]):
            raise ValueError("trial prompt accounting mismatch")
        if workload["cache_policy"] == "cold" and cached != 0:
            raise ValueError("cached work mislabeled cold")
        if workload["cache_policy"] == "warm" and cached == 0:
            raise ValueError("warm work lacks prefix reuse")
        if require_live and trial.get("hardware_validated") is not True:
            raise ValueError("offline trial cannot qualify live admission")
    gates = evidence.get("gates", {})
    for name, gate in gates.items():
        if name not in REQUIRED_GATES or gate.get("status") not in ("pending", "passed", "failed"):
            raise ValueError("invalid gate name/status")
        if gate["status"] == "passed":
            if gate.get("scope") != "hardware" or not _hash(gate.get("artifact_sha256")):
                raise ValueError("passed gate needs content-addressed hardware evidence")
            if gate.get("ranks") != list(EXPECTED_RANKS) or gate.get("manifest_sha256") != manifest["manifest_sha256"]:
                raise ValueError("gate rank/manifest coverage mismatch")
    if require_live:
        if any(gates.get(name, {}).get("status") != "passed" for name in REQUIRED_GATES):
            raise ValueError("hardware promotion gates pending or failed")
        covered = {trial["workload_id"] for trial in evidence.get("trials", [])}
        if covered != set(workloads):
            raise ValueError("hardware workload coverage incomplete")
        components = manifest["candidate"]["native_components"]
        native = evidence.get("native_component_evidence", {})
        for name in components:
            component = native.get(name, {})
            if component.get("status") != "passed" or component.get("scope") != "hardware":
                raise ValueError("native component hardware evidence pending")
            if component.get("candidate_sha256") != identities["candidate_sha256"]:
                raise ValueError("native component candidate hash mismatch")
            if component.get("ranks") != list(EXPECTED_RANKS) or not _hash(component.get("artifact_sha256")):
                raise ValueError("native component artifact/ranks missing")
    return evidence


def write_append_only(path, value):
    """Create one JSON receipt exclusively; never overwrite an existing path.

    This prevents clobbering, not partial-file visibility during a crash/write.
    Consumers must parse and validate the complete sealed JSON before use.
    """
    content = json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    with Path(path).open("x", encoding="utf-8") as stream:
        stream.write(content)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--revision", default="2a2a4e416")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    write_append_only(args.output, freeze_reference(args.root, args.revision))


if __name__ == "__main__":
    main()
