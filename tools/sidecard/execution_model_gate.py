# SPDX-License-Identifier: Apache-2.0
"""Teacher-forced full-state qualification, independent of performance timing.

T11 integration contract:
* Config execution_diagnostics.enabled must be true. Its authentication is an
  authenticated JSON reference {path, sha256}. That receipt binds identity
  {source_sha256, checkpoint_sha256, native_sha256}, architecture, normalized
  config_sha256, and files [{path, sha256, category: source|checkpoint|native}].
  Normalization removes only execution_diagnostics.authentication.sha256 to
  avoid a config/receipt hash cycle. A qualified reference additionally needs
  qualification {passed:true, default_off:true, evidence:{path,sha256}} whose
  evidence binds the same identity, architecture, normalized config_sha256,
  default_off=true and passed startup gates.
* engine.execution_diagnostics.capabilities() returns schema_version=1,
  complete_state=true, complete_selectors=true, graph_replay_observed=true,
  execution_path_sha256=canonical_sha256 of execution_path(spec), and owners.
  No eager substitute is allowed for a configured graph path. During capture,
  register persistent selector/output refs; after replay, synchronize that
  owner and observe the actual refs. Never invoke CPU capture inside a graph.
* begin_fixture(plan, directory) installs one DiagnosticWriter per owner;
  finish_phase('prefill'|'decode') captures final caches via capture_layers,
  drains both owners and waits for capture completion; end_fixture() closes
  writers and returns {owner: private_manifest_path}. All hooks are gate-only.
* Every logical row's handoff and actual selectors are appended with the plan
  key and offset. capture_selection pads DSA indices with -1 to index_topk+3.
  Full vocabulary [1,V] is appended to vocabulary at prefill completion and
  each teacher replay. Intermediate prompt chunk logits are appended under
  chunk_vocabulary with chunk-end positions in chunk_positions. Each owner's
  chunks [position,rows] are recorded as an int64 tensor, not inferred here.
* Block boundaries: cuda.prefix, npu-rank0.input/output,
  npu-rank1.input/output, cuda.suffix, all four deferred fields. Expert-only
  owners expose layers 18..32 FP16 input / FP32 partial [rows,H], int32 wire
  IDs and FP32 masked weights, with zeros for no local contributions. Owner
  schedules normalize rows to the CUDA chunks; the adapter must authenticate
  actual batch/skip receipts before declaring complete rank observations.

The CLI fails closed until these hooks are installed. Private requests, tools,
teacher IDs and tensors remain in the output's private directory. Reports bind
their hashes; neither hashes alone nor selected scalar samples replace complete
numerical comparison. Each architecture requires its own reference. Unused
uninitialized reserved capacity is excluded; allocation geometry is checked.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path

from .stage_diagnostics import DEFAULT_CHUNK_BYTES, DTYPE_BYTES, TENSOR_KINDS, private_json, sha256_file

OWNERS = ("cuda", "npu-rank0", "npu-rank1")
BLOCK_ARCHITECTURE = "cuda-prefix-ascend-middle-cuda-suffix"
EXPERT_ARCHITECTURE = "cuda-backbone-ascend-experts"
OUTPUT_TOKENS = 256
REQUIRED_FIXTURES = frozenset({"2K", "8K", "full-code"})
PATH_OPTIONS = (
    "cuda_decode_graphs",
    "npu_decode_graphs",
    "cuda_prefill_conv",
    "npu_native_bf16_round",
    "prefill_chunk_size",
    "prefill_window_size",
    "prefill_overlap",
    "direct_dma",
    "execution_candidates",
)


def canonical_sha256(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def execution_path(spec):
    """Options whose actual execution path the gate adapter must authenticate."""
    return {key: spec[key] for key in PATH_OPTIONS if key in spec}


def config_identity(spec):
    value = json.loads(json.dumps(spec))
    value["execution_diagnostics"]["authentication"].pop("sha256", None)
    return canonical_sha256(value)


def authenticated_json(reference, root):
    path = Path(reference["path"])
    if not path.is_absolute():
        path = root / path
    if sha256_file(path) != reference["sha256"]:
        raise ValueError(f"authentication failed: {path.name}")
    return json.loads(path.read_bytes()), path


def authenticate_config(spec, root):
    options = spec.get("execution_diagnostics", {})
    if options.get("enabled") is not True or spec.get("prompt_prefix_cache", False):
        raise ValueError("explicit diagnostic mode with prefix reuse disabled required")
    receipt, path = authenticated_json(options["authentication"], root)
    if receipt.get("config_sha256") != config_identity(spec):
        raise ValueError("authenticated normalized runtime configuration differs")
    architecture = spec.get("architecture", BLOCK_ARCHITECTURE)
    if architecture not in (BLOCK_ARCHITECTURE, EXPERT_ARCHITECTURE) or receipt.get("architecture") != architecture:
        raise ValueError("unsupported or unauthenticated architecture")
    identity = receipt["identity"]
    for name in ("source_sha256", "checkpoint_sha256", "native_sha256"):
        value = identity.get(name, "")
        if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
            raise ValueError("source/checkpoint/native SHA256 identities required")
    categories = set()
    for artifact in receipt.get("files", []):
        category = artifact["category"]
        if category not in ("source", "checkpoint", "native"):
            raise ValueError("unknown authenticated input category")
        target = Path(artifact["path"])
        target = target if target.is_absolute() else path.parent / target
        if sha256_file(target) != artifact["sha256"]:
            raise ValueError("source/native/checkpoint input changed")
        categories.add(category)
    if categories != {"source", "checkpoint", "native"}:
        raise ValueError("incomplete source/native/checkpoint authentication")
    return receipt


def validate_precision(contract):
    if contract.get("fixed_before_native_device_tests") is not True:
        raise ValueError("precision bounds must be fixed before native device tests")
    bounds = contract["model_bounds"]
    if bounds["same_math"] != {"all_tensors": "bitwise"}:
        raise ValueError("same-math contract must retain complete bitwise equality")
    changed = bounds["changed_math"]
    for kind in ("floating_state", "vocabulary"):
        if set(changed[kind]) != {"atol", "rtol"} or any(
            type(value) not in (int, float) or not math.isfinite(value) or value < 0 for value in changed[kind].values()
        ):
            raise ValueError("undeclared or invalid model precision bounds")
    if any(changed[kind] != "bitwise" for kind in ("selectors", "integer_boolean_state")):
        raise ValueError("selectors and integer state require exact parity")
    if changed.get("nonfinite") != "reject" or changed.get("unknown_tensor_kind") != "reject":
        raise ValueError("nonfinite and unknown state must fail closed")
    return contract


def chunks(length, chunk_rows, decode_steps):
    return [[position, min(chunk_rows, length - position)] for position in range(0, length, chunk_rows)] + [
        [length + step, 1] for step in range(decode_steps)
    ]


def make_plan(name, prompt, model, spec, *, output_tokens=OUTPUT_TOKENS):
    """Declare required observations independently of worker-provided manifests."""
    architecture = spec.get("architecture", BLOCK_ARCHITECTURE)
    layer_types = model["layer_types"]
    width = model["hidden_size"]
    head_dim, heads, streams, top_k = 128, 64, 4, 8
    chunk_rows = spec.get("prefill_chunk_size", 32)
    if type(chunk_rows) is not int or chunk_rows < 1 or output_tokens < 2 or not prompt:
        raise ValueError("invalid diagnostic chunk/output geometry")
    if architecture not in (BLOCK_ARCHITECTURE, EXPERT_ARCHITECTURE):
        raise ValueError("unrecognized diagnostic architecture")
    expected = {owner: {} for owner in OWNERS}
    allocations = {owner: {} for owner in OWNERS}

    def add(owner, key, shape, dtype, kind, value=None):
        expected[owner][key] = dict(shape=shape, dtype=dtype, kind=kind)
        if value is not None:
            expected[owner][key]["value"] = value

    length = len(prompt)
    schedule = chunks(length, chunk_rows, output_tokens - 1)
    for owner in OWNERS:
        add(owner, "chunks", [len(schedule), 2], "int64", "integer_boolean_state", schedule)
    for phase, rows in (("prefill", length), ("decode", output_tokens - 1)):
        if architecture == BLOCK_ARCHITECTURE:
            for owner, boundaries in (
                ("cuda", ("prefix", "suffix")),
                ("npu-rank0", ("input", "output")),
                ("npu-rank1", ("input", "output")),
            ):
                for boundary in boundaries:
                    for field, shape, dtype in (
                        ("hidden", [rows, width], "float16"),
                        ("residual", [rows, streams, width], "float32"),
                        ("post", [rows, streams, 1], "float32"),
                        ("comb", [rows, streams, streams], "float32"),
                    ):
                        add(owner, f"{phase}/boundary/{boundary}/{field}", shape, dtype, "floating_state")
        for index, layer_type in enumerate(layer_types):
            owners = ("cuda",) if architecture == EXPERT_ARCHITECTURE or index < 18 or index >= 33 else OWNERS[1:]
            for owner in owners:
                prefix = f"{phase}/selection/layer{index}"
                if index >= 3:
                    add(owner, f"{prefix}/route", [rows, top_k], "int64", "selectors")
                    add(owner, f"{prefix}/route_weights", [rows, top_k], "float32", "floating_state")
                if layer_type != "linear_attention":
                    add(owner, f"{prefix}/indexer", [rows, model["index_topk"] + 3], "int64", "selectors")
            if architecture == EXPERT_ARCHITECTURE and 18 <= index < 33:
                for field, shape, dtype, kind in (
                    ("input", [rows, width], "float16", "floating_state"),
                    ("ids", [rows, top_k], "int32", "selectors"),
                    ("weights", [rows, top_k], "float32", "floating_state"),
                ):
                    add("cuda", f"{phase}/expert/layer{index}/{field}", shape, dtype, kind)
                for owner in OWNERS[1:]:
                    for field, shape, dtype, kind in (
                        ("input", [rows, width], "float16", "floating_state"),
                        ("output", [rows, width], "float32", "floating_state"),
                        ("ids", [rows, top_k], "int32", "selectors"),
                        ("weights", [rows, top_k], "float32", "floating_state"),
                    ):
                        add(owner, f"{phase}/expert/layer{index}/{field}", shape, dtype, kind)
    for phase, end in (("prefill", length), ("decode", length + output_tokens - 1)):
        for index, layer_type in enumerate(layer_types):
            owners = ("cuda",) if architecture == EXPERT_ARCHITECTURE or index < 18 or index >= 33 else OWNERS[1:]
            for owner in owners:
                prefix = f"{phase}/state/layer{index}"
                # Existing KDA TextLayer.length stays zero; the owner schedule
                # binds its logical token position without changing that field.
                add(
                    owner,
                    f"{prefix}/length",
                    [1],
                    "int64",
                    "integer_boolean_state",
                    [0 if layer_type == "linear_attention" else end],
                )
                local_heads = heads if owner == "cuda" else heads // 2
                if layer_type == "linear_attention":
                    add(owner, f"{prefix}/recurrent", [local_heads, head_dim, head_dim], "float32", "floating_state")
                    add(owner, f"{prefix}/conv", [3 * local_heads * head_dim, 3], "float16", "floating_state")
                    allocations[owner][str(index)] = dict(
                        recurrent=[local_heads, head_dim, head_dim], conv=[3 * local_heads * head_dim, 3]
                    )
                else:
                    for field, shape, dtype in (
                        ("kv", [end, model["kv_lora_rank"]], "float16"),
                        ("pools", [end // 4, head_dim], "float16"),
                        ("tail_keys", [end % 4, head_dim], "float32"),
                        ("tail_gates", [end % 4, head_dim], "float32"),
                    ):
                        add(owner, f"{prefix}/{field}", shape, dtype, "floating_state")
                    add(owner, f"{prefix}/pool_count", [1], "int64", "integer_boolean_state", [end // 4])
                    add(owner, f"{prefix}/tail_length", [1], "int64", "integer_boolean_state", [end % 4])
                    allocations[owner][str(index)] = dict(
                        kv=[spec["max_context"], model["kv_lora_rank"]],
                        pools=[spec["max_context"] // 4, head_dim],
                        tail_keys=[3, head_dim],
                        tail_gates=[3, head_dim],
                    )
    add("cuda", "vocabulary", [output_tokens, model["vocab_size"]], "float32", "vocabulary")
    positions = [[position + rows - 1] for position, rows in schedule[: len(schedule) - output_tokens + 1]]
    add("cuda", "chunk_positions", [len(positions), 1], "int64", "integer_boolean_state", positions)
    add("cuda", "chunk_vocabulary", [len(positions), model["vocab_size"]], "float32", "vocabulary")
    plan = dict(
        schema_version=1,
        name=name,
        architecture=architecture,
        prompt_tokens=length,
        prompt_ids_sha256=canonical_sha256(prompt),
        output_tokens=output_tokens,
        schedule=schedule,
        expected=expected,
        allocation_geometry=allocations,
        execution_path_sha256=canonical_sha256(execution_path(spec)),
    )
    if architecture == EXPERT_ARCHITECTURE:
        window = spec.get("prefill_window_size", chunk_rows)
        if type(window) is not int or window < chunk_rows or window % chunk_rows:
            raise ValueError("expert diagnostic windows must preserve complete CUDA chunks")
        middle = list(range(18, min(33, len(layer_types))))
        plan["expert_physical_schedule"] = [
            dict(phase="prefill", position=position, rows=min(window, length - position), layer=layer)
            for position in range(0, length, window)
            for layer in middle
        ] + [
            dict(phase="decode", position=length + step, rows=1, layer=layer)
            for step in range(output_tokens - 1)
            for layer in middle
        ]
        plan["expert_execution"] = {
            name: spec.get("direct_dma", {}).get(name, False)
            for name in ("npu_expert_graphs", "npu_prefill_graphs", "npu_single_route_graphs")
        }
    return plan


def _contained(root, name):
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()) or Path(name).is_absolute():
        raise ValueError("private tensor path escapes diagnostic directory")
    return path


def validate_capture(reference, expected, allocation_geometry, *, expert_schedule=None, expert_execution=None):
    """Authenticate complete data and geometry, not only a 'complete' flag."""
    import numpy as np

    path = Path(reference)
    body = json.loads(path.read_bytes())
    if body.get("schema_version") != 1 or body.get("complete") is not True:
        raise ValueError("incomplete owner diagnostics")
    if body.get("allocation_geometry") != allocation_geometry:
        raise ValueError("cache reservation/allocation geometry changed")
    records = body.get("tensors", [])
    by_key = {record["key"]: record for record in records}
    if len(by_key) != len(records) or set(by_key) != set(expected):
        raise ValueError("missing, duplicate or unknown rank/state/boundary coverage")
    for key, declared in expected.items():
        record = by_key[key]
        if any(record.get(field) != declared[field] for field in ("shape", "dtype", "kind")):
            raise ValueError(f"diagnostic tensor shape/dtype/kind changed: {key}")
        offset = 0
        for segment in record["segments"]:
            if segment["offset"] != offset or type(segment["rows"]) is not int or segment["rows"] < 0:
                raise ValueError("incomplete/reordered diagnostic chunk observation")
            offset += segment["rows"]
        if offset != declared["shape"][0] or not record["segments"]:
            raise ValueError("incomplete chunk/tail observations")
        target = _contained(path.parent, record["path"])
        count = math.prod(declared["shape"])
        size = count * DTYPE_BYTES[declared["dtype"]]
        if record["bytes"] != size or target.stat().st_size != size or sha256_file(target) != record["sha256"]:
            raise ValueError("private diagnostic tensor changed or was truncated")
        if "value" in declared:
            observed = np.fromfile(target, dtype=declared["dtype"]).reshape(declared["shape"])
            if not np.array_equal(observed, np.asarray(declared["value"])):
                raise ValueError(f"owner-local position/count/schedule changed: {key}")
    if body.get("owner") != "cuda" and any("/expert/" in key for key in expected):
        if expert_schedule is None or expert_execution is None:
            raise ValueError("expert native coverage requires authenticated physical calls/skips")
        from .expert_diagnostics import validate_expert_provenance

        validate_expert_provenance(body, path.parent, by_key, expert_schedule, expert_execution)
    return body, by_key


def compare_tensor(left_path, right_path, dtype, kind, mode, contract, *, left_offset=0, right_offset=0, elements=None):
    """Read bounded full data; report every element checked and worst errors."""
    import numpy as np

    changed = contract["model_bounds"]["changed_math"]
    if kind not in TENSOR_KINDS:
        raise ValueError("unknown diagnostic tensor kind")
    bound = "bitwise" if mode == "same_math" else changed.get(kind, "reject")
    if bound == "reject":
        raise ValueError("unknown diagnostic tensor kind")
    floating = dtype.startswith("float")
    itemsize = DTYPE_BYTES[dtype]
    chunk_size = DEFAULT_CHUNK_BYTES // itemsize * itemsize
    passed, bitwise, compared, maximum, relative = True, True, 0, 0.0, 0.0
    with Path(left_path).open("rb") as left, Path(right_path).open("rb") as right:
        left.seek(left_offset)
        right.seek(right_offset)
        remaining = None if elements is None else elements * itemsize
        while True:
            if remaining == 0:
                break
            size = chunk_size if remaining is None else min(chunk_size, remaining)
            a, b = left.read(size), right.read(size)
            if not a and not b:
                if remaining:
                    raise ValueError("incomplete bounded tensor range")
                break
            if len(a) != len(b) or len(a) % itemsize:
                raise ValueError("full tensor comparison received unequal/truncated geometry")
            bitwise = bitwise and a == b
            x, y = np.frombuffer(a, dtype=dtype), np.frombuffer(b, dtype=dtype)
            compared += len(x)
            if floating:
                if not np.isfinite(x).all() or not np.isfinite(y).all():
                    raise ValueError("nonfinite full-state/vocabulary observation")
                error = np.abs(x.astype(np.float64) - y.astype(np.float64))
                if len(error):
                    maximum = max(maximum, float(error.max()))
                    relative = max(relative, float((error / np.maximum(np.abs(x.astype(np.float64)), 1e-30)).max()))
                if isinstance(bound, dict):
                    passed = passed and bool(np.all(error <= bound["atol"] + bound["rtol"] * np.abs(x)))
            if bound == "bitwise":
                passed = passed and a == b
            if remaining is not None:
                remaining -= len(a)
    return dict(
        passed=passed,
        bitwise_equal=bitwise,
        elements=compared,
        max_abs_error=maximum,
        max_relative_error=relative,
        bound=bound,
    )


def compare_captures(candidate, reference, candidate_plan, reference_plan, mode, contract):
    validate_precision(contract)
    if mode not in ("same_math", "changed_math"):
        raise ValueError("explicit same_math or changed_math comparison required")
    for name in ("architecture", "prompt_ids_sha256", "output_tokens", "name"):
        if candidate_plan[name] != reference_plan[name]:
            raise ValueError("architecture, prompt or teacher replay coverage differs")
    if set(candidate) != set(OWNERS) or set(reference) != set(OWNERS):
        raise ValueError("both rank owners and CUDA diagnostics required")
    results = []
    for owner in OWNERS:
        current_body, current = validate_capture(
            candidate[owner],
            candidate_plan["expected"][owner],
            candidate_plan["allocation_geometry"][owner],
            expert_schedule=candidate_plan.get("expert_physical_schedule"),
            expert_execution=candidate_plan.get("expert_execution"),
        )
        baseline_body, baseline = validate_capture(
            reference[owner],
            reference_plan["expected"][owner],
            reference_plan["allocation_geometry"][owner],
            expert_schedule=reference_plan.get("expert_physical_schedule"),
            expert_execution=reference_plan.get("expert_execution"),
        )
        if current_body["owner"] != owner or baseline_body["owner"] != owner:
            raise ValueError("diagnostic rank owner changed")
        if (
            owner != "cuda"
            and "expert_provenance" in current_body
            and (
                current_body["expert_provenance"]["hot_ids_by_layer"]
                != baseline_body["expert_provenance"]["hot_ids_by_layer"]
            )
        ):
            raise ValueError("expert placement changed outside this qualified architecture")
        if set(current) != set(baseline):
            raise ValueError("candidate changed complete layer/state coverage")
        if candidate_plan["allocation_geometry"][owner] != reference_plan["allocation_geometry"][owner]:
            raise ValueError("candidate changed reserved cache geometry")
        for key, record in current.items():
            if key in ("chunks", "chunk_positions", "chunk_vocabulary"):
                continue
            previous = baseline[key]
            if any(record[field] != previous[field] for field in ("shape", "dtype", "kind")):
                raise ValueError("candidate changed complete state geometry")
            result = compare_tensor(
                Path(reference[owner]).parent / previous["path"],
                Path(candidate[owner]).parent / record["path"],
                record["dtype"],
                record["kind"],
                mode,
                contract,
            )
            results.append(dict(owner=owner, key=key, **result))
        # All produced chunk logits are checked, including those without a
        # matching reference endpoint. Common endpoints additionally compare.
        if owner == "cuda":
            import numpy as np

            ends = {}
            for label, records, captures in (("candidate", current, candidate), ("reference", baseline, reference)):
                positions_record, logits_record = records["chunk_positions"], records["chunk_vocabulary"]
                root = Path(captures[owner]).parent
                positions = np.fromfile(root / positions_record["path"], dtype="int64").tolist()
                ends[label] = dict(zip(positions, range(len(positions)), strict=True))
                compare_tensor(
                    root / logits_record["path"],
                    root / logits_record["path"],
                    "float32",
                    "vocabulary",
                    "same_math",
                    contract,
                )
            for position in ends["candidate"].keys() & ends["reference"].keys():
                width = current["chunk_vocabulary"]["shape"][1]
                if width != baseline["chunk_vocabulary"]["shape"][1]:
                    raise ValueError("full vocabulary geometry changed")
                result = compare_tensor(
                    Path(reference[owner]).parent / baseline["chunk_vocabulary"]["path"],
                    Path(candidate[owner]).parent / current["chunk_vocabulary"]["path"],
                    "float32",
                    "vocabulary",
                    mode,
                    contract,
                    left_offset=ends["reference"][position] * width * 4,
                    right_offset=ends["candidate"][position] * width * 4,
                    elements=width,
                )
                results.append(dict(owner=owner, key=f"chunk_vocabulary@{position}", **result))
    if candidate_plan["architecture"] == EXPERT_ARCHITECTURE:
        from .expert_diagnostics import validate_expert_handoffs

        validate_expert_handoffs(candidate, candidate_plan)
        validate_expert_handoffs(reference, reference_plan)
    return dict(
        passed=all(result["passed"] for result in results),
        tensors=results,
        unmatched_chunk_logits="retained and checked finite; only common logical endpoints are numerically comparable",
    )


def validate_capabilities(adapter, plan):
    capabilities = adapter.capabilities()
    if capabilities.get("schema_version") != 1 or set(capabilities.get("owners", [])) != set(OWNERS):
        raise ValueError("complete authenticated owner diagnostics unavailable")
    if any(
        capabilities.get(name) is not True for name in ("complete_state", "complete_selectors", "graph_replay_observed")
    ):
        raise ValueError("complete real execution/graph selector observation unavailable")
    if capabilities.get("execution_path_sha256") != plan["execution_path_sha256"]:
        raise ValueError("diagnostic eager/graph path differs from configured runtime")


def validate_fixtures(manifest):
    fixtures = manifest.get("fixtures", [])
    if (
        manifest.get("schema_version") != 1
        or {row["name"] for row in fixtures} != REQUIRED_FIXTURES
        or len(fixtures) != 3
    ):
        raise ValueError("exact 2K, 8K and original full Code fixtures required")
    for fixture in fixtures:
        if canonical_sha256(fixture["request"]) != fixture["request_sha256"]:
            raise ValueError("private original request/messages/tools changed")
        if (
            fixture["name"] in ("2K", "8K")
            and fixture["expected_prompt_tokens"] != {"2K": 2048, "8K": 8192}[fixture["name"]]
        ):
            raise ValueError("synthetic fixture length changed")
        if len(fixture["request"].get("tools", [])) != fixture["expected_tool_count"]:
            raise ValueError("original tool definitions missing")
    return fixtures


def load_reference(path, digest, authentication, contract_digest):
    if digest is None:
        raise ValueError("reference SHA256 required")
    body, _ = authenticated_json({"path": str(Path(path).resolve()), "sha256": digest}, Path.cwd())
    if body.get("passed") is not True or body.get("reference_qualified") is not True:
        raise ValueError("unqualified or incomplete reference")
    if body["precision_contract_sha256"] != contract_digest:
        raise ValueError("precision contract changed after reference qualification")
    if (
        body["architecture"] != authentication["architecture"]
        or body["identity"]["checkpoint_sha256"] != authentication["identity"]["checkpoint_sha256"]
    ):
        raise ValueError("each architecture/checkpoint requires its own reference")
    if len(body["workloads"]) != 3 or {row["name"] for row in body["workloads"]} != REQUIRED_FIXTURES:
        raise ValueError("reference omitted full 2K/8K/Code qualification")
    return body


def qualify_default_reference(authentication, spec, root):
    qualification = authentication.get("qualification", {})
    if qualification.get("passed") is not True or qualification.get("default_off") is not True:
        raise ValueError("reference requires authenticated default-off qualification")
    evidence, _ = authenticated_json(qualification["evidence"], root)
    if (
        evidence.get("passed") is not True
        or evidence.get("default_off") is not True
        or evidence.get("identity") != authentication["identity"]
        or evidence.get("architecture") != authentication["architecture"]
        or evidence.get("config_sha256") != config_identity(spec)
    ):
        raise ValueError("default-off reference qualification identity/configuration differs")


def load_teacher(workload, root, plan, vocab_size):
    teacher, _ = authenticated_json(workload["teacher_ids"], root)
    forced = teacher["ids"]
    if (
        teacher["prompt_ids_sha256"] != plan["prompt_ids_sha256"]
        or len(forced) != plan["output_tokens"]
        or any(type(token) is not int or not 0 <= token < vocab_size for token in forced)
    ):
        raise ValueError("teacher IDs or their prompt binding changed")
    return forced


def run(config, fixtures, output, precision_contract, *, reference=None, reference_sha256=None):
    """Own one fresh runtime; adapters are supplied by explicit T11 hooks."""
    config, fixtures, output, precision_contract = map(Path, (config, fixtures, output, precision_contract))
    spec = json.loads(config.read_bytes())
    authentication = authenticate_config(spec, config.parent)
    contract = validate_precision(json.loads(precision_contract.read_bytes()))
    cases = validate_fixtures(json.loads(fixtures.read_bytes()))
    model = json.loads((Path(spec["checkpoint"]) / "config.json").read_bytes())["text_config"]
    model.setdefault("vocab_size", 154880)
    mode = spec["execution_diagnostics"].get("math_mode", "same_math")
    if mode not in ("same_math", "changed_math"):
        raise ValueError("explicit same_math or changed_math comparison required")
    previous = None
    if reference is not None:
        previous = load_reference(reference, reference_sha256, authentication, sha256_file(precision_contract))
    elif reference_sha256 is not None:
        raise ValueError("reference digest supplied without reference path")
    else:
        qualify_default_reference(authentication, spec, config.parent)
    output.parent.mkdir(parents=True, exist_ok=True)
    private_root = output.with_suffix(output.suffix + ".private")
    private_root.mkdir(mode=0o700, exist_ok=False)
    report = dict(
        schema_version=1,
        passed=False,
        reference_qualified=False,
        architecture=authentication["architecture"],
        identity=authentication["identity"],
        authentication=spec["execution_diagnostics"]["authentication"],
        raw_config_sha256=sha256_file(config),
        config_sha256=config_identity(spec),
        fixture_manifest_sha256=sha256_file(fixtures),
        precision_contract_sha256=sha256_file(precision_contract),
        production_performance_validated=False,
        workloads=[],
    )
    engine = None
    try:
        from transformers import AutoTokenizer

        from .expert_actor import ExpertCoordinator
        from .layer_actor import Coordinator
        from .layer_server import request_tokens

        tokenizer = AutoTokenizer.from_pretrained(spec["checkpoint"], trust_remote_code=False)
        engine = (
            ExpertCoordinator(config) if authentication["architecture"] == EXPERT_ARCHITECTURE else Coordinator(config)
        )
        adapter = getattr(engine, "execution_diagnostics", None)
        if adapter is None:
            raise ValueError("T11 authenticated diagnostic hooks are not installed")
        for fixture in cases:
            tokens, _ = request_tokens(
                fixture["request"], tokenizer, engine.max_context, chat="messages" in fixture["request"]
            )
            if len(tokens) != fixture["expected_prompt_tokens"] or len(tokens) + OUTPUT_TOKENS > engine.max_context:
                raise ValueError("tokenized original prompt geometry changed")
            plan = make_plan(fixture["name"], tokens, model, spec)
            validate_capabilities(adapter, plan)
            old = (
                next((row for row in previous["workloads"] if row["name"] == fixture["name"]), None)
                if previous
                else None
            )
            if previous and (old is None or old["request_sha256"] != fixture["request_sha256"]):
                raise ValueError("original reference prompts/tools changed")
            forced = []
            if old:
                forced = load_teacher(old, Path(reference).parent, plan, model["vocab_size"])
            engine.reset(context_limit=len(tokens) + OUTPUT_TOKENS)
            directory = private_root / fixture["name"]
            directory.mkdir(mode=0o700)
            adapter.begin_fixture(plan, directory)
            logits = engine.prefill(tokens)
            adapter.finish_phase("prefill")
            for step in range(OUTPUT_TOKENS):
                if not old:
                    forced.append(int(logits.argmax().cpu()))
                if step < OUTPUT_TOKENS - 1:
                    logits = engine.forward([forced[step]])
            adapter.finish_phase("decode")
            captures = adapter.end_fixture()
            if set(captures) != set(OWNERS):
                raise ValueError("missing CUDA or NPU rank diagnostics")
            for owner in OWNERS:
                body, records = validate_capture(
                    captures[owner],
                    plan["expected"][owner],
                    plan["allocation_geometry"][owner],
                    expert_schedule=plan.get("expert_physical_schedule"),
                    expert_execution=plan.get("expert_execution"),
                )
                if body["owner"] != owner:
                    raise ValueError("diagnostic rank owner changed")
                for record in records.values():
                    target = Path(captures[owner]).parent / record["path"]
                    compare_tensor(target, target, record["dtype"], record["kind"], "same_math", contract)
            if plan["architecture"] == EXPERT_ARCHITECTURE:
                from .expert_diagnostics import validate_expert_handoffs

                validate_expert_handoffs(captures, plan)
            teacher_path = directory / "teacher-ids.json"
            private_json(teacher_path, dict(ids=forced, prompt_ids_sha256=plan["prompt_ids_sha256"]))
            workload = dict(
                name=fixture["name"],
                request_sha256=fixture["request_sha256"],
                plan=plan,
                teacher_ids=dict(path=str(teacher_path.resolve()), sha256=sha256_file(teacher_path)),
                captures={
                    owner: dict(path=str(Path(path).resolve()), sha256=sha256_file(path))
                    for owner, path in captures.items()
                },
            )
            if old:
                baseline = {}
                for owner in OWNERS:
                    _, path = authenticated_json(old["captures"][owner], Path(reference).parent)
                    baseline[owner] = path
                workload["comparison"] = compare_captures(captures, baseline, plan, old["plan"], mode, contract)
                if not workload["comparison"]["passed"]:
                    report["workloads"].append(workload)
                    raise AssertionError("complete model state/selector/vocabulary numerical gate failed")
            report["workloads"].append(workload)
            private_json(output, report)
        report.update(
            passed=True,
            reference_qualified=previous is None,
            coverage="complete deferred rows, selectors, final caches, vocabulary and owner schedules",
        )
        return report
    except BaseException as error:
        report["error"] = dict(type=type(error).__name__, message=str(error))
        raise
    finally:
        try:
            if engine is not None:
                engine.close()
            authenticate_config(spec, config.parent)
        except BaseException as error:
            report.update(
                passed=False,
                reference_qualified=False,
                shutdown_error=dict(type=type(error).__name__, message=str(error)),
            )
            raise
        finally:
            private_json(output, report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--precision-contract", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--reference-sha256")
    args = parser.parse_args()
    run(
        args.config,
        args.fixtures,
        args.output,
        args.precision_contract,
        reference=args.reference,
        reference_sha256=args.reference_sha256,
    )


if __name__ == "__main__":
    main()
