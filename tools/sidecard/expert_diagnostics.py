# SPDX-License-Identifier: Apache-2.0
"""Full expert-architecture diagnostics with witnessed native calls and skips.

T11 hooks:
* ExpertCoordinatorDiagnostic(config_path, all_cuda_layers, owner_command,
  synchronize) is the T19 adapter. Attach .observer before graph construction.
  Existing route/selection/capture/replayed hooks apply; selection must pass
  rows=count, including a None dense result. This architecture has no deferred
  mHC handoff, so suffix_state validates its inputs without recording a boundary.
* For ExpertCoordinator.forward use observer.begin_chunk(position, rows), then
  vocabulary(logits), end_chunk(). For prefill_window use begin_window(position,
  window_rows), retain original layer-major CUDA chunks, call vocabulary(logits,
  position=chunk_position, rows=chunk_rows) for EACH produced vector, end_window().
  Eager selectors are observed at their actual callbacks before temporary refs
  can be overwritten. Graph callbacks retain refs only; replayed fences/copies.
* DmaDispatch calls adapter.dispatch_call(layer, x, wire_ids, cold_weights,
  active, sequences) BEFORE sending the real native COMMAND tags. These are
  actual FP16 x, int32 wire IDs and FP32 hot-masked weights. The adapter infers
  logical position from the active window and authenticates physical batching.
* Native ExpertOwner(config_path, rank, hot_ids_by_layer, synchronize) dispatches
  begin/phase/end/call/skip JSON through handle_command. 'call' prepares one
  authenticated physical invocation; AFTER the actual actor/graph returns and
  its stream completes, BEFORE its buffers are reused, WorkerSlots.compute calls
  observe_compute(sequence, layer, x, ids, weights, partial, execution=...).
  execution is actual eager/decode-graph/prefill-graph/single-route-graph path.
  Metadata must match the real COMMAND, upload and returned FP32 partial.

Control payloads stay under 16KiB and carry nonce/plan hash. Skip inputs are
private bounded disk artifacts captured from CUDA; the native owner independently
verifies no positive owned route, then records a provenance-tagged zero. A skip
is never described as actual native calculation. Both native owners authenticate
every physical layer/window, including skips, before normalizing logical rows.
Concurrent windows/pipelining are rejected, never silently serialized here.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path

from .execution_diagnostic_runtime import (
    CoordinatorDiagnostic,
    OwnerDiagnostic,
    _handoff_private_files,
    _inside,
    encode_command,
    validate_runtime,
)
from .execution_model_gate import EXPERT_ARCHITECTURE, config_identity, validate_capture
from .stage_diagnostics import (
    DEFAULT_CHUNK_BYTES,
    DTYPE_BYTES,
    DiagnosticWriter,
    capture_chunk,
    capture_route,
    capture_selection,
    capture_vocabulary,
    private_json,
    sha256_file,
)

NORMALIZATION = (
    "logical rows backed by complete authenticated physical calls; skipped outputs are proven zero contributions"
)
EXPERTS_PER_RANK = 144


def _array(value):
    import numpy as np

    return value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)


def _span_hash(path, offset, size):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        source.seek(offset)
        remaining = size
        while remaining:
            data = source.read(min(DEFAULT_CHUNK_BYTES, remaining))
            if not data:
                raise ValueError("truncated authenticated physical tensor span")
            digest.update(data)
            remaining -= len(data)
    return digest.hexdigest()


def _route_count(ids, weights, rank, hot_ids):
    import numpy as np

    if ids.dtype != np.int32 or weights.dtype != np.float32 or ids.shape != weights.shape or ids.shape[1] != 8:
        raise ValueError("actual native wire IDs/weights changed dtype or geometry")
    if not np.isfinite(weights).all() or (weights < 0).any() or (ids < 0).any() or (ids >= 288).any():
        raise ValueError("invalid actual native routing metadata")
    if np.any(np.isin(ids, list(hot_ids)) & (weights != 0)):
        raise ValueError("hot CUDA routes were not masked in actual native inputs")
    return int(
        np.count_nonzero((ids >= rank * EXPERTS_PER_RANK) & (ids < (rank + 1) * EXPERTS_PER_RANK) & (weights > 0))
    )


def _execution(rows, count, options):
    if rows == 1 and options.get("npu_expert_graphs"):
        return "single-route-graph" if count == 1 and options.get("npu_single_route_graphs") else "decode-graph"
    if rows == 32 and options.get("npu_expert_graphs") and options.get("npu_prefill_graphs"):
        return "prefill-graph"
    return "eager"


def _skip_inputs(reference, root, call, nonce, width):
    import numpy as np

    path = _inside(root, reference["path"])
    if sha256_file(path) != reference["sha256"]:
        raise ValueError("authenticated skipped CUDA input artifact changed")
    body = json.loads(path.read_bytes())
    if body.get("source_owner") != "cuda" or body.get("call") != call or body.get("nonce") != nonce:
        raise ValueError("skip proof belongs to another owner, invocation or nonce")
    records = {record["key"]: record for record in body["tensors"]}
    if len(records) != 3 or set(records) != {"input", "ids", "weights"}:
        raise ValueError("incomplete actual CUDA skip input proof")
    arrays = {}
    for key, dtype, columns in (("input", "float16", width), ("ids", "int32", 8), ("weights", "float32", 8)):
        record = records[key]
        target = _inside(path.parent, str(path.parent / record["path"]))
        size = call["rows"] * columns * DTYPE_BYTES[dtype]
        if (
            record["shape"] != [call["rows"], columns]
            or record["dtype"] != dtype
            or target.stat().st_size != size
            or record["bytes"] != size
            or sha256_file(target) != record["sha256"]
        ):
            raise ValueError("actual CUDA skip tensor geometry/data changed")
        arrays[key] = np.memmap(target, mode="r", dtype=dtype, shape=(call["rows"], columns))
    return arrays


def validate_expert_provenance(body, directory, records, schedule, options):
    """Independent full-byte witness validation invoked by the T19 model gate."""
    import numpy as np

    proof = body.get("expert_provenance", {})
    owner = body["owner"]
    if owner not in ("npu-rank0", "npu-rank1"):
        raise ValueError("unexpected native expert proof owner")
    rank = int(owner[-1])
    calls = proof.get("calls", [])
    if (
        proof.get("schema_version") != 1
        or proof.get("complete") is not True
        or proof.get("normalization") != NORMALIZATION
        or len(calls) != len(schedule)
    ):
        raise ValueError("missing complete physical expert call/skip provenance")
    hot = proof["hot_ids_by_layer"]
    expected_layers = {str(item["layer"]) for item in schedule}
    if set(hot) != expected_layers:
        raise ValueError("native expert proof omitted its exact placement")
    if any(type(value) is not bool for value in options.values()):
        raise ValueError("expert graph execution options must retain boolean configuration")
    next_sequence = None
    for index, (receipt, declared) in enumerate(zip(calls, schedule, strict=True)):
        call = receipt["call"]
        if (
            any(call.get(key) != value for key, value in declared.items())
            or set(call) != set(declared) | {"rank", "sequence", "active", "index"}
            or call.get("rank") != rank
            or call.get("index") != index
            or type(call.get("active")) is not bool
            or type(call.get("sequence")) is not int
            or call["sequence"] < 0
        ):
            raise ValueError("physical expert layer/position/row/rank schedule changed")
        if next_sequence is not None and call["sequence"] != next_sequence:
            raise ValueError("physical expert sequence skipped/repeated")
        next_sequence = call["sequence"] + int(call["active"])
        spans = receipt["tensor_spans"]
        if set(spans) != {"input", "ids", "weights", "output"}:
            raise ValueError("incomplete physical input/output witness")
        offset = call["position"] if call["phase"] == "prefill" else call["position"] - proof["prompt_tokens"]
        metadata = {}
        for field, span in spans.items():
            key = f"{call['phase']}/expert/layer{call['layer']}/{field}"
            record = records[key]
            row_bytes = math.prod(record["shape"][1:]) * DTYPE_BYTES[record["dtype"]]
            if span != dict(key=key, offset=offset, rows=call["rows"], sha256=span.get("sha256")):
                raise ValueError("physical tensor witness changed logical row ownership")
            path = directory / record["path"]
            if _span_hash(path, offset * row_bytes, call["rows"] * row_bytes) != span["sha256"]:
                raise ValueError("physical witness no longer matches complete captured tensor")
            if field in ("ids", "weights"):
                metadata[field] = np.memmap(
                    path, mode="r", dtype=record["dtype"], offset=offset * row_bytes, shape=(call["rows"], 8)
                )
        count = _route_count(metadata["ids"], metadata["weights"], rank, hot[str(call["layer"])])
        if count != receipt.get("positive_owned_routes"):
            raise ValueError("native call ownership count was not witnessed")
        if call["active"]:
            if (
                count < 1
                or receipt.get("status") != "executed"
                or receipt.get("execution") != _execution(call["rows"], count, options)
            ):
                raise ValueError("native computation or graph replay path was not observed")
        else:
            if count != 0 or receipt.get("status") != "verified_zero_skip" or receipt.get("execution") != "skipped":
                raise ValueError("skipped native contribution was not independently proven zero")
            inputs = _skip_inputs(
                receipt["skip_inputs"],
                directory.parent.parent,
                call,
                proof["nonce"],
                records[spans["input"]["key"]]["shape"][1],
            )
            for field in ("input", "ids", "weights"):
                if hashlib.sha256(memoryview(inputs[field]).cast("B")).hexdigest() != spans[field]["sha256"]:
                    raise ValueError("skip witness differs from actual CUDA input artifact")
            output = records[spans["output"]["key"]]
            width = output["shape"][1]
            values = np.memmap(
                directory / output["path"],
                mode="r",
                dtype="float32",
                offset=offset * width * 4,
                shape=(call["rows"], width),
            )
            row_chunk = max(1, DEFAULT_CHUNK_BYTES // (width * 4))
            if any(np.count_nonzero(values[start : start + row_chunk]) for start in range(0, call["rows"], row_chunk)):
                raise ValueError("skipped expert output contains an invented nonzero contribution")


def validate_expert_handoffs(captures, plan):
    """Check complete actual CUDA/wire/native operand bytes for both ranks."""
    import numpy as np

    manifests = {owner: json.loads(Path(path).read_bytes()) for owner, path in captures.items()}
    if (
        manifests["npu-rank0"]["expert_provenance"]["hot_ids_by_layer"]
        != manifests["npu-rank1"]["expert_provenance"]["hot_ids_by_layer"]
    ):
        raise ValueError("native rank hot placements disagree")
    tables = {owner: {row["key"]: row for row in body["tensors"]} for owner, body in manifests.items()}
    for owner in ("npu-rank0", "npu-rank1"):
        for key, record in tables["cuda"].items():
            if "/expert/" not in key:
                continue
            peer = tables[owner][key]
            if any(record[field] != peer[field] for field in ("shape", "dtype", "bytes")):
                raise ValueError("actual expert CUDA/native handoff geometry changed")
            with (
                (Path(captures["cuda"]).parent / record["path"]).open("rb") as left,
                (Path(captures[owner]).parent / peer["path"]).open("rb") as right,
            ):
                while True:
                    a, b = left.read(DEFAULT_CHUNK_BYTES), right.read(DEFAULT_CHUNK_BYTES)
                    if a != b:
                        raise ValueError("actual expert CUDA/native operand bytes differ")
                    if not a:
                        break
        for phase in ("prefill", "decode"):
            for layer in sorted({item["layer"] for item in plan["expert_physical_schedule"]}):
                prefix = f"{phase}/expert/layer{layer}"
                wire = tables["cuda"][f"{prefix}/ids"]
                route = tables["cuda"][f"{phase}/selection/layer{layer}/route"]
                weights = tables["cuda"][f"{prefix}/weights"]
                original = tables["cuda"][f"{phase}/selection/layer{layer}/route_weights"]
                root = Path(captures["cuda"]).parent
                with (
                    (root / wire["path"]).open("rb") as ids_file,
                    (root / route["path"]).open("rb") as route_file,
                    (root / weights["path"]).open("rb") as weights_file,
                    (root / original["path"]).open("rb") as original_file,
                ):
                    rows = max(1, DEFAULT_CHUNK_BYTES // (8 * 8))
                    while True:
                        ids = np.frombuffer(ids_file.read(rows * 8 * 4), dtype="int32")
                        routes = np.frombuffer(route_file.read(rows * 8 * 8), dtype="int64")
                        masked = np.frombuffer(weights_file.read(rows * 8 * 4), dtype="float32")
                        values = np.frombuffer(original_file.read(rows * 8 * 4), dtype="float32")
                        if not len(ids):
                            if len(routes) or len(masked) or len(values):
                                raise ValueError("expert handoff routing metadata truncated")
                            break
                        hot = manifests[owner]["expert_provenance"]["hot_ids_by_layer"][str(layer)]
                        if (
                            not np.array_equal(ids, routes)
                            or np.where(np.isin(ids, hot), 0, values).tobytes() != masked.tobytes()
                        ):
                            raise ValueError("expert wire operands differ from actual CUDA route/weight selection")


class ExpertCudaObserver(OwnerDiagnostic):
    """Observe original layer-major windows without changing their execution."""

    def __init__(self, config_path, layers, synchronize):
        super().__init__(config_path, "cuda", layers, synchronize, allow_expert=True)
        self.window, self.selector_rows, self.vocabulary_cursor = None, {}, 0

    def _begin(self, command):
        super()._begin(command)
        self.window, self.selector_rows, self.vocabulary_cursor = None, {}, 0

    def begin_window(self, position, rows):
        if self.writer is None or self.window is not None or self.capture_key is not None or type(rows) is not int:
            raise ValueError("expert diagnostic window lifetime is invalid; concurrent windows unsupported")
        if self.cursor >= len(self.plan["schedule"]) or position != self.plan["schedule"][self.cursor][0]:
            raise ValueError("expert logical window is stale/reordered")
        phase = "prefill" if position < self.plan["prompt_tokens"] else "decode"
        width = (
            self.spec.get("prefill_window_size", self.spec.get("prefill_chunk_size", 32)) if phase == "prefill" else 1
        )
        expected_rows = min(width, self.plan["prompt_tokens"] - position) if phase == "prefill" else 1
        if rows != expected_rows or self.phase != phase:
            raise ValueError("expert physical window changed frozen batching/phase")
        self.window, self.chunk = (position, rows), (position, rows)
        self.selector_rows, self.chunk_graphs, self.observed = {}, set(), set()

    begin_chunk = begin_window

    def _remember(self, key, values):
        if self.capture_key is not None:
            super()._remember(key, values)
        elif self.writer is not None:
            if self.window is None:
                raise ValueError("actual expert selector has no active window")
            self.synchronize()
            self._write(key, values)

    def selection(self, layer, selected, *, rows=None):
        if layer.index not in self.layers or layer.linear:
            raise ValueError("expert indexer belongs to a different CUDA owner/type")
        if rows is None:
            rows = selected.shape[0] if selected is not None else None
        if type(rows) is not int or rows < 1:
            raise ValueError("dense indexer observation requires the actual row count")
        self._remember(("indexer", layer.index), (selected, rows))

    def suffix_state(self, state):
        if len(state) != 4 or any(value is None for value in state):
            raise ValueError("actual expert CUDA final pre-contraction state is incomplete")
        # No mHC state crosses the vendor boundary in the expert architecture.

    def _write(self, key, values):
        kind, layer = key
        if self.window is None or kind not in ("route", "indexer"):
            raise ValueError("unexpected actual expert CUDA observation")
        rows = values[0].shape[0] if kind == "route" else values[1]
        consumed = self.selector_rows.get(key, 0)
        if consumed + rows > self.window[1] or rows > self.spec.get("prefill_chunk_size", 32):
            raise ValueError("expert CUDA selector exceeded original logical chunk/window rows")
        offset = self.window[0] + consumed - (self.plan["prompt_tokens"] if self.phase == "decode" else 0)
        prefix = f"{self.phase}/selection/layer{layer}"
        if kind == "route":
            capture_route(self.writer, prefix, *values, offset=offset)
        else:
            width = self.plan["expected"]["cuda"][f"{prefix}/indexer"]["shape"][1]
            capture_selection(self.writer, f"{prefix}/indexer", values[0], offset=offset, rows=rows, width=width)
        self.selector_rows[key] = consumed + rows

    def vocabulary(self, logits, *, position=None, rows=None):
        if self.window is None or self.capture_key is not None:
            raise ValueError("expert vocabulary has no real completed logical chunk")
        if position is None:
            if self.window[1] > self.spec.get("prefill_chunk_size", 32):
                raise ValueError("batched prefill must observe every actual chunk vocabulary vector")
            position, rows = self.window
        if [position, rows] != self.plan["schedule"][self.vocabulary_cursor]:
            raise ValueError("expert vocabulary chunk omitted/reordered")
        self.synchronize()
        if self.phase == "prefill":
            capture_vocabulary(
                self.writer,
                logits,
                step=0 if position + rows == self.plan["prompt_tokens"] else None,
                prompt_position=position + rows - 1,
                prompt_chunk=self.vocabulary_cursor,
            )
        else:
            capture_vocabulary(self.writer, logits, step=position - self.plan["prompt_tokens"] + 1)
        self.vocabulary_cursor += 1

    def end_window(self):
        if self.window is None:
            raise ValueError("no active expert CUDA window")
        needed = {
            (field, layer.index)
            for layer in self.layers.values()
            for field in ((["route"] if layer.index >= 3 else []) + (["indexer"] if not layer.linear else []))
        }
        if set(self.selector_rows) != needed or any(rows != self.window[1] for rows in self.selector_rows.values()):
            raise ValueError("expert window omitted actual per-layer selector rows")
        finish = self.window[0] + self.window[1]
        while self.cursor < len(self.plan["schedule"]) and self.plan["schedule"][self.cursor][0] < finish:
            capture_chunk(self.writer, *self.plan["schedule"][self.cursor], iteration=self.cursor)
            self.cursor += 1
        if self.vocabulary_cursor != self.cursor:
            raise ValueError("expert window omitted produced full vocabulary vectors")
        if (
            self.phase == "decode"
            and self.spec.get("cuda_decode_graphs", False)
            and (not self.graphs or self.chunk_graphs != set(self.graphs))
        ):
            raise ValueError("expert configured CUDA graph path silently fell back to eager")
        self.window, self.chunk = None, None

    end_chunk = end_window


class ExpertOwner(OwnerDiagnostic):
    """Stateless native banks: real physical invocations, not attention caches."""

    def __init__(self, config_path, rank, hot_ids_by_layer, synchronize):
        if type(rank) is not int or rank not in (0, 1):
            raise ValueError("native expert diagnostics require rank0 or rank1")
        super().__init__(config_path, f"npu-rank{rank}", {}, synchronize, allow_expert=True)
        self.rank = rank
        self.hot = {str(layer): sorted(ids) for layer, ids in hot_ids_by_layer.items()}
        if any(
            len(values) != len(set(values)) or any(type(value) is not int or not 0 <= value < 288 for value in values)
            for values in self.hot.values()
        ):
            raise ValueError("actual native hot placement requires unique valid expert IDs")
        self.pending_call, self.call_cursor, self.next_sequence, self.receipts = None, 0, None, []

    def _begin(self, command):
        super()._begin(command)
        if set(self.hot) != {str(item["layer"]) for item in self.plan["expert_physical_schedule"]}:
            raise ValueError("actual native hot placement does not cover the planned banks")
        self.pending_call, self.call_cursor, self.next_sequence, self.receipts = None, 0, None, []

    def _call(self, call):
        if (
            self.writer is None
            or self.pending_call is not None
            or self.call_cursor >= len(self.plan["expert_physical_schedule"])
        ):
            raise ValueError("missing/overlapping native physical invocation")
        declared = self.plan["expert_physical_schedule"][self.call_cursor]
        if (
            any(call.get(key) != value for key, value in declared.items())
            or set(call) != set(declared) | {"rank", "sequence", "active", "index"}
            or any(type(call.get(key)) is not int for key in ("rank", "index", "sequence", "position", "rows", "layer"))
            or call.get("rank") != self.rank
            or call.get("index") != self.call_cursor
            or type(call.get("active")) is not bool
            or type(call.get("sequence")) is not int
            or call["sequence"] < 0
            or call["phase"] != self.phase
            or self.next_sequence is not None
            and call["sequence"] != self.next_sequence
        ):
            raise ValueError("native physical layer/position/rows/rank/sequence differs")
        return call

    def _record(self, call, inputs, output, count, execution, *, skip_inputs=None):
        import numpy as np

        offset = call["position"] - (self.plan["prompt_tokens"] if call["phase"] == "decode" else 0)
        spans = {}
        for field in ("input", "ids", "weights", "output"):
            key = f"{call['phase']}/expert/layer{call['layer']}/{field}"
            declared = self.plan["expected"][self.owner][key]
            value = output if field == "output" else inputs[field]
            if value is None:
                width = declared["shape"][1]
                chunk_rows = max(1, self.writer.max_chunk_bytes // (width * 4))
                for start in range(0, call["rows"], chunk_rows):
                    self.writer.append(
                        key,
                        np.zeros((min(chunk_rows, call["rows"] - start), width), dtype="float32"),
                        "floating_state",
                        offset=offset + start,
                    )
            else:
                if (
                    list(value.shape) != [call["rows"], declared["shape"][1]]
                    or str(value.dtype).removeprefix("torch.") != declared["dtype"]
                ):
                    raise ValueError("actual native input/output geometry or dtype changed")
                self.writer.append(key, value, declared["kind"], offset=offset)
            record = self.writer.records[key]
            row_bytes = math.prod(record["shape"][1:]) * DTYPE_BYTES[record["dtype"]]
            spans[field] = dict(
                key=key,
                offset=offset,
                rows=call["rows"],
                sha256=_span_hash(self.writer.directory / record["path"], offset * row_bytes, call["rows"] * row_bytes),
            )
        receipt = dict(
            call=call,
            status="executed" if call["active"] else "verified_zero_skip",
            execution=execution,
            positive_owned_routes=count,
            tensor_spans=spans,
        )
        if skip_inputs is not None:
            receipt["skip_inputs"] = skip_inputs
        self.receipts.append(receipt)
        self.next_sequence = call["sequence"] + int(call["active"])
        self.call_cursor += 1

    def observe_compute(self, sequence, layer, x, ids, weights, partial, *, execution):
        if self.pending_call is None:
            raise ValueError("actual native calculation lacks authenticated physical metadata")
        call = self.pending_call
        if (sequence, layer, x.shape[0]) != (call["sequence"], call["layer"], call["rows"]):
            raise ValueError("actual native COMMAND/upload differs from prepared call")
        self.synchronize()
        count = _route_count(_array(ids), _array(weights), self.rank, self.hot[str(layer)])
        if count < 1 or execution != _execution(call["rows"], count, self.plan["expert_execution"]):
            raise ValueError("actual native computation/replay path does not match qualified configuration")
        self._record(call, dict(input=x, ids=ids, weights=weights), partial, count, execution)
        self.pending_call = None

    def finish_phase(self, phase):
        if self.writer is None or self.plan is None:
            raise ValueError("native phase has no authenticated fixture")
        expected = (
            sum(item["phase"] == "prefill" for item in self.plan["expert_physical_schedule"])
            if phase == "prefill"
            else len(self.plan["expert_physical_schedule"])
        )
        if self.writer is None or self.pending_call is not None or phase != self.phase or self.call_cursor != expected:
            raise ValueError("native phase has missing/unfinished physical calls/skips")
        self._verify_active()
        self.synchronize()
        while self.cursor < len(self.plan["schedule"]):
            position, rows = self.plan["schedule"][self.cursor]
            if (position < self.plan["prompt_tokens"]) != (phase == "prefill"):
                break
            capture_chunk(self.writer, position, rows, iteration=self.cursor)
            self.cursor += 1
        self.completed_phases.add(phase)
        self.phase = "decode" if phase == "prefill" else "finished"

    def close(self):
        if self.writer is None or self.pending_call is not None or self.completed_phases != {"prefill", "decode"}:
            raise ValueError("incomplete native expert provenance")
        self._verify_active()
        path = self.writer.close()
        body = json.loads(path.read_bytes())
        body["expert_provenance"] = dict(
            schema_version=1,
            complete=True,
            normalization=NORMALIZATION,
            nonce=self.nonce,
            plan_sha256=self.plan_sha256,
            config_sha256=config_identity(self.spec),
            prompt_tokens=self.plan["prompt_tokens"],
            hot_ids_by_layer=self.hot,
            calls=self.receipts,
        )
        private_json(path, body)
        validate_capture(
            path,
            self.plan["expected"][self.owner],
            {},
            expert_schedule=self.plan["expert_physical_schedule"],
            expert_execution=self.plan["expert_execution"],
        )
        _handoff_private_files(path.parent, body)
        self.writer = None
        return path

    def handle_command(self, command):
        action = command.get("action")
        if action not in ("call", "skip"):
            return super().handle_command(command)
        encode_command(command)
        if command.get("nonce") != self.nonce or command.get("plan_sha256") != self.plan_sha256:
            raise ValueError("stale native expert command nonce/plan")
        call = self._call(command["call"])
        if action == "call":
            if not call["active"]:
                raise ValueError("prepared native calculation is marked skipped")
            self.pending_call = call
            status = "prepared"
        else:
            if call["active"]:
                raise ValueError("active native calculation cannot be replaced by a skip")
            width = self.plan["expected"][self.owner][f"{call['phase']}/expert/layer{call['layer']}/input"]["shape"][1]
            inputs = _skip_inputs(command["inputs"], self.root, call, self.nonce, width)
            count = _route_count(inputs["ids"], inputs["weights"], self.rank, self.hot[str(call["layer"])])
            if count:
                raise ValueError("CUDA declared a skip with positive native-owned routes")
            self._record(call, inputs, None, 0, "skipped", skip_inputs=command["inputs"])
            status = "verified_zero_skip"
        return dict(
            action=action,
            nonce=self.nonce,
            plan_sha256=self.plan_sha256,
            owner=self.owner,
            passed=True,
            call_index=call["index"],
            status=status,
        )


class ExpertCoordinatorDiagnostic(CoordinatorDiagnostic):
    def __init__(self, config_path, layers, owner_command, synchronize):
        observer = ExpertCudaObserver(config_path, layers, synchronize)
        super().__init__(config_path, layers, owner_command, synchronize, observer=observer)
        self.dispatch_cursor = 0

    def begin_fixture(self, plan, directory):
        super().begin_fixture(plan, directory)
        self.dispatch_cursor = 0

    def dispatch_call(self, layer, x, wire_ids, cold_weights, active, sequences):
        if self.command is None or self.observer.window is None:
            raise ValueError("native dispatch lacks an active authenticated expert window")
        plan = self.observer.plan
        declared = plan["expert_physical_schedule"][self.dispatch_cursor]
        position, rows = self.observer.window
        if (declared["layer"], declared["position"], declared["rows"]) != (layer, position, rows) or x.shape[0] != rows:
            raise ValueError("actual native dispatch batching changed physical expert schedule")
        if len(active) != 2 or any(type(flag) is not bool for flag in active) or len(sequences) != 2:
            raise ValueError("both actual rank admission flags/sequences required")
        self.observer.synchronize()
        offset = position - (plan["prompt_tokens"] if declared["phase"] == "decode" else 0)
        for field, value, kind in (
            ("input", x, "floating_state"),
            ("ids", wire_ids, "selectors"),
            ("weights", cold_weights, "floating_state"),
        ):
            self.observer.writer.append(f"{declared['phase']}/expert/layer{layer}/{field}", value, kind, offset=offset)
        for rank in range(2):
            call = dict(
                **declared, rank=rank, sequence=sequences[rank], active=active[rank], index=self.dispatch_cursor
            )
            command = {key: self.command[key] for key in ("nonce", "plan_sha256")}
            command.update(action="call" if active[rank] else "skip", call=call)
            if not active[rank]:
                directory = self.directory / f"skip-rank{rank}-call{self.dispatch_cursor}"
                writer = DiagnosticWriter(
                    directory,
                    "cuda",
                    max_chunk_bytes=self.observer.spec["execution_diagnostics"].get(
                        "max_chunk_bytes", DEFAULT_CHUNK_BYTES
                    ),
                )
                writer.append("input", x, "floating_state")
                writer.append("ids", wire_ids, "selectors")
                writer.append("weights", cold_weights, "floating_state")
                path = writer.close()
                body = json.loads(path.read_bytes())
                body.update(source_owner="cuda", call=call, nonce=self.command["nonce"])
                private_json(path, body)
                command["inputs"] = dict(path=str(path), sha256=sha256_file(path))
            reply = self._roundtrip(f"npu-rank{rank}", command)
            if reply.get("call_index") != self.dispatch_cursor or reply.get("status") != (
                "prepared" if active[rank] else "verified_zero_skip"
            ):
                raise ValueError("native owner did not authenticate this physical call/skip")
        self.dispatch_cursor += 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, required=True, help="Validate expert gate config without allocating devices"
    )
    args = parser.parse_args()
    result = validate_runtime(args.config, allow_expert=True)
    if result is not None and result[0].get("architecture") != EXPERT_ARCHITECTURE:
        raise ValueError("expert diagnostic config requires the expert architecture")


if __name__ == "__main__":
    main()
