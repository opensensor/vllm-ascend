# SPDX-License-Identifier: Apache-2.0
"""Gate-only owner observers and authenticated direct-DMA control adapters.

Shared hooks required in T11 (all conditional on diagnostics being enabled):
1. Construct OwnerDiagnostic before building graphs; attach it to each owned
   TextLayer. TextLayer.route calls observer.route(self, ids, weights); DSA
   dsa_attend calls observer.selection(self, selected), including None for its
   dense path. At layer44 finish_mlp, BEFORE final mhc_post/mean contraction,
   call observer.suffix_state((x, residual, post, comb)). cuda.suffix means this
   actual four-field pre-contraction state, never reconstructed head input.
2. Bracket ONLY actual graph capture with capture_begin(key)/capture_end(key).
   Startup warmups outside capture are ignored. No observation/CPU copy happens
   during capture. Immediately after graph.replay(), call replayed(key) before
   graph scratch can be reused; this fences and snapshots registered selectors.
   Every configured captured graph must replay for every decode chunk. Attach
   these hooks to manual replay paths (including forward_traced) as well.
3. For each real model chunk: begin_chunk(position, rows), boundary('prefix',
   state) and boundary('suffix', ...) only through suffix_state on CUDA; native
   owners use boundary('input', state), boundary('output', state). CUDA then
   calls vocabulary(actual_logits). end_chunk() observes eager refs and checks
   complete coverage. boundary() fences and snapshots immediately to avoid
   delayed references to overwritten handoff buffers. No cache address changes.
4. CoordinatorDiagnostic(config_path, layers, owner_command, synchronize)
   exposes the T19 adapter and .observer for CUDA hooks. owner_command(owner,
   command) must deliver a command and return that SAME owner's reply. Native
   owners dispatch commands through observer.handle_command(command), while
   idle between chunks. Begin/phase/end payloads contain a nonce and plan hash;
   the large plan is a private file with an authenticated path, never inline.

execution_diagnostics.directory_root is a pre-existing private absolute
directory, bound by normalized runtime authentication. All plans/data must stay
inside it. Socket diagnostics fail before allocation; normal socket startup
parity remains available with execution_diagnostics disabled. The expert path
uses expert_diagnostics.py's explicit per-call physical batch/skip receipts.
Callers must opt into that adapter instead of treating stateless expert banks
as complete recurrent layers.
"""

import argparse
import json
import os
import secrets
import stat
from pathlib import Path

from .execution_model_gate import (
    BLOCK_ARCHITECTURE,
    OWNERS,
    authenticate_config,
    canonical_sha256,
    config_identity,
    execution_path,
    make_plan,
    validate_capture,
)
from .stage_diagnostics import (
    DEFAULT_CHUNK_BYTES,
    DiagnosticWriter,
    capture_boundary,
    capture_chunk,
    capture_layers,
    capture_route,
    capture_selection,
    capture_vocabulary,
    private_json,
    sha256_file,
)

MAX_CONTROL_BYTES = 16 << 10


def _inside(root, path, *, existing=True):
    target = Path(path)
    if not target.is_absolute():
        raise ValueError("diagnostic paths must be absolute")
    resolved = target.resolve(strict=existing)
    if resolved == root or not resolved.is_relative_to(root):
        raise ValueError("diagnostic path escapes its authenticated private root")
    return resolved


def validate_runtime(config_path, *, allow_expert=False):
    """Call before allocating devices; disabled diagnostics have no side effects."""
    config_path = Path(config_path)
    spec = json.loads(config_path.read_bytes())
    if spec.get("execution_diagnostics", {}).get("enabled", False) is False:
        return None
    authenticate_config(spec, config_path.parent)
    expert = spec.get("architecture", BLOCK_ARCHITECTURE) == "cuda-backbone-ascend-experts"
    if expert and not allow_expert or not expert and spec.get("architecture", BLOCK_ARCHITECTURE) != BLOCK_ARCHITECTURE:
        raise ValueError("expert diagnostics require authenticated physical batch/skip normalization")
    if spec.get("expert_transport" if expert else "stage_transport", "socket") != "direct-dma":
        raise ValueError("active diagnostics require direct-DMA owner control")
    directory = Path(spec["execution_diagnostics"]["directory_root"])
    if not directory.is_absolute() or not directory.is_dir():
        raise ValueError("diagnostic directory_root must be an existing absolute private directory")
    if directory.stat().st_mode & 0o077:
        raise ValueError("diagnostic directory_root must exclude group/other access")
    return spec, directory.resolve()


def encode_command(command):
    body = json.dumps(command, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    if len(body) > MAX_CONTROL_BYTES:
        raise ValueError("diagnostic control exceeds 16KiB; send an authenticated private plan path")
    return body


def _storage_range(value):
    if value is None or type(value) is int:
        # Expert graph callbacks also retain immutable row-count metadata.
        return None
    if hasattr(value, "data_ptr"):
        return value.data_ptr(), value.numel() * value.element_size()
    if hasattr(value, "ctypes"):
        return value.ctypes.data, value.nbytes
    raise ValueError("diagnostic references need concrete owner tensor storage")


def _overlap(left, right):
    return (
        left is not None
        and right is not None
        and left[1]
        and right[1]
        and (left[0] < right[0] + right[1] and right[0] < left[0] + left[1])
    )


def _handoff_private_files(directory, manifest):
    """Let a user coordinator read artifacts finalized by a sudo worker.

    The owner directory stays inaccessible to that user until every file has
    closed and changed ownership. Only newly created authenticated regular
    files are touched; symlinks/hardlinks and unexpected directory entries fail.
    """
    parent_owner = directory.parent.stat()
    names = {record["path"] for record in manifest["tensors"]} | {"manifest.json"}
    if set(path.name for path in directory.iterdir()) != names:
        raise ValueError("unexpected files in finalized private diagnostic owner directory")
    for name in names:
        path = directory / name
        info = path.lstat()
        if path.parent != directory or not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise ValueError("diagnostic ownership handoff rejects symlinks/hardlinks")
        if (info.st_uid, info.st_gid) != (parent_owner.st_uid, parent_owner.st_gid):
            os.chown(path, parent_owner.st_uid, parent_owner.st_gid, follow_symlinks=False)
    owner = directory.stat()
    if (owner.st_uid, owner.st_gid) != (parent_owner.st_uid, parent_owner.st_gid):
        os.chown(directory, parent_owner.st_uid, parent_owner.st_gid, follow_symlinks=False)


class OwnerDiagnostic:
    """One owner, fixed graph refs, one authenticated fixture/chunk at a time."""

    def __init__(self, config_path, owner, layers, synchronize, *, allow_expert=False):
        result = validate_runtime(config_path, allow_expert=allow_expert)
        if result is None or owner not in OWNERS or not callable(synchronize):
            raise ValueError("enabled diagnostics, expected owner and owner-local fence required")
        self.config_path, self.owner = Path(config_path), owner
        self.spec, self.root = result
        self.layers = dict(layers) if hasattr(layers, "items") else {layer.index: layer for layer in layers}
        if any(index != layer.index for index, layer in self.layers.items()):
            raise ValueError("layer mapping changed owner indices")
        self.synchronize = synchronize
        self.graphs, self.capture_key = {}, None
        self.writer, self.plan, self.nonce, self.plan_sha256 = None, None, None, None
        self.plan_path, self.plan_file_sha256 = None, None
        self.seen_nonces = set()
        self.chunk, self.cursor, self.phase, self.completed_phases = None, 0, "prefill", set()
        self.pending, self.observed, self.chunk_graphs = {}, set(), set()
        self.replay_counts, self.geometry = {}, {}

    def capture_begin(self, key):
        if (
            self.writer is not None
            or self.capture_key is not None
            or not isinstance(key, str)
            or not key
            or key in self.graphs
        ):
            raise ValueError("graph refs must be registered once before any fixture")
        self.graphs[key] = {}
        self.capture_key = key

    def capture_end(self, key):
        if self.capture_key != key:
            raise ValueError("graph capture registration ended out of order")
        self.capture_key = None

    def discard_capture(self, key):
        """Drop only a drained/rejected startup capture, never active gate refs."""
        if self.writer is not None or self.capture_key is not None:
            raise ValueError("cannot discard capture references during a fixture or active capture")
        self.graphs.pop(key, None)

    def _remember(self, key, values):
        if self.capture_key is not None:
            target = self.graphs[self.capture_key]
            if key in target:
                raise ValueError("duplicate selector registration inside one graph")
            ranges = [_storage_range(value) for value in values]
            for other in target.values():
                if any(_overlap(left, _storage_range(right)) for left in ranges for right in other):
                    raise ValueError("captured outputs alias another selector/state; observe before reuse")
            target[key] = tuple(values)
        elif self.writer is not None:
            if self.chunk is None or key in self.pending or key in self.observed:
                raise ValueError("selector observed outside a chunk or more than once")
            self.pending[key] = tuple(values)

    def route(self, layer, ids, weights):
        if layer.index not in self.layers:
            raise ValueError("route callback belongs to an unexpected layer owner")
        self._remember(("route", layer.index), (ids, weights))

    def selection(self, layer, selected, *, rows=None):
        if layer.index not in self.layers or layer.linear:
            raise ValueError("indexer callback belongs to an unexpected layer owner/type")
        if rows is not None and self.chunk is not None and rows != self.chunk[1]:
            raise ValueError("indexer callback changed the actual chunk row count")
        self._remember(("indexer", layer.index), (selected,))

    def suffix_state(self, state):
        if self.owner != "cuda" or 44 not in self.layers or len(state) != 4 or any(value is None for value in state):
            raise ValueError("suffix requires actual layer44 complete pre-contraction state")
        self._remember(("boundary", "suffix"), state)

    def _write(self, key, values):
        if key in self.observed:
            raise ValueError("selector or boundary observed twice for one logical chunk")
        kind, index = key
        position, rows = self.chunk
        offset = position if self.phase == "prefill" else position - self.plan["prompt_tokens"]
        if kind == "boundary":
            capture_boundary(self.writer, f"{self.phase}/boundary/{index}", values, offset=offset)
        else:
            prefix = f"{self.phase}/selection/layer{index}"
            if kind == "route":
                capture_route(self.writer, prefix, *values, offset=offset)
            elif kind == "indexer":
                width = self.plan["expected"][self.owner][f"{prefix}/indexer"]["shape"][1]
                capture_selection(self.writer, f"{prefix}/indexer", values[0], offset=offset, rows=rows, width=width)
            else:
                raise ValueError("unknown diagnostic callback kind")
        self.observed.add(key)

    def replayed(self, key):
        if self.writer is None:
            return
        if self.capture_key is not None or self.chunk is None or key not in self.graphs or key in self.chunk_graphs:
            raise ValueError("unregistered, duplicated or delayed graph replay observation")
        if self.chunk[1] != 1:
            raise ValueError("registered decode graph replay changed row geometry")
        self.synchronize()
        for name, values in self.graphs[key].items():
            self._write(name, values)
        self.chunk_graphs.add(key)
        self.replay_counts[key] = self.replay_counts.get(key, 0) + 1

    def begin_chunk(self, position, rows):
        if self.writer is None or self.capture_key is not None or self.chunk is not None:
            raise ValueError("diagnostic fixture/chunk lifetime is invalid")
        if type(position) is not int or type(rows) is not int or rows < 1:
            raise ValueError("logical chunk position/rows must retain integer geometry")
        if self.cursor >= len(self.plan["schedule"]) or [position, rows] != self.plan["schedule"][self.cursor]:
            raise ValueError("stale/reordered logical chunk or changed chunk geometry")
        expected_phase = "prefill" if position < self.plan["prompt_tokens"] else "decode"
        if self.phase != expected_phase or self.phase in self.completed_phases:
            raise ValueError("chunk crossed an unfinished diagnostic phase boundary")
        self.chunk, self.pending, self.observed, self.chunk_graphs = (position, rows), {}, set(), set()

    def boundary(self, name, state):
        if self.chunk is None or self.capture_key is not None:
            raise ValueError("boundary observation needs a real chunk outside graph capture")
        allowed = ("prefix",) if self.owner == "cuda" else ("input", "output")
        if name not in allowed:
            raise ValueError("unexpected handoff boundary; suffix must use layer44 suffix_state")
        self.synchronize()
        self._write(("boundary", name), state)

    def vocabulary(self, logits):
        if (
            self.owner != "cuda"
            or self.chunk is None
            or self.capture_key is not None
            or ("vocabulary", 0) in self.observed
        ):
            raise ValueError("vocabulary observation is missing its CUDA chunk or was repeated")
        self.synchronize()
        position, rows = self.chunk
        if self.phase == "prefill":
            final = position + rows == self.plan["prompt_tokens"]
            capture_vocabulary(
                self.writer,
                logits,
                step=0 if final else None,
                prompt_position=position + rows - 1,
                prompt_chunk=self.cursor,
            )
        else:
            capture_vocabulary(self.writer, logits, step=position - self.plan["prompt_tokens"] + 1)
        self.observed.add(("vocabulary", 0))

    def end_chunk(self):
        if self.chunk is None or self.capture_key is not None:
            raise ValueError("no completed diagnostic chunk")
        self.synchronize()
        for key, values in self.pending.items():
            self._write(key, values)
        needed = set()
        for key in self.plan["expected"][self.owner]:
            if key.startswith(f"{self.phase}/selection/"):
                _, _, layer, field = key.split("/")
                if field != "route_weights":
                    needed.add(("route" if field == "route" else "indexer", int(layer.removeprefix("layer"))))
            elif key.startswith(f"{self.phase}/boundary/"):
                needed.add(("boundary", key.split("/")[2]))
        if self.owner == "cuda":
            needed.add(("vocabulary", 0))
        if self.observed != needed:
            raise ValueError("missing or unknown actual selector/deferred/vocabulary coverage")
        graph_enabled = self.spec.get("cuda_decode_graphs" if self.owner == "cuda" else "npu_decode_graphs", False)
        if self.phase == "decode" and graph_enabled and (not self.graphs or self.chunk_graphs != set(self.graphs)):
            raise ValueError("configured graph path was replaced by eager or missing replay")
        capture_chunk(self.writer, *self.chunk, iteration=self.cursor)
        self.cursor += 1
        self.chunk, self.pending = None, {}

    def finish_phase(self, phase):
        if self.writer is None or self.chunk is not None or phase != self.phase or phase in self.completed_phases:
            raise ValueError("stale, repeated or unfinished diagnostic phase")
        prefill_chunks = sum(position < self.plan["prompt_tokens"] for position, _ in self.plan["schedule"])
        expected_cursor = prefill_chunks if phase == "prefill" else len(self.plan["schedule"])
        if self.cursor != expected_cursor:
            raise ValueError("phase completed before all logical chunks")
        self._verify_active()
        self.synchronize()
        geometry = capture_layers(self.writer, f"{phase}/state", self.layers)
        if geometry != self.plan["allocation_geometry"][self.owner]:
            raise ValueError("actual owner cache reservation geometry changed")
        self.geometry = geometry
        self.completed_phases.add(phase)
        self.phase = "decode" if phase == "prefill" else "finished"

    def close(self):
        if self.writer is None or self.chunk is not None or self.completed_phases != {"prefill", "decode"}:
            raise ValueError("cannot qualify incomplete diagnostic owner state")
        self._verify_active()
        path = self.writer.close(allocation_geometry=self.geometry)
        body, _ = validate_capture(path, self.plan["expected"][self.owner], self.geometry)
        body["runtime_receipt"] = dict(
            nonce=self.nonce,
            plan_sha256=self.plan_sha256,
            execution_path_sha256=self.plan["execution_path_sha256"],
            replay_counts=self.replay_counts,
            chunks=self.cursor,
        )
        private_json(path, body)
        _handoff_private_files(path.parent, body)
        self.writer = None
        return path

    def _verify_config(self):
        current = validate_runtime(
            self.config_path, allow_expert=self.spec.get("architecture") == "cuda-backbone-ascend-experts"
        )
        if current is None or canonical_sha256(current[0]) != canonical_sha256(self.spec):
            raise ValueError("diagnostic runtime configuration drifted after owner construction")

    def _verify_active(self):
        self._verify_config()
        if sha256_file(self.plan_path) != self.plan_file_sha256:
            raise ValueError("authenticated private plan drifted during diagnostic observation")

    def _begin(self, command):
        if self.writer is not None or self.capture_key is not None:
            raise ValueError("owner already has an active diagnostic fixture/capture")
        nonce = command["nonce"]
        if (
            not isinstance(nonce, str)
            or len(nonce) != 32
            or any(c not in "0123456789abcdef" for c in nonce)
            or nonce in self.seen_nonces
        ):
            raise ValueError("fresh 128-bit diagnostic nonce required")
        self._verify_config()
        plan_path = _inside(self.root, command["plan"]["path"])
        if sha256_file(plan_path) != command["plan"]["sha256"]:
            raise ValueError("private plan file drifted")
        wrapper = json.loads(plan_path.read_bytes())
        plan = wrapper["plan"]
        if wrapper["config_sha256"] != config_identity(self.spec) or canonical_sha256(plan) != command["plan_sha256"]:
            raise ValueError("diagnostic plan/configuration authentication differs")
        directory = _inside(self.root, command["directory"])
        if directory != plan_path.parent or wrapper["directory"] != str(directory) or directory.stat().st_mode & 0o077:
            raise ValueError("plan/data directory must be the same private authenticated fixture directory")
        model = json.loads((Path(self.spec["checkpoint"]) / "config.json").read_bytes())["text_config"]
        model.setdefault("vocab_size", 154880)
        if (
            type(plan.get("prompt_tokens")) is not int
            or type(plan.get("output_tokens")) is not int
            or plan["prompt_tokens"] < 1
            or plan["output_tokens"] < 2
            or plan["prompt_tokens"] + plan["output_tokens"] > self.spec["max_context"]
        ):
            raise ValueError("diagnostic plan exceeds authenticated context geometry")
        expected = make_plan(
            plan["name"], [0] * plan["prompt_tokens"], model, self.spec, output_tokens=plan["output_tokens"]
        )
        expected["prompt_ids_sha256"] = plan["prompt_ids_sha256"]
        if plan != expected or set(map(str, self.layers)) != set(plan["allocation_geometry"][self.owner]):
            raise ValueError("plan changed expected complete-state or layer ownership coverage")
        self.plan, self.plan_sha256, self.nonce = plan, command["plan_sha256"], nonce
        self.plan_path, self.plan_file_sha256 = plan_path, command["plan"]["sha256"]
        self.seen_nonces.add(nonce)
        self.writer = DiagnosticWriter(
            directory / self.owner,
            self.owner,
            max_chunk_bytes=self.spec["execution_diagnostics"].get("max_chunk_bytes", DEFAULT_CHUNK_BYTES),
        )
        self.cursor, self.phase, self.completed_phases, self.replay_counts = 0, "prefill", set(), {}

    def handle_command(self, command):
        encode_command(command)
        action = command.get("action")
        if action == "begin":
            self._begin(command)
        else:
            if (
                self.writer is None
                or command.get("nonce") != self.nonce
                or command.get("plan_sha256") != self.plan_sha256
            ):
                raise ValueError("stale diagnostic command nonce/plan or no active owner")
            if action == "phase":
                self.finish_phase(command["phase"])
            elif action == "end":
                path = self.close()
            else:
                raise ValueError("unknown diagnostic control action")
        reply = dict(action=action, nonce=self.nonce, plan_sha256=self.plan_sha256, owner=self.owner, passed=True)
        if action == "phase":
            reply["phase"] = command["phase"]
        if action == "end":
            reply["manifest"] = dict(path=str(path), sha256=sha256_file(path))
        return reply


class CoordinatorDiagnostic:
    def __init__(self, config_path, layers, owner_command, synchronize, *, observer=None):
        if not callable(owner_command):
            raise ValueError("native owner command roundtrip callback required")
        self.observer = observer if observer is not None else OwnerDiagnostic(config_path, "cuda", layers, synchronize)
        if self.observer.owner != "cuda" or self.observer.config_path.resolve() != Path(config_path).resolve():
            raise ValueError("diagnostic coordinator requires its own authenticated CUDA observer")
        self.owner_command = owner_command
        self.command, self.directory = None, None

    def capabilities(self):
        return dict(
            schema_version=1,
            owners=list(OWNERS),
            complete_state=True,
            complete_selectors=True,
            graph_replay_observed=True,
            execution_path_sha256=canonical_sha256(execution_path(self.observer.spec)),
        )

    def _roundtrip(self, owner, command):
        encode_command(command)
        reply = self.observer.handle_command(command) if owner == "cuda" else self.owner_command(owner, command)
        if (
            reply.get("passed") is not True
            or reply.get("owner") != owner
            or any(reply.get(key) != command.get(key) for key in ("action", "nonce", "plan_sha256"))
            or command["action"] == "phase"
            and reply.get("phase") != command["phase"]
        ):
            raise ValueError("diagnostic reply belongs to a stale command, phase or different owner")
        return reply

    def begin_fixture(self, plan, directory):
        if self.command is not None:
            raise ValueError("diagnostic coordinator fixture already active")
        directory = _inside(self.observer.root, directory)
        path = directory / "diagnostic-plan.json"
        if path.exists():
            raise ValueError("never overwrite a prior authenticated diagnostic plan")
        private_json(path, dict(plan=plan, directory=str(directory), config_sha256=config_identity(self.observer.spec)))
        self.command = dict(
            action="begin",
            nonce=secrets.token_hex(16),
            plan_sha256=canonical_sha256(plan),
            plan=dict(path=str(path), sha256=sha256_file(path)),
            directory=str(directory),
        )
        self.directory = directory
        for owner in OWNERS:
            self._roundtrip(owner, self.command)

    def finish_phase(self, phase):
        if self.command is None:
            raise ValueError("no active diagnostic fixture")
        command = {key: self.command[key] for key in ("nonce", "plan_sha256")}
        command.update(action="phase", phase=phase)
        for owner in OWNERS:
            self._roundtrip(owner, command)

    def end_fixture(self):
        if self.command is None:
            raise ValueError("no active diagnostic fixture")
        command = {key: self.command[key] for key in ("nonce", "plan_sha256")}
        command["action"] = "end"
        captures = {}
        for owner in OWNERS:
            reply = self._roundtrip(owner, command)
            path = _inside(self.directory, reply["manifest"]["path"])
            if path != self.directory / owner / "manifest.json" or sha256_file(path) != reply["manifest"]["sha256"]:
                raise ValueError("owner manifest path/content authentication failed")
            captures[owner] = path
        self.command = None
        return captures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Validate gate runtime without allocating devices")
    args = parser.parse_args()
    validate_runtime(args.config)


if __name__ == "__main__":
    main()
