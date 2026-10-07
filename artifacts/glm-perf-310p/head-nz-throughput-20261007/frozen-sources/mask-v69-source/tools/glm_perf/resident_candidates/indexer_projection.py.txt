# SPDX-License-Identifier: Apache-2.0
"""Experimental shared-input FP32 indexer projection, prepared before capture.

The closure-owned bank never replaces model weights or initializes from forward.
The resident harness removes both method replacements and clears their graphs
on rollback. Target and MTP indexers must both be prepared before recapture.
"""

import ast
import inspect
import textwrap
import weakref

import torch

MAX_EXTRA_WEIGHT_BYTES = 64 * 1024 * 1024


class Projection:
    __slots__ = ("key_weight", "gate_weight", "packed")

    def __init__(self, key_weight, gate_weight, packed):
        self.key_weight, self.gate_weight, self.packed = key_weight, gate_weight, packed

    def matches(self, indexer):
        return self.key_weight is indexer._wk_weight_f32 and self.gate_weight is indexer._gate_weight_f32


class ProjectionBank:
    def __init__(self, max_bytes=MAX_EXTRA_WEIGHT_BYTES):
        if max_bytes <= 0:
            raise ValueError("projection memory budget must be positive")
        self.max_bytes = max_bytes
        self.entries = weakref.WeakKeyDictionary()
        self.prepared_bytes = 0

    @staticmethod
    def weights(indexer):
        key, gate = indexer._wk_weight_f32, indexer._gate_weight_f32
        if (
            key is None
            or gate is None
            or key.dtype != torch.float32
            or gate.dtype != torch.float32
            or key.ndim != 2
            or gate.ndim != 2
            or key.shape[0] != indexer.head_dim + indexer.n_head
            or gate.shape != (indexer.head_dim, key.shape[1])
            or key.device != gate.device
            or not key.is_contiguous()
            or not gate.is_contiguous()
        ):
            raise ValueError("indexer requires prepared contiguous FP32 key/head and gate weights")
        return key, gate

    def prepare(self, indexers):
        unique = list(dict.fromkeys(indexers))
        if not unique:
            raise ValueError("no GLM indexers found; refusing an inactive projection experiment")
        weights = [(indexer, *self.weights(indexer)) for indexer in unique]
        required = sum((key.numel() + gate.numel()) * key.element_size() for _, key, gate in weights)
        if required > self.max_bytes:
            raise ValueError(f"packed indexers need {required} bytes, exceeding budget {self.max_bytes}")
        # Reuse an existing capture's operands. A changed model/weight set needs
        # a fresh resident transaction, so do not hold old and new copies at once.
        if self.entries:
            if len(self.entries) != len(unique) or any(
                indexer not in self.entries or not self.entries[indexer].matches(indexer) for indexer in unique
            ):
                raise ValueError("projection weights changed; restore baseline and prepare a fresh candidate")
            return
        pending = weakref.WeakKeyDictionary()
        for indexer, key, gate in weights:
            packed = torch.cat((key.detach(), gate.detach()), dim=0).contiguous()
            pending[indexer] = Projection(key, gate, packed)
        self.entries = pending
        self.prepared_bytes = required

    def project(self, indexer, hidden):
        entry = self.entries.get(indexer)
        if entry is None or not entry.matches(indexer):
            raise RuntimeError("indexer projection was not prepared for these weights before graph capture")
        combined = torch.mm(hidden, entry.packed.t())
        width = entry.key_weight.shape[0]
        return combined[:, :width], combined[:, width:]


def prepare_runner(bank, runner, indexer_type):
    wrappers = [runner.model]
    drafter = getattr(runner, "drafter", None)
    if drafter is not None:
        wrappers.append(drafter.model)
    indexers = []
    for wrapper in wrappers:
        # Match the qualified resident wrapper; no model search via globals/GC.
        model = getattr(wrapper, "runnable", wrapper)
        indexers.extend(module for module in model.modules() if isinstance(module, indexer_type))
    bank.prepare(indexers)


def rewrite_forward(source):
    tree = ast.parse(textwrap.dedent(source))
    function = tree.body[0]
    key_assignment = ast.parse("kw = torch.mm(hidden_f32, self._wk_weight_f32.t())").body[0]
    gate_assignment = ast.parse("gate_score = torch.mm(hidden_f32, gate_weight_f32.t())").body[0]
    key_positions = [i for i, node in enumerate(function.body) if ast.dump(node) == ast.dump(key_assignment)]
    gate_positions = [i for i, node in enumerate(function.body) if ast.dump(node) == ast.dump(gate_assignment)]
    if len(key_positions) != 1 or len(gate_positions) != 1 or key_positions[0] >= gate_positions[0]:
        raise ValueError("indexer projection source changed: re-audit before applying")
    function.body[key_positions[0]] = ast.parse("kw, gate_score = _candidate_projection(self, hidden_f32)").body[0]
    del function.body[gate_positions[0]]
    return ast.fix_missing_locations(tree)


def make_capture(original, bank, indexer_type):
    original = getattr(original, "__glm_resident_original__", original)

    def capture(runner, *args, **kwargs):
        prepare_runner(bank, runner, indexer_type)
        return original(runner, *args, **kwargs)

    capture.__glm_resident_original__ = original
    return capture


def replacements(native_resources=None):
    # Worker-only imports. No tensors are allocated while staging replacements.
    from vllm_ascend._310p.model_runner_310p import NPUModelRunner310
    from vllm_ascend.models.glm5next.attention import Indexer

    bank = ProjectionBank()
    original_forward = getattr(Indexer.forward, "__glm_resident_original__", Indexer.forward)
    scope = dict(original_forward.__globals__, _candidate_projection=bank.project)
    exec(
        compile(rewrite_forward(inspect.getsource(original_forward)), inspect.getfile(original_forward), "exec"), scope
    )
    candidate_forward = scope[original_forward.__name__]

    def forward(self, *args, **kwargs):
        # Guard before entering the original lazy-cast block. Never allow an
        # unprepared indexer to create persistent weight tensors during capture.
        entry = bank.entries.get(self)
        if entry is None or not entry.matches(self):
            raise RuntimeError("prepare target and draft indexer projections before forward")
        return candidate_forward(self, *args, **kwargs)

    forward.__glm_resident_original__ = original_forward
    return {
        "vllm_ascend.models.glm5next.attention:Indexer.forward": forward,
        "vllm_ascend._310p.model_runner_310p:NPUModelRunner310.capture_model": make_capture(
            NPUModelRunner310.capture_model, bank, Indexer
        ),
    }
