# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""W8A8 expert-tensor mapping + load-time rejection for Qwen4Exp on 310P (T3.1).

This module is the single authority for translating the exported ModelSlim
W8A8-DYNAMIC checkpoint's *per-expert* tensors into the fused-MoE weight layout
consumed by :class:`AscendW8A8DynamicFusedMoEMethod310`
(``vllm_ascend/_310p/quantization/methods/w8a8_dynamic.py``), and for rejecting
a checkpoint that does not match the frozen geometry/dtype contract *before* any
224 GB of tensors are streamed off disk.

Ground-truth checkpoint scheme (verified against real shard headers)
-------------------------------------------------------------------
Per (layer ``L``, expert ``E``) there are three projections, each with a
quantized weight plus per-output-channel scale and offset::

    model.language_model.layers.{L}.mlp.experts.{E}.gate_proj.weight        I8  [moe, hidden]
    model.language_model.layers.{L}.mlp.experts.{E}.up_proj.weight          I8  [moe, hidden]
    model.language_model.layers.{L}.mlp.experts.{E}.down_proj.weight        I8  [hidden, moe]
    ...{gate_proj|up_proj|down_proj}.weight_scale                           F32 [out, 1]
    ...{gate_proj|up_proj|down_proj}.weight_offset                          F32 [out, 1]

where ``out`` is the row (dim-0) size of the corresponding weight. With 48
layers x 512 experts x 3 projections this is **73,728** quantized weights plus
73,728 scales and 73,728 offsets.

Target fused-MoE layout (from :class:`AscendW8A8DynamicFusedMoEMethod310`)
-------------------------------------------------------------------------
``gate_proj`` and ``up_proj`` fuse column-wise into ``w13_*`` and ``down_proj``
lands in ``w2_*``::

    w13_weight        int8    [E, 2 * moe, hidden]   gate -> rows [0, moe), up -> rows [moe, 2*moe)
    w2_weight         int8    [E, hidden, moe]        down -> full slot
    w13_weight_scale  float32 [E, 2 * moe, 1]
    w13_weight_offset float32 [E, 2 * moe, 1]
    w2_weight_scale   float32 [E, hidden, 1]
    w2_weight_offset  float32 [E, hidden, 1]

Router / shared-expert / attention (QSA) / LM-head / embeddings / PLE / norms
stay non-quantized F16 and are **not** mapped here (they are validated against
the frozen dtype policy but pass through untouched).

Dtype policy
------------
Every tensor is checked against the frozen T1.2 policy
(:data:`vllm_ascend.models.qwen4_exp.dtype_policy.ASCEND_QWEN4EXP_DTYPE_POLICY`)
before it is mapped: expert weights are ``int8``, expert scale/offset are
``float32`` (the policy's accumulation dtype), and any *floating* non-expert
tensor must equal the policy main dtype (``float16``). Integer index tensors
(e.g. PLE n-gram vocab sizes) are tolerated.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import torch

from vllm_ascend.models.qwen4_exp.dtype_policy import (
    ASCEND_QWEN4EXP_DTYPE_POLICY,
    Qwen4ExpDtypePolicy,
)

__all__ = [
    "DuplicateTensorError",
    "ExpertTensorMapping",
    "ExtraTensorError",
    "MissingTensorError",
    "TensorDtypeError",
    "TensorShapeError",
    "WeightMapping",
    "WeightMappingError",
    "expected_expert_tensor_names",
    "geometry_from_manifest",
    "map_expert_tensor",
    "validate_expert_weight_map",
]

# --------------------------------------------------------------------------- #
# Constants (naming scheme + projection fan-in)
# --------------------------------------------------------------------------- #
GATE_PROJ = "gate_proj"
UP_PROJ = "up_proj"
DOWN_PROJ = "down_proj"
PROJECTIONS: tuple[str, ...] = (GATE_PROJ, UP_PROJ, DOWN_PROJ)
# gate_proj + up_proj fuse column-wise into the w13_* params.
W13_PROJECTIONS: frozenset[str] = frozenset({GATE_PROJ, UP_PROJ})

WEIGHT = "weight"
WEIGHT_SCALE = "weight_scale"
WEIGHT_OFFSET = "weight_offset"
KINDS: tuple[str, ...] = (WEIGHT, WEIGHT_SCALE, WEIGHT_OFFSET)

# model.language_model.layers.{L}.mlp.experts.{E}.{proj}.{kind}
_EXPERT_RE = re.compile(
    r"^model\.language_model\.layers\.(?P<layer>\d+)"
    r"\.mlp\.experts\.(?P<expert>\d+)"
    r"\.(?P<proj>gate_proj|up_proj|down_proj)"
    r"\.(?P<kind>weight|weight_scale|weight_offset)$"
)

# safetensors header dtype strings -> torch.dtype.
_ST_DTYPE_TO_TORCH: dict[str, torch.dtype] = {
    "I8": torch.int8,
    "U8": torch.uint8,
    "I16": torch.int16,
    "I32": torch.int32,
    "I64": torch.int64,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "F32": torch.float32,
    "F64": torch.float64,
    "BOOL": torch.bool,
}

_REQUIRED_GEOMETRY_KEYS = (
    "num_hidden_layers",
    "num_experts",
    "moe_intermediate_size",
    "hidden_size",
)

# Meta describes one tensor the way a safetensors header does.
TensorMeta = Mapping[str, object]
# The "provided" index may be a name->meta mapping (a safetensors index cannot
# repeat a key) or a sequence of (name, meta) pairs (which can, to express a
# duplicate source tensor).
ProvidedIndex = Mapping[str, TensorMeta] | Sequence[tuple[str, TensorMeta]]


# --------------------------------------------------------------------------- #
# Errors (one class per rejection category, all actionable)
# --------------------------------------------------------------------------- #
class WeightMappingError(ValueError):
    """Base class for every W8A8 load-time rejection."""

    category = "invalid"

    def __init__(self, message: str, tensors: Sequence[str] | None = None):
        super().__init__(message)
        self.tensors: list[str] = list(tensors or [])


class MissingTensorError(WeightMappingError):
    category = "missing"


class ExtraTensorError(WeightMappingError):
    category = "extra"


class DuplicateTensorError(WeightMappingError):
    category = "duplicate"


class TensorShapeError(WeightMappingError):
    category = "wrong_shape"


class TensorDtypeError(WeightMappingError):
    category = "wrong_dtype"


# --------------------------------------------------------------------------- #
# Mapping result types
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ExpertTensorMapping:
    """Placement of one source expert tensor into the fused-MoE layout."""

    source_name: str
    layer: int
    expert: int
    proj: str
    kind: str
    target_param: str
    expert_index: int
    row_start: int
    row_stop: int
    expected_dtype: torch.dtype
    expected_shape: tuple[int, ...]


@dataclass(frozen=True)
class WeightMapping:
    """Validated mapping of every expert tensor plus recorded non-expert names."""

    entries: list[ExpertTensorMapping]
    geometry: dict[str, int]
    non_expert_tensors: list[str] = field(default_factory=list)

    @property
    def weight_entries(self) -> list[ExpertTensorMapping]:
        return [e for e in self.entries if e.kind == WEIGHT]

    @property
    def scale_entries(self) -> list[ExpertTensorMapping]:
        return [e for e in self.entries if e.kind == WEIGHT_SCALE]

    @property
    def offset_entries(self) -> list[ExpertTensorMapping]:
        return [e for e in self.entries if e.kind == WEIGHT_OFFSET]


# --------------------------------------------------------------------------- #
# Geometry helpers
# --------------------------------------------------------------------------- #
def geometry_from_manifest(manifest: Mapping | str | Path) -> dict[str, int]:
    """Extract the four W8A8 geometry values from the checkpoint manifest.

    Accepts the parsed manifest dict, a manifest that is already a bare
    geometry block, or a path to the manifest JSON file.
    """
    if isinstance(manifest, (str, Path)):
        import json

        manifest = json.loads(Path(manifest).read_text())
    geometry = manifest.get("geometry", manifest) if isinstance(manifest, Mapping) else manifest
    try:
        return {key: int(geometry[key]) for key in _REQUIRED_GEOMETRY_KEYS}
    except KeyError as exc:
        raise WeightMappingError(
            f"manifest geometry is missing required key {exc.args[0]!r}; need all of {list(_REQUIRED_GEOMETRY_KEYS)}"
        ) from exc


def _normalize_geometry(geometry: Mapping[str, int]) -> dict[str, int]:
    try:
        norm = {key: int(geometry[key]) for key in _REQUIRED_GEOMETRY_KEYS}
    except KeyError as exc:
        raise WeightMappingError(
            f"geometry is missing required key {exc.args[0]!r}; need all of {list(_REQUIRED_GEOMETRY_KEYS)}"
        ) from exc
    for key, value in norm.items():
        if value <= 0:
            raise WeightMappingError(f"geometry[{key!r}] must be positive, got {value}")
    return norm


# --------------------------------------------------------------------------- #
# Expected set + per-tensor mapping
# --------------------------------------------------------------------------- #
def expected_expert_tensor_names(geometry: Mapping[str, int]) -> set[str]:
    """Build the full expected expert-tensor name set from the manifest geometry.

    Returns 3 x (layers x experts x projections) names: one weight, one scale and
    one offset per projection.
    """
    geom = _normalize_geometry(geometry)
    layers = geom["num_hidden_layers"]
    experts = geom["num_experts"]
    names: set[str] = set()
    for layer in range(layers):
        for expert in range(experts):
            base = f"model.language_model.layers.{layer}.mlp.experts.{expert}"
            for proj in PROJECTIONS:
                for kind in KINDS:
                    names.add(f"{base}.{proj}.{kind}")
    return names


def _expected_dtype(kind: str) -> torch.dtype:
    # Quantized weight is int8; per-channel scale/offset are float32.
    return torch.int8 if kind == WEIGHT else torch.float32


def map_expert_tensor(name: str, geometry: Mapping[str, int]) -> ExpertTensorMapping:
    """Resolve one source expert tensor name to its fused-MoE placement.

    Raises :class:`WeightMappingError` if the name is not an expert tensor or if
    its layer/expert index is outside the geometry.
    """
    geom = _normalize_geometry(geometry)
    match = _EXPERT_RE.match(name)
    if match is None:
        raise WeightMappingError(
            f"{name!r} is not a Qwen4Exp W8A8 expert tensor "
            f"(expected model.language_model.layers.{{L}}.mlp.experts.{{E}}.{{proj}}.{{kind}})",
            tensors=[name],
        )
    layer = int(match["layer"])
    expert = int(match["expert"])
    proj = match["proj"]
    kind = match["kind"]

    if not 0 <= layer < geom["num_hidden_layers"]:
        raise WeightMappingError(
            f"{name!r}: layer index {layer} out of range [0, {geom['num_hidden_layers']})",
            tensors=[name],
        )
    if not 0 <= expert < geom["num_experts"]:
        raise WeightMappingError(
            f"{name!r}: expert index {expert} out of range [0, {geom['num_experts']})",
            tensors=[name],
        )

    moe = geom["moe_intermediate_size"]
    hidden = geom["hidden_size"]
    is_w13 = proj in W13_PROJECTIONS
    # Row (dim-0) size of the source tensor.
    out_dim = moe if is_w13 else hidden
    in_dim = hidden if is_w13 else moe

    if kind == WEIGHT:
        target_param = "w13_weight" if is_w13 else "w2_weight"
        expected_shape: tuple[int, ...] = (out_dim, in_dim)
    else:
        prefix = "w13_" if is_w13 else "w2_"
        target_param = f"{prefix}{kind}"
        expected_shape = (out_dim, 1)

    # gate occupies the first `moe` rows of w13_*, up the next `moe`; down is full.
    row_start = moe if proj == UP_PROJ else 0
    row_stop = row_start + out_dim

    return ExpertTensorMapping(
        source_name=name,
        layer=layer,
        expert=expert,
        proj=proj,
        kind=kind,
        target_param=target_param,
        expert_index=expert,
        row_start=row_start,
        row_stop=row_stop,
        expected_dtype=_expected_dtype(kind),
        expected_shape=expected_shape,
    )


# --------------------------------------------------------------------------- #
# Dtype/shape checks (all routed through the frozen policy)
# --------------------------------------------------------------------------- #
def _to_torch_dtype(raw: object) -> torch.dtype:
    if isinstance(raw, torch.dtype):
        return raw
    if isinstance(raw, str):
        try:
            return _ST_DTYPE_TO_TORCH[raw.upper()]
        except KeyError as exc:
            raise WeightMappingError(f"unknown safetensors dtype string {raw!r}") from exc
    raise WeightMappingError(f"unsupported dtype value {raw!r} (expected torch.dtype or safetensors string)")


def _meta_shape(meta: TensorMeta, name: str) -> tuple[int, ...]:
    try:
        shape = meta["shape"]
    except (KeyError, TypeError) as exc:
        raise WeightMappingError(f"{name!r}: index entry has no 'shape'", tensors=[name]) from exc
    return tuple(int(dim) for dim in shape)


def _meta_dtype(meta: TensorMeta, name: str) -> torch.dtype:
    try:
        raw = meta["dtype"]
    except (KeyError, TypeError) as exc:
        raise WeightMappingError(f"{name!r}: index entry has no 'dtype'", tensors=[name]) from exc
    return _to_torch_dtype(raw)


def _assert_policy_contract(policy: Qwen4ExpDtypePolicy) -> None:
    """Guard against a dtype-policy drift that would silently break mapping."""
    if policy.main_dtype is not torch.float16:
        raise WeightMappingError(
            f"dtype policy main_dtype is {policy.main_dtype}; the 310P W8A8 mapping assumes float16 non-expert tensors"
        )
    if policy.accumulation_dtype is not torch.float32:
        raise WeightMappingError(
            f"dtype policy accumulation_dtype is {policy.accumulation_dtype}; the W8A8 "
            "per-channel scale/offset assume float32"
        )


def _check_expert_tensor(mapping: ExpertTensorMapping, meta: TensorMeta) -> None:
    name = mapping.source_name
    # Dtype first (policy check), then shape.
    actual_dtype = _meta_dtype(meta, name)
    if actual_dtype != mapping.expected_dtype:
        kind_label = "quantized weight" if mapping.kind == WEIGHT else mapping.kind.replace("_", " ")
        raise TensorDtypeError(
            f"{name!r}: expert {kind_label} must be {mapping.expected_dtype} "
            f"(per frozen W8A8 dtype policy), got {actual_dtype}",
            tensors=[name],
        )
    actual_shape = _meta_shape(meta, name)
    if actual_shape != mapping.expected_shape:
        raise TensorShapeError(
            f"{name!r}: expected shape {mapping.expected_shape} for {mapping.proj}.{mapping.kind}, got {actual_shape}",
            tensors=[name],
        )


def _check_non_expert_tensor(name: str, meta: TensorMeta, policy: Qwen4ExpDtypePolicy) -> None:
    """Non-expert tensors stay F16; integer index tensors are tolerated."""
    actual_dtype = _meta_dtype(meta, name)
    if actual_dtype.is_floating_point and actual_dtype != policy.main_dtype:
        raise TensorDtypeError(
            f"{name!r}: non-expert tensor must be {policy.main_dtype} (F16) per the frozen "
            f"dtype policy, got {actual_dtype}",
            tensors=[name],
        )


# --------------------------------------------------------------------------- #
# Top-level validation
# --------------------------------------------------------------------------- #
def _iter_items(provided: ProvidedIndex) -> list[tuple[str, TensorMeta]]:
    if isinstance(provided, Mapping):
        return list(provided.items())
    return [(name, meta) for name, meta in provided]


def validate_expert_weight_map(
    provided: ProvidedIndex,
    geometry: Mapping[str, int],
    *,
    policy: Qwen4ExpDtypePolicy | None = None,
) -> WeightMapping:
    """Validate a checkpoint index against the frozen W8A8 expert contract.

    ``provided`` maps (or lists) tensor name -> safetensors-style meta
    (``{"dtype": <str|torch.dtype>, "shape": [...]}``). The expected expert-tensor
    set is derived from ``geometry`` (driven from the checkpoint manifest).

    Rejects, with an actionable error, any of:
      * duplicate source tensors (:class:`DuplicateTensorError`),
      * missing expected expert tensors (:class:`MissingTensorError`),
      * unexpected/extra expert tensors (:class:`ExtraTensorError`),
      * wrong shape (:class:`TensorShapeError`),
      * wrong dtype, expert or non-expert (:class:`TensorDtypeError`).

    On success returns a :class:`WeightMapping` covering all expert projections.
    """
    policy = policy or ASCEND_QWEN4EXP_DTYPE_POLICY
    _assert_policy_contract(policy)
    geom = _normalize_geometry(geometry)

    items = _iter_items(provided)

    # 1. Duplicate source tensors.
    seen: dict[str, TensorMeta] = {}
    duplicates: list[str] = []
    for name, meta in items:
        if name in seen:
            duplicates.append(name)
        seen[name] = meta
    if duplicates:
        preview = ", ".join(sorted(set(duplicates))[:5])
        raise DuplicateTensorError(
            f"{len(set(duplicates))} duplicate source tensor(s) in checkpoint index: {preview}",
            tensors=sorted(set(duplicates)),
        )

    # 2. Partition expert vs non-expert.
    expert_meta: dict[str, TensorMeta] = {}
    non_expert_meta: dict[str, TensorMeta] = {}
    for name, meta in seen.items():
        if _EXPERT_RE.match(name):
            expert_meta[name] = meta
        else:
            non_expert_meta[name] = meta

    # 3. Missing / extra against the manifest-driven expected set.
    expected_names = expected_expert_tensor_names(geom)
    provided_names = set(expert_meta)
    missing = expected_names - provided_names
    extra = provided_names - expected_names
    if missing:
        preview = ", ".join(sorted(missing)[:5])
        raise MissingTensorError(
            f"{len(missing)} expected expert tensor(s) missing from checkpoint index "
            f"(e.g. {preview}); expected {len(expected_names)} expert tensors for "
            f"{geom['num_hidden_layers']} layers x {geom['num_experts']} experts",
            tensors=sorted(missing),
        )
    if extra:
        preview = ", ".join(sorted(extra)[:5])
        raise ExtraTensorError(
            f"{len(extra)} unexpected expert-shaped tensor(s) in checkpoint index "
            f"(e.g. {preview}); index does not match the frozen geometry",
            tensors=sorted(extra),
        )

    # 4. Per-tensor dtype+shape check, then build the mapping.
    entries: list[ExpertTensorMapping] = []
    for name in expected_names:
        mapping = map_expert_tensor(name, geom)
        _check_expert_tensor(mapping, expert_meta[name])
        entries.append(mapping)

    # 5. Non-expert tensors are checked against the policy but not mapped.
    for name, meta in non_expert_meta.items():
        _check_non_expert_tensor(name, meta, policy)

    return WeightMapping(
        entries=entries,
        geometry=geom,
        non_expert_tensors=sorted(non_expert_meta),
    )
