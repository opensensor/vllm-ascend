# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Weight mapping for GLM-5.3-Flash (``glm5_next``) W2 experts on Ascend 310P (G6).

This is the *seam between the on-disk GLM W2 artifact and the E1.3 device method*.
The artifact produced by ``tools/glm_w2/convert_full.py``
(``/run/media/matteius/20TB-drive/models/GLM-5.3-Flash-W2-310p`` -- 19 shards +
``model.safetensors.index.json``) **preserves the source-style tensor names**
(unlike the DeepSeek V4.1 converter, which renames to ``ffn.experts.*.{w1,w2,w3}``).
A GLM routed-expert weight therefore streams in as::

    model.language_model.layers.{L}.mlp.experts.{E}.{gate_proj|up_proj|down_proj}_codes  (U8)
    model.language_model.layers.{L}.mlp.experts.{E}.{gate_proj|up_proj|down_proj}_scale  (F32 [out/32,in/32])

with everything else (router ``mlp.gate.weight`` + ``e_score_correction_bias``,
FP16 ``shared_experts.*``, dense layers 0-2 ``mlp.{gate,up,down}_proj.*``, MLA /
DSA indexer / self-attn / norms / ``hc_{attn,ffn}_{base,fn,scale}`` hyper-connection
/ ``lm_head`` / ``embed_tokens``) as plain FP16. The ``model.visual.*`` vision
tower is excluded (text-only deployment).

Two responsibilities, both **metadata-only** (no rank ever materialises the
97 GB checkpoint or the full 288-expert bank):

1. **Name -> destination classification + expert fusion** (:func:`classify_tensor`,
   :func:`map_expert_tensor`). The E1.3 method
   (:class:`~vllm_ascend._310p.quantization.methods.w2_dynamic.AscendW2DynamicFusedMoEMethod310`)
   fuses gate/up into a single ``w13`` param, so GLM's ``gate_proj`` maps to row 0
   of ``w13_*`` and ``up_proj`` to the ``moe_intermediate_size`` row offset;
   ``down_proj`` maps to ``w2_*``.

2. **Coverage validation** (:func:`validate_weight_map`). Every MoE-carrying block
   (decoder layers ``first_k_dense_replace..num_hidden_layers-1`` plus the MTP
   layer) must contribute exactly its ``n_routed_experts`` x 6 expert tensors;
   dense layers 0..first_k_dense_replace-1 carry NO experts and are rejected if
   they do. The DeepSeek W2 format constants + ``TensorMeta`` + rejection classes
   are reused verbatim (the packed-W2 storage contract is shared).

GLM has **no Engram host table and no separate sparse-indexer packed format** --
the DeepSeek placement simulator's host-table dedup is not needed here, so this
module keeps a lean per-chip byte estimate (routed W2 experts sharded EP4, FP16
sharded TP4) instead of importing the DeepSeek accountant.
"""

from __future__ import annotations

import json
import re
import struct
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

from tools.deepseek_w2.w2_format import (
    W2_BLOCK_COLS,
    W2_BLOCK_ROWS,
    W2_CODES_PER_BYTE,
)

# Reuse the DeepSeek W2 metadata + rejection primitives verbatim: the packed-W2
# storage contract (uint8 codes + fp32 [32,32] block scales) is identical across
# both models, only the *names* differ.
from vllm_ascend.models.deepseek_v41.weight_mapping import (
    DtypeMismatchError,
    DuplicateTensorError,
    ExpertTensorMapping,
    ExtraTensorError,
    MissingTensorError,
    ShapeMismatchError,
    TensorMeta,
    WeightClass,
    WeightMappingError,
)

__all__ = [
    "WeightClass",
    "WeightMappingError",
    "MissingTensorError",
    "ExtraTensorError",
    "DuplicateTensorError",
    "ShapeMismatchError",
    "DtypeMismatchError",
    "TensorMeta",
    "ExpertTensorMapping",
    "classify_tensor",
    "map_expert_tensor",
    "expected_expert_shape",
    "expected_expert_dtype",
    "validate_expert_tensor",
    "moe_block_ids",
    "expected_expert_blocks",
    "validate_weight_map",
    "iter_artifact_tensor_metas",
    "estimate_per_chip_bytes",
    "DEFAULT_TP_SIZE",
    "DEFAULT_EP_SIZE",
    "PER_CHIP_DEVICE_WEIGHT_BUDGET_GIB",
    "PER_CHIP_DEVICE_WEIGHT_BUDGET_BYTES",
]

DEFAULT_TP_SIZE = 4
DEFAULT_EP_SIZE = 4

_BYTES_PER_GIB = 1024**3
# Each 310P memory domain is 48 GiB; keep the same 40 GiB device-weight ceiling
# used for DeepSeek. GLM is far lighter (~20 GiB W2 experts/chip, no Engram).
PER_CHIP_DEVICE_WEIGHT_BUDGET_GIB = 40.0
PER_CHIP_DEVICE_WEIGHT_BUDGET_BYTES = int(round(PER_CHIP_DEVICE_WEIGHT_BUDGET_GIB * _BYTES_PER_GIB))

_CODES_DTYPE = "U8"
_SCALE_DTYPE = "F32"

# GLM routed-expert artifact name:
#   model.language_model.layers.{L}.mlp.experts.{E}.{gate_proj|up_proj|down_proj}_{codes|scale}
_EXPERT_RE = re.compile(
    r"^model\.language_model\.layers\.(?P<layer>\d+)\.mlp\.experts\.(?P<expert>\d+)\."
    r"(?P<proj>gate_proj|up_proj|down_proj)_(?P<kind>codes|scale)$"
)

# Text-only deployment: drop the GLM vision tower.
_EXCLUDE_RE = re.compile(r"(^model\.visual\.|(?:^|\.)visual\.|image_token)")

# gate_proj (w1) + up_proj (w3) fuse into w13; down_proj (w2) stands alone.
_PROJ_TO_SLOT = {"gate_proj": "w1", "up_proj": "w3", "down_proj": "w2"}


def classify_tensor(name: str) -> WeightClass:
    """Route a GLM artifact tensor name to its :class:`WeightClass` lane.

    GLM has no Engram host table, so only ``W2_EXPERT`` / ``FP16`` / ``EXCLUDE``
    are ever returned.
    """
    if _EXCLUDE_RE.search(name):
        return WeightClass.EXCLUDE
    if _EXPERT_RE.match(name):
        return WeightClass.W2_EXPERT
    return WeightClass.FP16


def map_expert_tensor(name: str, geometry: dict) -> ExpertTensorMapping:
    """Map a GLM artifact expert tensor name to its fused E1.3 target slot.

    Raises :class:`ValueError` if ``name`` is not a routed-expert tensor.
    """
    match = _EXPERT_RE.match(name)
    if match is None:
        raise ValueError(f"{name!r} is not a GLM routed-expert tensor")
    slot = _PROJ_TO_SLOT[match.group("proj")]
    kind = match.group("kind")
    block = f"layers.{match.group('layer')}"
    expert_id = int(match.group("expert"))
    inter = geometry["moe_intermediate_size"]

    if slot == "w2":  # down_proj -> standalone w2 param
        target = "w2_codes" if kind == "codes" else "w2_scale"
        return ExpertTensorMapping(
            source_name=name,
            block=block,
            expert_id=expert_id,
            slot=slot,
            kind=kind,
            target_param=target,
            row_offset=0,
            fuses_into_w13=False,
        )
    # gate_proj (w1) / up_proj (w3) fuse into w13 along the output axis.
    target = "w13_codes" if kind == "codes" else "w13_scale"
    if slot == "w1":
        row_offset = 0
    else:  # w3 (up_proj) stacks after the full gate half
        row_offset = inter if kind == "codes" else inter // W2_BLOCK_ROWS
    return ExpertTensorMapping(
        source_name=name,
        block=block,
        expert_id=expert_id,
        slot=slot,
        kind=kind,
        target_param=target,
        row_offset=row_offset,
        fuses_into_w13=True,
    )


def expected_expert_shape(slot: str, kind: str, geometry: dict) -> tuple[int, int]:
    """Geometry-derived expected shape of a *source* GLM expert tensor.

    Codes are packed along the input axis (``W2_CODES_PER_BYTE`` per byte);
    scales are one fp32 per ``[32, 32]`` block. Matches the converted artifact:
    gate/up codes ``[2048, 1024]`` scale ``[64, 128]``; down codes ``[4096, 512]``
    scale ``[128, 64]`` for hidden 4096 / inter 2048.
    """
    hidden = geometry["hidden_size"]
    inter = geometry["moe_intermediate_size"]
    if slot in ("w1", "w3"):  # gate/up: [inter, hidden]
        out_features, in_features = inter, hidden
    elif slot == "w2":  # down: [hidden, inter]
        out_features, in_features = hidden, inter
    else:  # pragma: no cover - defensive
        raise ValueError(f"unknown expert slot {slot!r}")
    if kind == "codes":
        return (out_features, in_features // W2_CODES_PER_BYTE)
    return (out_features // W2_BLOCK_ROWS, in_features // W2_BLOCK_COLS)


def expected_expert_dtype(kind: str) -> str:
    """Expected storage dtype token for an expert tensor of ``kind``."""
    return _CODES_DTYPE if kind == "codes" else _SCALE_DTYPE


def validate_expert_tensor(meta: TensorMeta, geometry: dict) -> ExpertTensorMapping:
    """Validate a single GLM expert tensor's shape + dtype and return its mapping.

    Raises :class:`ShapeMismatchError` / :class:`DtypeMismatchError`.
    """
    mapping = map_expert_tensor(meta.name, geometry)
    if mapping.kind == "codes":
        # Codes width encodes the per-layer bit depth: W2 packs 4 codes/byte
        # (in//4), W4 packs 2 (in//2). Accept either so mixed-precision W2/W4
        # expert banks validate; the runtime infers bits from the same width.
        hidden = geometry["hidden_size"]
        inter = geometry["moe_intermediate_size"]
        out_features, in_features = (inter, hidden) if mapping.slot in ("w1", "w3") else (hidden, inter)
        accepted = {(out_features, in_features // 4), (out_features, in_features // 2)}
        if tuple(meta.shape) not in accepted:
            raise ShapeMismatchError(
                f"{meta.name}: codes shape {tuple(meta.shape)} not in {sorted(accepted)} "
                f"(slot={mapping.slot}; W2=in//4, W4=in//2)"
            )
    else:
        expected_shape = expected_expert_shape(mapping.slot, mapping.kind, geometry)
        if tuple(meta.shape) != expected_shape:
            raise ShapeMismatchError(
                f"{meta.name}: shape {tuple(meta.shape)} != expected {expected_shape} "
                f"(slot={mapping.slot}, kind={mapping.kind})"
            )
    expected_dtype = expected_expert_dtype(mapping.kind)
    if meta.dtype != expected_dtype:
        raise DtypeMismatchError(
            f"{meta.name}: dtype {meta.dtype!r} != expected {expected_dtype!r} (kind={mapping.kind})"
        )
    return mapping


def moe_block_ids(geometry: dict) -> list[int]:
    """The decoder + MTP layer indices that carry routed experts.

    GLM's first ``first_k_dense_replace`` layers are plain dense MLP (no experts);
    decoder layers ``[first_k_dense_replace, num_hidden_layers)`` are MoE, plus the
    single MTP layer at ``mtp_layer_index`` (defaults to ``num_hidden_layers``).
    """
    first_dense = geometry.get("first_k_dense_replace", 0)
    num_layers = geometry["num_hidden_layers"]
    ids = list(range(first_dense, num_layers))
    num_mtp = geometry.get("num_nextn_predict_layers", 0)
    if num_mtp:
        mtp_index = geometry.get("mtp_layer_index", num_layers)
        for m in range(num_mtp):
            ids.append(mtp_index + m)
    return ids


def expected_expert_blocks(geometry: dict) -> dict[str, int]:
    """Ground-truth ``{block: num_experts}`` schema for GLM.

    Every MoE-carrying block (decoder layers past ``first_k_dense_replace`` plus
    the MTP layer) carries ``n_routed_experts``.
    """
    n_routed = geometry["n_routed_experts"]
    return {f"layers.{layer}": n_routed for layer in moe_block_ids(geometry)}


# GLM's per-expert source tensors (gate/up/down x codes/scale).
_EXPERT_PROJS: tuple[str, ...] = ("gate_proj", "up_proj", "down_proj")
_EXPERT_KINDS: tuple[str, ...] = ("codes", "scale")


def _expert_name(layer: int, expert: int, proj: str, kind: str) -> str:
    return f"model.language_model.layers.{layer}.mlp.experts.{expert}.{proj}_{kind}"


def validate_weight_map(names: Iterable[str], geometry: dict) -> dict[str, int]:
    """Validate the routed-expert coverage of a GLM name stream.

    Checks (in order): duplicate names, then that every ``{block, expert}`` cell in
    the geometry schema has exactly its six expert tensors, then that no unexpected
    routed-expert tensor is present (a dense layer 0..first_k_dense_replace-1 that
    carries experts, or an out-of-range expert id, trips :class:`ExtraTensorError`).
    Non-expert names are ignored here (shape/dtype-validated when streamed).

    Raises :class:`DuplicateTensorError`, :class:`MissingTensorError` or
    :class:`ExtraTensorError`.
    """
    seen: set[str] = set()
    expert_names: set[str] = set()
    for name in names:
        if name in seen:
            raise DuplicateTensorError(f"tensor {name!r} appears more than once in the source stream")
        seen.add(name)
        if _EXPERT_RE.match(name):
            expert_names.add(name)

    schema = expected_expert_blocks(geometry)
    expected: set[str] = set()
    for block, num_experts in schema.items():
        layer = int(block.split(".")[1])
        for expert in range(num_experts):
            for proj in _EXPERT_PROJS:
                for kind in _EXPERT_KINDS:
                    expected.add(_expert_name(layer, expert, proj, kind))

    missing = expected - expert_names
    if missing:
        sample = sorted(missing)[:5]
        raise MissingTensorError(f"{len(missing)} routed-expert tensor(s) missing from the artifact, e.g. {sample}")
    extra = expert_names - expected
    if extra:
        sample = sorted(extra)[:5]
        raise ExtraTensorError(f"{len(extra)} unexpected routed-expert tensor(s) in the artifact, e.g. {sample}")
    return {
        "expert_blocks": len(schema),
        "expert_tensors": len(expected),
        "total_names": len(seen),
    }


def iter_artifact_tensor_metas(artifact_dir: str | Path) -> Iterator[TensorMeta]:
    """Stream :class:`TensorMeta` for every GLM artifact tensor, headers only.

    Reads ``model.safetensors.index.json`` for the tensor->shard map, then opens
    each shard once and parses only its JSON header (8-byte length prefix + header
    bytes); no tensor payload is ever read, so the working set stays at a single
    shard header regardless of the 97 GB total. Raises :class:`FileNotFoundError`
    if the artifact volume is not mounted.
    """
    root = Path(artifact_dir)
    index_path = root / "model.safetensors.index.json"
    if not index_path.is_file():
        raise FileNotFoundError(f"artifact index not found: {index_path}")
    weight_map: dict[str, str] = json.loads(index_path.read_text())["weight_map"]
    by_shard: dict[str, list[str]] = {}
    for name, shard in weight_map.items():
        by_shard.setdefault(shard, []).append(name)
    for shard in sorted(by_shard):
        shard_path = root / shard
        with shard_path.open("rb") as handle:
            header_len = struct.unpack("<Q", handle.read(8))[0]
            header = json.loads(handle.read(header_len))
        for name in by_shard[shard]:
            meta = header[name]
            yield TensorMeta(name=name, dtype=meta["dtype"], shape=tuple(meta["shape"]))


@dataclass
class Glm5NextW2Placement:
    """Lean per-chip byte estimate for the GLM W2 deployment (no Engram)."""

    world_size: int
    w2_expert_bytes: int = 0
    fp16_bytes: int = 0
    excluded_bytes: int = 0
    w2_expert_count: int = 0
    fp16_count: int = 0
    excluded_count: int = 0

    @property
    def per_chip_device_bytes(self) -> int:
        """Both lanes shard across the mesh, so device bytes split evenly."""
        return (self.w2_expert_bytes + self.fp16_bytes) // self.world_size

    def as_dict(self) -> dict:
        return {
            "world_size": self.world_size,
            "w2_expert_bytes": self.w2_expert_bytes,
            "fp16_bytes": self.fp16_bytes,
            "excluded_bytes": self.excluded_bytes,
            "w2_expert_count": self.w2_expert_count,
            "fp16_count": self.fp16_count,
            "excluded_count": self.excluded_count,
            "per_chip_device_bytes": self.per_chip_device_bytes,
            "per_chip_device_gib": self.per_chip_device_bytes / _BYTES_PER_GIB,
            "budget_gib": PER_CHIP_DEVICE_WEIGHT_BUDGET_GIB,
            "within_budget": self.per_chip_device_bytes <= PER_CHIP_DEVICE_WEIGHT_BUDGET_BYTES,
        }


def estimate_per_chip_bytes(
    tensor_metas: Iterable[TensorMeta],
    *,
    world_size: int = DEFAULT_EP_SIZE,
) -> Glm5NextW2Placement:
    """Estimate per-chip device bytes from a stream of GLM tensor metas.

    W2 routed experts shard EP4 and FP16 tensors shard TP4; with ``world_size``
    ranks both lanes divide evenly (no rank holds the full 288-expert bank). GLM
    has no shared host table, so unlike DeepSeek there is nothing to dedup.
    """
    placement = Glm5NextW2Placement(world_size=world_size)
    for meta in tensor_metas:
        cls = classify_tensor(meta.name)
        if cls is WeightClass.EXCLUDE:
            placement.excluded_bytes += meta.num_bytes
            placement.excluded_count += 1
        elif cls is WeightClass.W2_EXPERT:
            placement.w2_expert_bytes += meta.num_bytes
            placement.w2_expert_count += 1
        else:
            placement.fp16_bytes += meta.num_bytes
            placement.fp16_count += 1
    return placement
