# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Weight mapping + streamed W2 placement for DeepSeek V4.1 on Ascend 310P (E3.4).

This module is the *seam between the on-disk W2 artifact and the device method*.
The artifact produced by ``tools/deepseek_w2/convert_full.py`` (50 shards +
``model.safetensors.index.json`` + ``conversion_manifest.json``) stores the
routed experts as split ``{w1|w2|w3}_codes`` (uint8) + ``{w1|w2|w3}_scale``
(fp32 ``[out/32, in/32]``) tensors, the two Engram n-gram tables as ~W4 packed
host rows, and everything else (MLA / indexer / dense / shared-expert / LM-head
/ norms / gates / MTP heads / Engram q,k) as FP16.

Two responsibilities live here, both of which stay *metadata-only* (no rank ever
materialises the 258 GB checkpoint or the full 384-expert bank):

1. **Name -> destination classification + expert fusion** (:func:`classify_tensor`,
   :func:`map_expert_tensor`). The E1.3 method
   (:class:`~vllm_ascend._310p.quantization.methods.w2_dynamic.AscendW2DynamicFusedMoEMethod310`)
   fuses gate/up into a single ``w13`` param, so the artifact's ``w1``/``w3``
   split tensors are mapped to row-offset slices of ``w13_codes`` / ``w13_scale``
   and ``w2`` to ``w2_codes`` / ``w2_scale``. This is the interface E4.1 drives to
   fill the method's params from a streaming source.

2. **TP4 / EP4 placement simulation + E0.5 accounting**
   (:class:`DeepSeekV41W2PlacementSimulator`). It walks the artifact's index +
   shard *headers* (or a synthetic metadata stream in the UT), assigns every
   logical tensor to device ranks (or the shared Engram host table), and reports
   predicted per-chip device bytes through the DeepSeek W2 accountant
   (``vllm_ascend.observability.deepseek_w2_mem_accounting``). The Engram host
   table is recorded once (never ``x world_size``); the routed-expert bank is
   sharded so no rank holds the whole 384-expert bank.

The placement logic reuses the Qwen4Exp streaming primitives from
``vllm_ascend._310p.sharded_state_loader_310p`` (``ShardPolicy``, ``TensorGroup``,
``PlacementLedger``, ``DoubleInstantiationError``, ``_split_even``) rather than
re-deriving them. Dtypes are read from the E2.1 policy; the accountant is the
E0.5 DeepSeek layer over the Qwen per-rank accountant.
"""

from __future__ import annotations

import json
import re
import struct
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from tools.deepseek_w2.w2_format import (
    W2_BLOCK_COLS,
    W2_BLOCK_ROWS,
    W2_CODES_PER_BYTE,
)
from vllm_ascend._310p.sharded_state_loader_310p import (
    DoubleInstantiationError,
    PlacementLedger,
    ShardPolicy,
    TensorGroup,
    _split_even,
)
from vllm_ascend.models.deepseek_v41.dtype_policy import (
    ASCEND_DEEPSEEKV41_DTYPE_POLICY,
)
from vllm_ascend.observability.deepseek_w2_mem_accounting import (
    DeepSeekW2MemComponent,
    DeepSeekW2MemoryAccountant,
    MemComponent,
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
    "expected_expert_blocks",
    "validate_weight_map",
    "iter_artifact_tensor_metas",
    "DeepSeekV41W2PlacementSimulator",
    "DEFAULT_TP_SIZE",
    "DEFAULT_EP_SIZE",
    "PER_CHIP_DEVICE_WEIGHT_BUDGET_GIB",
    "PER_CHIP_DEVICE_WEIGHT_BUDGET_BYTES",
]

DEFAULT_TP_SIZE = 4
DEFAULT_EP_SIZE = 4

_BYTES_PER_GIB = 1024**3

# Per-chip device *weight* budget on the four-chip Ascend 310P DeepSeek V4.1
# deployment. Each 310P memory domain is 48 GiB; this budget reserves headroom
# for the KV / indexer caches, the INT8 active-expert unpack cache and kernel
# workspaces. The balanced TP4/EP4 prediction from the real artifact lands at
# ~36.95 GiB/chip (W2 experts + FP16), comfortably under this ceiling.
PER_CHIP_DEVICE_WEIGHT_BUDGET_GIB = 40.0
PER_CHIP_DEVICE_WEIGHT_BUDGET_BYTES = int(round(PER_CHIP_DEVICE_WEIGHT_BUDGET_GIB * _BYTES_PER_GIB))

# safetensors / manifest dtype token -> bytes per element. The artifact is
# already dequantised into these storage dtypes (codes uint8, scales fp32,
# everything else fp16); the F8 tokens are kept so a raw source header would not
# crash the walker.
_DTYPE_BYTES: dict[str, int] = {
    "U8": 1,
    "I8": 1,
    "BOOL": 1,
    "F8_E4M3": 1,
    "F8_E8M0": 1,
    "F16": 2,
    "BF16": 2,
    "F32": 4,
    "I32": 4,
    "I64": 8,
    "F64": 8,
}

# uint8 packed 2-bit codes and fp32 block scales, per the E1.1 pack contract.
_CODES_DTYPE = "U8"
_SCALE_DTYPE = "F32"

# Routed-expert tensor name: ``layers.{L}.ffn.experts.{E}.{w1|w2|w3}_{codes|scale}``
# or the MTP equivalent ``mtp.{N}.ffn.experts.{E}...``.
_EXPERT_RE = re.compile(
    r"^(?P<block>(?:layers|mtp)\.\d+)\.ffn\.experts\.(?P<expert>\d+)\."
    r"(?P<slot>w1|w2|w3)_(?P<kind>codes|scale)$"
)

# Engram ~W4 host-table tensors: the partitioned embedding rows
# (``embed_codes.pN`` / ``embed_scale.pN``) and the wkv projection
# (``wkv_codes`` / ``wkv_scale``). The Engram q/k projections are tiny FP16
# tensors and deliberately NOT matched here (they ride the FP16 device lane).
_ENGRAM_HOST_RE = re.compile(
    r"^(?:layers|mtp)\.\d+\.engram\.(?:embed_codes\.p\d+|embed_scale\.p\d+|wkv_codes|wkv_scale)$"
)

# Text-only deployment: the vision tower / aligner / image-token tensors are
# excluded from every device and host budget.
_EXCLUDE_RE = re.compile(r"(^vision\.|\baligner\b|image_token)")


class WeightClass(str, Enum):
    """Destination lane for an artifact tensor."""

    # Routed-expert 2-bit codes + fp32 block scales -> E1.3 method params,
    # sharded across the device mesh (TP4 columns / EP4 experts).
    W2_EXPERT = "w2_expert"
    # ~W4 Engram n-gram table (embedding rows + wkv) -> single shared host copy.
    ENGRAM_HOST = "engram_host"
    # FP16 MLA / indexer / dense / shared-expert / LM-head / embed / norms /
    # gates / MTP heads / Engram q,k -> device, sharded.
    FP16 = "fp16"
    # Vision tower (text-only deployment) -> dropped.
    EXCLUDE = "exclude"


# --------------------------------------------------------------------------- #
# Rejection classes (missing / extra / duplicate / wrong-shape / wrong-dtype)
# --------------------------------------------------------------------------- #


class WeightMappingError(RuntimeError):
    """Base class for every artifact-vs-schema rejection."""


class MissingTensorError(WeightMappingError):
    """An expected expert tensor is absent from the artifact."""


class ExtraTensorError(WeightMappingError):
    """The artifact carries an expert tensor the schema does not expect."""


class DuplicateTensorError(WeightMappingError):
    """A tensor name appears more than once in the source stream."""


class ShapeMismatchError(WeightMappingError):
    """A tensor's shape disagrees with the geometry-derived expectation."""


class DtypeMismatchError(WeightMappingError):
    """A tensor's storage dtype disagrees with the E1.1 pack / E2.1 policy."""


# --------------------------------------------------------------------------- #
# Tensor metadata (name + dtype token + shape); never a payload
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class TensorMeta:
    """Header-level metadata for one checkpoint tensor (no payload)."""

    name: str
    dtype: str
    shape: tuple[int, ...]

    @property
    def num_bytes(self) -> int:
        elements = 1
        for dim in self.shape:
            elements *= dim
        try:
            return elements * _DTYPE_BYTES[self.dtype]
        except KeyError as exc:  # pragma: no cover - defensive
            raise WeightMappingError(f"unknown dtype token {self.dtype!r} for tensor {self.name!r}") from exc


# --------------------------------------------------------------------------- #
# Classification + expert fusion mapping
# --------------------------------------------------------------------------- #


def classify_tensor(name: str) -> WeightClass:
    """Route a checkpoint tensor name to its :class:`WeightClass` lane."""
    if _EXCLUDE_RE.search(name):
        return WeightClass.EXCLUDE
    if _EXPERT_RE.match(name):
        return WeightClass.W2_EXPERT
    if _ENGRAM_HOST_RE.match(name):
        return WeightClass.ENGRAM_HOST
    return WeightClass.FP16


@dataclass(frozen=True)
class ExpertTensorMapping:
    """How one artifact expert tensor lands in the E1.3 fused param layout.

    ``target_param`` is the destination method param (``w13_codes`` /
    ``w13_scale`` / ``w2_codes`` / ``w2_scale``). ``row_offset`` is the starting
    row of this source tensor inside the (possibly fused) target: ``w1`` and
    ``w2`` load at row 0; ``w3`` loads at ``inter`` (codes) / ``inter // 32``
    (scale) because gate (``w1``) and up (``w3``) are concatenated along the
    output axis to form ``w13``.
    """

    source_name: str
    block: str
    expert_id: int
    slot: str  # w1 | w2 | w3
    kind: str  # codes | scale
    target_param: str
    row_offset: int
    fuses_into_w13: bool


def map_expert_tensor(name: str, geometry: dict) -> ExpertTensorMapping:
    """Map an artifact expert tensor name to its fused E1.3 target slot.

    Raises :class:`ValueError` if ``name`` is not a routed-expert tensor.
    """
    match = _EXPERT_RE.match(name)
    if match is None:
        raise ValueError(f"{name!r} is not a routed-expert tensor")
    slot = match.group("slot")
    kind = match.group("kind")
    inter = geometry["moe_intermediate_size"]
    if slot == "w2":
        target = "w2_codes" if kind == "codes" else "w2_scale"
        return ExpertTensorMapping(
            source_name=name,
            block=match.group("block"),
            expert_id=int(match.group("expert")),
            slot=slot,
            kind=kind,
            target_param=target,
            row_offset=0,
            fuses_into_w13=False,
        )
    # w1 (gate) / w3 (up) fuse into w13 along the output axis.
    target = "w13_codes" if kind == "codes" else "w13_scale"
    if slot == "w1":
        row_offset = 0
    else:  # w3 stacks after the full gate half
        row_offset = inter if kind == "codes" else inter // W2_BLOCK_ROWS
    return ExpertTensorMapping(
        source_name=name,
        block=match.group("block"),
        expert_id=int(match.group("expert")),
        slot=slot,
        kind=kind,
        target_param=target,
        row_offset=row_offset,
        fuses_into_w13=True,
    )


def expected_expert_shape(slot: str, kind: str, geometry: dict) -> tuple[int, int]:
    """Geometry-derived expected shape of a *source* expert tensor.

    Codes are packed along the input axis (``W2_CODES_PER_BYTE`` per byte);
    scales are one fp32 per ``[32, 32]`` block.
    """
    hidden = geometry["hidden_size"]
    inter = geometry["moe_intermediate_size"]
    if slot in ("w1", "w3"):  # [inter, hidden]
        out_features, in_features = inter, hidden
    elif slot == "w2":  # down [hidden, inter]
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
    """Validate a single expert tensor's shape + dtype and return its mapping.

    Raises :class:`ShapeMismatchError` / :class:`DtypeMismatchError`.
    """
    mapping = map_expert_tensor(meta.name, geometry)
    expected_shape = expected_expert_shape(mapping.slot, mapping.kind, geometry)
    if tuple(meta.shape) != expected_shape:
        raise ShapeMismatchError(
            f"{meta.name}: shape {tuple(meta.shape)} != expected {expected_shape} "
            f"(slot={mapping.slot}, kind={mapping.kind})"
        )
    expected_dtype = expected_expert_dtype(mapping.kind)
    if meta.dtype != expected_dtype:
        raise DtypeMismatchError(
            f"{meta.name}: dtype {meta.dtype!r} != expected {expected_dtype!r} "
            f"(kind={mapping.kind})"
        )
    return mapping


def expected_expert_blocks(geometry: dict) -> dict[str, int]:
    """Ground-truth ``{block: num_experts}`` schema derived from the manifest.

    Dense decoder layers carry ``n_routed_experts``; the MTP (dSpark) predict
    layers carry ``dspark_n_routed_experts``.
    """
    blocks: dict[str, int] = {}
    n_routed = geometry["n_routed_experts"]
    for layer in range(geometry["num_hidden_layers"]):
        blocks[f"layers.{layer}"] = n_routed
    num_mtp = geometry.get("num_nextn_predict_layers", 0)
    mtp_experts = geometry.get("dspark_n_routed_experts", n_routed)
    for mtp in range(num_mtp):
        blocks[f"mtp.{mtp}"] = mtp_experts
    return blocks


_EXPERT_SLOTS: tuple[str, ...] = (
    "w1_codes",
    "w1_scale",
    "w2_codes",
    "w2_scale",
    "w3_codes",
    "w3_scale",
)


def validate_weight_map(names: Iterable[str], geometry: dict) -> dict[str, int]:
    """Validate the routed-expert coverage of a name stream.

    Checks (in order) for duplicate names, then that every ``{block, expert}``
    cell in the geometry schema has exactly its six expert tensors, then that no
    unexpected expert tensor is present. Non-expert names are ignored here (they
    are shape/dtype-validated when streamed). Returns a small summary dict.

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
        for expert in range(num_experts):
            for slot in _EXPERT_SLOTS:
                expected.add(f"{block}.ffn.experts.{expert}.{slot}")

    missing = expected - expert_names
    if missing:
        sample = sorted(missing)[:5]
        raise MissingTensorError(
            f"{len(missing)} routed-expert tensor(s) missing from the artifact, e.g. {sample}"
        )
    extra = expert_names - expected
    if extra:
        sample = sorted(extra)[:5]
        raise ExtraTensorError(
            f"{len(extra)} unexpected routed-expert tensor(s) in the artifact, e.g. {sample}"
        )
    return {
        "expert_blocks": len(schema),
        "expert_tensors": len(expected),
        "total_names": len(seen),
    }


# --------------------------------------------------------------------------- #
# Streaming artifact metadata reader (headers only)
# --------------------------------------------------------------------------- #


def iter_artifact_tensor_metas(artifact_dir: str | Path) -> Iterator[TensorMeta]:
    """Stream :class:`TensorMeta` for every artifact tensor, headers only.

    Reads ``model.safetensors.index.json`` for the tensor->shard map, then opens
    each shard once and parses only its JSON header (8-byte length prefix +
    header bytes); no tensor payload is ever read, so the working set stays at a
    single shard header regardless of the 258 GB total. Raises
    :class:`FileNotFoundError` if the artifact volume is not mounted.
    """
    root = Path(artifact_dir)
    index_path = root / "model.safetensors.index.json"
    if not index_path.is_file():
        raise FileNotFoundError(f"artifact index not found: {index_path}")
    weight_map: dict[str, str] = json.loads(index_path.read_text())["weight_map"]
    # Group tensor names by shard so each shard header is parsed exactly once.
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


# --------------------------------------------------------------------------- #
# TP4 / EP4 placement simulator + E0.5 accounting
# --------------------------------------------------------------------------- #


@dataclass
class DeepSeekW2Ledger(PlacementLedger):
    """Placement ledger extended with excluded (vision) accounting."""

    excluded_bytes: int = 0
    excluded_count: int = 0


class DeepSeekV41W2PlacementSimulator:
    """Host-side TP4/EP4 placement simulator for the DeepSeek V4.1 W2 artifact.

    Walks a stream of :class:`TensorMeta` (from the real artifact headers or a
    synthetic UT stream), classifies each tensor, and places it:

    * ``W2_EXPERT`` / ``FP16`` -> device, ``SHARDED``. Under ``parallel_mode='tp'``
      the bytes are split evenly across the mesh; under ``'ep'`` each routed-
      expert tensor lands wholesale on ``expert_id % world_size`` (FP16 tensors
      still split evenly). Either way no rank ever holds the full 384-expert bank.
    * ``ENGRAM_HOST`` -> one shared host copy, recorded identically on every rank
      so the accountant dedups it (never ``x world_size``).
    * ``EXCLUDE`` -> dropped (text-only deployment).

    Expert tensors are shape/dtype-validated as they stream (rejection classes
    fire on a malformed artifact). The simulator never allocates a payload: the
    working set is one descriptor plus per-rank integer counters.
    """

    def __init__(
        self,
        geometry: dict,
        *,
        tp_size: int = DEFAULT_TP_SIZE,
        ep_size: int = DEFAULT_EP_SIZE,
        parallel_mode: str = "tp",
        validate_experts: bool = True,
    ) -> None:
        if parallel_mode not in ("tp", "ep"):
            raise ValueError(f"parallel_mode must be 'tp' or 'ep', got {parallel_mode!r}")
        self.geometry = geometry
        self.parallel_mode = parallel_mode
        self.tp_size = tp_size
        self.ep_size = ep_size
        self.world_size = ep_size if parallel_mode == "ep" else tp_size
        self.validate_experts = validate_experts
        # Pin the expert-weight/scale storage dtypes to the E2.1 policy.
        self._policy = ASCEND_DEEPSEEKV41_DTYPE_POLICY
        self._ledger = DeepSeekW2Ledger()

    @classmethod
    def from_manifest(cls, manifest: dict, **kwargs) -> DeepSeekV41W2PlacementSimulator:
        """Build a simulator from the E0.1 conversion manifest's ``geometry``."""
        return cls(manifest["geometry"], **kwargs)

    # -- tensor -> TensorGroup ------------------------------------------------

    def _group_for(self, meta: TensorMeta) -> TensorGroup | None:
        """Classify + validate one tensor, returning its placement group.

        Returns ``None`` for excluded tensors (accounted as excluded, not placed).
        """
        cls = classify_tensor(meta.name)
        if cls is WeightClass.EXCLUDE:
            self._ledger.excluded_bytes += meta.num_bytes
            self._ledger.excluded_count += 1
            return None
        if cls is WeightClass.W2_EXPERT:
            if self.validate_experts:
                mapping = validate_expert_tensor(meta, self.geometry)
                expert_key = mapping.expert_id
            else:
                expert_key = int(_EXPERT_RE.match(meta.name).group("expert"))
            return TensorGroup(
                name=meta.name,
                component=DeepSeekW2MemComponent.W2_EXPERT,
                policy=ShardPolicy.SHARDED,
                num_bytes=meta.num_bytes,
                count=1,
                expert_shard_key=expert_key,
            )
        if cls is WeightClass.ENGRAM_HOST:
            return TensorGroup(
                name=meta.name,
                component=DeepSeekW2MemComponent.ENGRAM_HOST,
                policy=ShardPolicy.HOST,
                num_bytes=meta.num_bytes,
                count=1,
            )
        # FP16 device lane. embed / lm_head get their own component for reporting.
        component = MemComponent.EMBEDDING if meta.name == "embed.weight" else MemComponent.NON_EXPERT_FP16
        return TensorGroup(
            name=meta.name,
            component=component,
            policy=ShardPolicy.SHARDED,
            num_bytes=meta.num_bytes,
            count=1,
        )

    # -- placement ------------------------------------------------------------

    def _place(self, group: TensorGroup, accountant: DeepSeekW2MemoryAccountant) -> None:
        ledger = self._ledger
        ledger.total_count += group.count
        if group.policy is ShardPolicy.HOST:
            for rank in range(self.world_size):
                accountant.rank_report(rank).add(group.component, group.num_bytes)
            ledger.host_total_bytes += group.num_bytes
            ledger.host_logical_bytes += group.num_bytes
            return
        ledger.device_logical_bytes += group.num_bytes
        # SHARDED: partition bytes across ranks exactly once.
        if self.parallel_mode == "ep" and group.expert_shard_key is not None:
            per_rank = [0] * self.world_size
            per_rank[group.expert_shard_key % self.world_size] = group.num_bytes
        else:
            per_rank = _split_even(group.num_bytes, self.world_size)
        placed = 0
        for rank, rank_bytes in enumerate(per_rank):
            if rank_bytes:
                accountant.rank_report(rank).add(group.component, rank_bytes)
            placed += rank_bytes
        if placed != group.num_bytes:
            raise DoubleInstantiationError(f"group {group.name}: placed {placed} != {group.num_bytes}")
        ledger.sharded_total_bytes += group.num_bytes
        ledger.sharded_placed_bytes += placed

    def simulate(self, tensor_metas: Iterable[TensorMeta]) -> DeepSeekW2MemoryAccountant:
        """Walk the metadata stream and return the populated accountant.

        The accountant is a :class:`DeepSeekW2MemoryAccountant` (E0.5), so the
        Engram host table is host-classified and deduped, while W2 experts and
        FP16 weights count toward the per-rank device budget.
        """
        self._ledger = DeepSeekW2Ledger()
        accountant = DeepSeekW2MemoryAccountant(world_size=self.world_size)
        # Pre-create every rank report so a rank that received no bytes still
        # participates in the imbalance check.
        for rank in range(self.world_size):
            accountant.rank_report(rank)
        working_set = 0
        for meta in tensor_metas:
            working_set = max(working_set, 1)
            group = self._group_for(meta)
            if group is None:
                continue
            self._place(group, accountant)
        self._ledger.peak_working_set_objects = working_set
        self._ledger.assert_no_double_instantiation()
        return accountant

    @property
    def ledger(self) -> DeepSeekW2Ledger:
        return self._ledger

    # -- reporting ------------------------------------------------------------

    def predicted_per_chip_bytes(self, accountant: DeepSeekW2MemoryAccountant) -> dict[int, int]:
        return {r.rank: r.device_bytes() for r in sorted(accountant.ranks.values(), key=lambda x: x.rank)}

    def predicted_report(self, tensor_metas: Iterable[TensorMeta]) -> dict:
        """Simulate and return a JSON-able per-chip byte report."""
        accountant = self.simulate(tensor_metas)
        per_chip = self.predicted_per_chip_bytes(accountant)
        max_device = max(per_chip.values()) if per_chip else 0
        return {
            "parallel_mode": self.parallel_mode,
            "world_size": self.world_size,
            "tensor_count": self._ledger.total_count,
            "excluded_count": self._ledger.excluded_count,
            "excluded_bytes": self._ledger.excluded_bytes,
            "device_aggregate_bytes": self._ledger.device_logical_bytes,
            "engram_host_bytes": accountant.host_table_bytes(),
            "per_chip_device_bytes": per_chip,
            "max_per_chip_device_bytes": max_device,
            "max_per_chip_device_gib": max_device / _BYTES_PER_GIB,
            "per_chip_device_budget_gib": PER_CHIP_DEVICE_WEIGHT_BUDGET_GIB,
            "under_budget": max_device <= PER_CHIP_DEVICE_WEIGHT_BUDGET_BYTES,
            "imbalance": accountant.imbalance(),
            "accountant": accountant.to_dict(),
        }
