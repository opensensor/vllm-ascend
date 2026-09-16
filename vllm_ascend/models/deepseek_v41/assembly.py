# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Host-runnable eager assembly for the Ascend 310P DeepSeek V4.1 W2 path (E4.1).

The shipped, config-driven ``deepseek_v4`` base that
``vllm_ascend.models.deepseek_v41.model.AscendDeepseekV41ForCausalLM`` subclasses
is ``torch_npu`` / Triton bound and is **not importable on the CPU host** (no NPU,
no Triton on the 310P dev lane). So -- exactly like the Qwen4Exp T1.5 assembly
(``vllm_ascend/models/qwen4_exp/model.py``) -- this module builds a *parallel,
host-runnable eager assembly* that ties the six E-wave component modules into a
40-layer decoder stack which CONSTRUCTS and FORWARDS on CPU with dummy weights,
so the whole graph + KV-spec materialization is validated host-side without the
NPU.

What is REAL (the E-wave components, wired exactly as ``model.py``'s hooks do)
-----------------------------------------------------------------------------
* **MLA** (E3.1 ``mla.py``): ``AscendDeepseekV41MLA.from_config`` -- the eager
  absorbed-latent multi-head latent attention, with its latent-KV cache.
* **Indexer** (E3.2 ``indexer.py``): ``AscendDeepseekV41Indexer`` on the
  ratio-{1,2} layers (ratio 0 == sliding window, indexer disabled); the real
  CSA2 compress -> Lightning-Indexer score -> deterministic top-k selection.
* **W2 MoE** (E3.3 ``moe.py``): ``DeepseekV41W2MoE`` -- host router (softmax ->
  top-6 -> renorm) -> E1.3 ``AscendW2DynamicFusedMoEMethod310`` (resolved
  lazily) -> eager ``routed * 1.5 + shared`` combine.
* **Engram** (E2.3 ``engram.py``): two ``AscendDeepseekV41Engram`` sub-blocks at
  ``engram_layer_ids=[1, 14]``, fed by the real ``DeepseekEngramHasher`` and the
  single shared ~W4 host tables (``create_engram_host_tables``).
* **KV specs** (E2.2 ``kv.py``): ``build_deepseekv41_layer_plan`` /
  ``build_deepseekv41_kv_cache_groups`` -- MLA latent + per-ratio indexer caches.
* **Weight map** (E3.4 ``weight_mapping.py``): ``map_expert_tensor`` fills the
  E1.3 fused param layout in :meth:`load_weights`; Engram -> host; FP16 by name;
  rejection classes fire on a malformed artifact.

Pure-eager stand-ins (no dedicated Ascend component exists yet), each tagged
-----------------------------------------------------------------------------
* **hyper-connection multi-stream residual** -- the shipped model carries the
  residual as ``hc_mult`` parallel streams mixed by device ``npu_hc_pre_v2`` /
  ``npu_hc_post`` sinkhorn kernels (device-only). Here the streams are mixed by a
  plain mean and the block output is injected back into every stream (eager
  ``mix`` / ``combine``, mirroring the Qwen ``_GatedResidual`` stand-in). The
  Engram sub-block reads/writes the full ``[T, hc_mult, dim]`` stream, as on
  device.
* **dense MLA attend** -- the eager MLA attends densely over the latent; the
  indexer selection is computed and validated on ratio-{1,2} layers but does not
  prune the eager attend (device sparse-gather is a D-wave concern).

Every dtype reads from the authoritative :data:`ASCEND_DEEPSEEKV41_DTYPE_POLICY`.
No Triton / CUDA / ``torch_npu`` module is imported on this path: the MLA /
indexer / Engram / KV components are torch-only, and the E1.3 W2 method is
resolved lazily by :class:`DeepseekV41W2MoE` only when the MoE first runs (the
CPU host takes its import-guarded host math path).

MTP-3 (``num_nextn_predict_layers=3``) is intentionally left to E4.2 and is not
part of this assembly.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

# The eager W2 expert bank is built from the reference W2 packer -- the same
# reference the shipped E1.2 host math (``w2_unpack``) already imports, so the
# host boot exercises real 2-bit packed experts (not a float stub). The E3.4
# loader (:meth:`load_weights`) fills the *device* E1.3 fused param layout.
from tests.ut.deepseek_w2.reference.w2_moe_reference import W2Expert  # noqa: E402

# Fused-expert param layout constants (E1.1 pack contract), used to allocate the
# E1.3 method param banks in load_weights without importing the torch_npu method.
from tools.deepseek_w2.w2_format import (  # noqa: E402
    W2_BLOCK_COLS,
    W2_BLOCK_ROWS,
    W2_CODES_PER_BYTE,
)
from vllm_ascend.models.qwen4_exp.ngram_embedding import AscendPLETransport

from .dtype_policy import ASCEND_DEEPSEEKV41_DTYPE_POLICY, DeepseekV41DtypePolicy
from .engram import (
    AscendDeepseekV41Engram,
    DeepseekEngramHasher,
    EngramHashLayout,
    create_engram_host_tables,
)
from .indexer import ACTIVE_COMPRESS_RATIOS, AscendDeepseekV41Indexer, IndexerSelection
from .kv import (
    build_deepseekv41_kv_cache_groups,
    build_deepseekv41_layer_plan,
)
from .mla import AscendDeepseekV41MLA
from .moe import DeepseekV41W2MoE
from .weight_mapping import (
    DtypeMismatchError,
    ShapeMismatchError,
    WeightClass,
    classify_tensor,
    expected_expert_dtype,
    expected_expert_shape,
    map_expert_tensor,
    validate_weight_map,
)

__all__ = [
    "AscendDeepseekV41EagerDecoderLayer",
    "AscendDeepseekV41EagerModel",
    "AscendDeepseekV41EagerForCausalLM",
    "build_dummy_w2_expert_bank",
]


# ===========================================================================
# Eager math helpers (Triton-free, deterministic; dtype-safe like Qwen T1.5)
# ===========================================================================
def _rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float, compute_dtype: torch.dtype) -> torch.Tensor:
    """RMSNorm (``* weight``) accumulated in ``compute_dtype``, cast back to x."""
    orig_dtype = x.dtype
    xc = x.to(compute_dtype)
    variance = xc.square().mean(dim=-1, keepdim=True)
    normed = xc * torch.rsqrt(variance + eps)
    return (normed * weight.to(compute_dtype)).to(orig_dtype)


def _linear(x: torch.Tensor, weight: torch.Tensor, compute_dtype: torch.dtype) -> torch.Tensor:
    """Dtype-safe eager matmul: compute in ``compute_dtype`` regardless of the
    stored (fp16) parameter dtype, so fp16 weights never clash with fp32 acts."""
    return F.linear(x.to(compute_dtype), weight.to(compute_dtype))


def _seeded(shape: tuple[int, ...], seed: int, scale: float, dtype: torch.dtype) -> torch.Tensor:
    """Deterministic small init (device-agnostic), matching the E-wave pattern."""
    gen = torch.Generator().manual_seed(seed)
    return (torch.randn(*shape, generator=gen, dtype=torch.float64) * scale).to(dtype)


def _engram_table_source(num_embeddings: int, embedding_dim: int, dtype: torch.dtype) -> torch.Tensor:
    """Deterministic ~W4 host-table source: packed uint8 codes or fp32 scales."""
    gen = torch.Generator().manual_seed(2024 + embedding_dim)
    if dtype == torch.uint8:
        return torch.randint(0, 256, (num_embeddings, embedding_dim), generator=gen, dtype=torch.uint8)
    return (torch.randn(num_embeddings, embedding_dim, generator=gen, dtype=torch.float32) * 0.02).to(dtype)


def _init_engram_params(engram: AscendDeepseekV41Engram, seed: int) -> None:
    """Deterministically fill the Engram ``wkv`` / gate params (else ``torch.empty``)."""
    with torch.no_grad():
        engram.wkv.weight.copy_(_seeded(tuple(engram.wkv.weight.shape), seed, 0.05, engram.wkv.weight.dtype))
        engram.q_weight.copy_(_seeded(tuple(engram.q_weight.shape), seed + 1, 0.1, engram.q_weight.dtype))
        engram.k_weight.copy_(_seeded(tuple(engram.k_weight.shape), seed + 2, 0.1, engram.k_weight.dtype))


# ===========================================================================
# Dummy W2 expert bank (real 2-bit packing over deterministic float weights)
# ===========================================================================
def build_dummy_w2_expert_bank(
    num_experts: int,
    hidden_size: int,
    moe_intermediate_size: int,
    *,
    seed: int = 0,
    scale: float = 0.05,
) -> list[W2Expert]:
    """Build ``num_experts`` W2-packed experts from deterministic float weights.

    Each expert is a SwiGLU MLP whose gate/up ``[inter, hidden]`` and down
    ``[hidden, inter]`` weights are quantized + packed to the canonical 2-bit W2
    format (the same :class:`W2Expert` the E1.2 host math consumes). Dimensions
    must tile the ``[32, 32]`` W2 block exactly.
    """
    experts: list[W2Expert] = []
    for e in range(num_experts):
        gate = _seeded((moe_intermediate_size, hidden_size), seed + 10 * e + 1, scale, torch.float64)
        up = _seeded((moe_intermediate_size, hidden_size), seed + 10 * e + 2, scale, torch.float64)
        down = _seeded((hidden_size, moe_intermediate_size), seed + 10 * e + 3, scale, torch.float64)
        experts.append(W2Expert(gate, up, down))
    return experts


# ===========================================================================
# Decoder layer: Engram (optional) -> MLA (+ indexer) -> W2 MoE, all wired
# through the eager multi-stream residual.
# ===========================================================================
class AscendDeepseekV41EagerDecoderLayer(nn.Module):
    """One eager DeepSeek V4.1 decoder layer.

    Wires the real MLA / indexer / W2-MoE / Engram components into the
    ``hc_mult``-stream residual. The per-layer type comes from
    ``compress_ratios[layer_idx]``: ratio 0 is a dense (sliding-window) MLA layer,
    ratios 1 and 2 additionally run the indexer. Engram injects on the layers in
    ``engram_layer_ids`` (``[1, 14]``).
    """

    def __init__(
        self,
        *,
        config: Any,
        layer_idx: int,
        compress_ratio: int,
        dtype_policy: DeepseekV41DtypePolicy,
        engram: AscendDeepseekV41Engram | None,
        engram_hash_index: int | None,
        w2_experts: list[W2Expert],
        shared_expert: W2Expert,
    ) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.compress_ratio = compress_ratio
        self.dtype_policy = dtype_policy
        self.eps = float(getattr(config, "rms_norm_eps", 1e-6))
        self.main_dtype = dtype_policy.main_dtype
        self.accum_dtype = dtype_policy.accumulation_dtype
        self.router_dtype = dtype_policy.router_dtype

        hidden = int(config.hidden_size)

        # -- Engram sub-block (E2.3), only on engram_layer_ids ----------------
        self.engram = engram
        self.engram_hash_index = engram_hash_index

        # -- attention: MLA (E3.1) + optional indexer (E3.2) ------------------
        self.input_layernorm = nn.Parameter(torch.ones(hidden, dtype=self.main_dtype))
        self.mla = AscendDeepseekV41MLA.from_config(config, dtype_policy=dtype_policy)
        self.indexer: AscendDeepseekV41Indexer | None = None
        self.idx_qr_proj: nn.Parameter | None = None
        self.idx_key_proj: nn.Parameter | None = None
        if compress_ratio in ACTIVE_COMPRESS_RATIOS:
            self.indexer = AscendDeepseekV41Indexer(config, compress_ratio=compress_ratio, policy=dtype_policy)
            q_lora_rank = int(config.q_lora_rank)
            index_head_dim = self.indexer.head_dim
            # Assembly-local eager seams feeding the indexer its q-LoRA query and
            # raw index keys (the device prolog fuses these into the MLA q path).
            self.idx_qr_proj = nn.Parameter(_seeded((q_lora_rank, hidden), 7000 + layer_idx, 0.05, self.main_dtype))
            self.idx_key_proj = nn.Parameter(_seeded((index_head_dim, hidden), 8000 + layer_idx, 0.05, self.main_dtype))
        self.last_selection: IndexerSelection | None = None

        # -- MoE (E3.3): host router -> E1.3 W2 method -> eager combine --------
        self.post_attention_layernorm = nn.Parameter(torch.ones(hidden, dtype=self.main_dtype))
        num_experts = int(config.n_routed_experts)
        self.gate = nn.Parameter(_seeded((num_experts, hidden), 9000 + layer_idx, 0.05, self.main_dtype))
        self.w2_experts = w2_experts
        self.moe = DeepseekV41W2MoE.from_config(
            config,
            w2_experts=w2_experts,
            shared_expert=shared_expert.forward,
            dtype_policy=dtype_policy,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        hash_ids_all: torch.Tensor | None,
    ) -> torch.Tensor:
        """``hidden_states`` is the ``[T, hc_mult, dim]`` multi-stream residual."""
        # Engram writes the gated n-gram lookup into every stream (E2.3).
        if self.engram is not None and hash_ids_all is not None:
            layer_hash_ids = hash_ids_all[:, self.engram_hash_index, :]
            hidden_states = self.engram(hidden_states, layer_hash_ids)

        # --- attention block (eager mix -> MLA -> combine) -------------------
        block_input = hidden_states.to(self.accum_dtype).mean(dim=1).to(self.main_dtype)
        normed = _rms_norm(block_input, self.input_layernorm, self.eps, self.accum_dtype)

        if self.indexer is not None and self.indexer.enabled:
            qr = _linear(normed, self.idx_qr_proj, self.main_dtype)
            raw_keys = _linear(normed, self.idx_key_proj, self.main_dtype)
            self.last_selection = self.indexer(normed, qr, raw_keys, positions)

        attn_out = self.mla(normed, positions)
        hidden_states = hidden_states + attn_out.unsqueeze(1)

        # --- MoE block (eager mix -> W2 experts + shared -> combine) ---------
        block_input = hidden_states.to(self.accum_dtype).mean(dim=1).to(self.main_dtype)
        normed = _rms_norm(block_input, self.post_attention_layernorm, self.eps, self.accum_dtype)
        router_logits = _linear(normed, self.gate, self.router_dtype)
        moe_out = self.moe(normed, router_logits, experts=self.w2_experts)
        hidden_states = hidden_states + moe_out.to(hidden_states.dtype).unsqueeze(1)
        return hidden_states


# ===========================================================================
# Backbone
# ===========================================================================
class AscendDeepseekV41EagerModel(nn.Module):
    """Eager DeepSeek V4.1 backbone: embed -> decoder stack -> stream collapse."""

    def __init__(self, *, config: Any, dtype_policy: DeepseekV41DtypePolicy) -> None:
        super().__init__()
        self.config = config
        self.dtype_policy = dtype_policy
        self.eps = float(getattr(config, "rms_norm_eps", 1e-6))
        self.main_dtype = dtype_policy.main_dtype
        self.accum_dtype = dtype_policy.accumulation_dtype

        self.vocab_size = int(config.vocab_size)
        self.hidden_size = int(config.hidden_size)
        self.hc_mult = int(getattr(config, "hc_mult", 2))
        self.num_hidden_layers = int(config.num_hidden_layers)
        self.compress_ratios = [int(r) for r in config.compress_ratios]
        self.engram_layer_ids = tuple(config.engram_layer_ids)

        self.embed_tokens = nn.Parameter(
            _seeded((self.vocab_size, self.hidden_size), 1234, 0.05, dtype_policy.embedding_dtype)
        )

        # One shared ~W4 Engram host copy (both layers), via the E2.3 dual
        # transport. SHARED_MMAP keeps it host-only (no pinned-UVA accelerator).
        self.engram_layout = EngramHashLayout(
            layer_ids=self.engram_layer_ids,
            max_ngram_size=int(getattr(config, "engram_max_ngram_size", 2)),
            n_heads=int(getattr(config, "engram_n_heads", 2)),
            engram_vocab_size=int(config.engram_vocab_size),
            compressed_vocab_size=int(config.engram_compressed_vocab_size),
            pad_id=int(getattr(config, "engram_pad_token_id", 0)),
            head_dim=int(getattr(config, "engram_head_dim", 32)),
        )
        self.hasher = DeepseekEngramHasher(self.engram_layout)
        self.compressed_vocab_size = self.engram_layout.compressed_vocab_size
        num_engram = len(self.engram_layer_ids)
        self.engram_tables = create_engram_host_tables(
            self.engram_layout,
            head_dim=self.engram_layout.head_dim,
            dtype_policy=dtype_policy,
            transport=AscendPLETransport.SHARED_MMAP,
            shm_dir=getattr(config, "engram_shm_dir", None),
            code_sources=[_engram_table_source] * num_engram,
            scale_sources=[_engram_table_source] * num_engram,
            prefix=getattr(config, "engram_shm_prefix", "e41asm"),
        )
        engram_layer_to_hash = {layer_id: idx for idx, layer_id in enumerate(self.engram_layer_ids)}

        # One shared dummy W2 expert bank + shared expert for every MoE layer.
        num_experts = int(config.n_routed_experts)
        moe_inter = int(config.moe_intermediate_size)
        self.w2_experts = build_dummy_w2_expert_bank(num_experts, self.hidden_size, moe_inter, seed=100)
        self.shared_expert = build_dummy_w2_expert_bank(1, self.hidden_size, moe_inter, seed=555)[0]

        layers: list[AscendDeepseekV41EagerDecoderLayer] = []
        for layer_idx in range(self.num_hidden_layers):
            ratio = self.compress_ratios[layer_idx] if layer_idx < len(self.compress_ratios) else 0
            hash_index = engram_layer_to_hash.get(layer_idx)
            engram_block: AscendDeepseekV41Engram | None = None
            if hash_index is not None:
                engram_block = AscendDeepseekV41Engram(
                    config=config,
                    host_table=self.engram_tables[hash_index],
                    layer_hash_index=hash_index,
                    layout=self.engram_layout,
                    dtype_policy=dtype_policy,
                )
                _init_engram_params(engram_block, seed=6000 + layer_idx)
            layers.append(
                AscendDeepseekV41EagerDecoderLayer(
                    config=config,
                    layer_idx=layer_idx,
                    compress_ratio=ratio,
                    dtype_policy=dtype_policy,
                    engram=engram_block,
                    engram_hash_index=hash_index,
                    w2_experts=self.w2_experts,
                    shared_expert=self.shared_expert,
                )
            )
        self.layers = nn.ModuleList(layers)
        self.norm = nn.Parameter(torch.ones(self.hidden_size, dtype=self.main_dtype))
        # Layer indices that actually carry an Engram sub-block (for validation).
        self.engram_injected_layers = tuple(idx for idx, layer in enumerate(self.layers) if layer.engram is not None)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return F.embedding(input_ids.long(), self.embed_tokens.to(self.main_dtype))

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        hidden = self.embed_input_ids(input_ids)  # [T, hidden]
        # Expand to the hc_mult parallel streams ([T, H] -> [T, hc, H]).
        hidden_states = hidden.unsqueeze(1).repeat(1, self.hc_mult, 1)

        # Real SplitMix-free Engram hashing over the compressed token ids.
        compressed_ids = input_ids.long().remainder(self.compressed_vocab_size)
        hash_ids_all = self.hasher(compressed_ids)  # [T, num_engram_layers, n_hash_cols]

        for layer in self.layers:
            hidden_states = layer(hidden_states, positions, hash_ids_all)

        # Collapse the streams and final-norm to the sampled single stream.
        sample_hidden = hidden_states.to(self.accum_dtype).mean(dim=1).to(self.main_dtype)
        sample_hidden = _rms_norm(sample_hidden, self.norm, self.eps, self.accum_dtype)
        return sample_hidden

    def close(self) -> None:
        """Release the shared Engram host-table mmaps."""
        self.engram_tables.close()


# ===========================================================================
# Causal LM
# ===========================================================================
class AscendDeepseekV41EagerForCausalLM(nn.Module):
    """Host-runnable eager DeepSeek V4.1 causal LM for the dummy-weight CPU boot.

    Owns the backbone, the LM head, the KV-spec materialization (E2.2) and the
    streamed W2 weight loader (E3.4). This is the CPU twin of the device
    ``AscendDeepseekV41ForCausalLM`` (whose shipped ``deepseek_v4`` base is not
    CPU-importable); it is not registered as an architecture and is used only to
    validate the assembled graph host-side.
    """

    def __init__(self, *, config: Any, dtype_policy: DeepseekV41DtypePolicy | None = None) -> None:
        super().__init__()
        self.config = config
        self.dtype_policy = dtype_policy or ASCEND_DEEPSEEKV41_DTYPE_POLICY
        self.model = AscendDeepseekV41EagerModel(config=config, dtype_policy=self.dtype_policy)
        self.lm_head = nn.Parameter(
            _seeded((int(config.vocab_size), int(config.hidden_size)), 4321, 0.05, self.dtype_policy.lm_head_dtype)
        )
        # E1.3 fused param banks, allocated on demand by load_weights.
        self._expert_param_banks: dict[str, dict[str, torch.Tensor]] = {}
        self.load_report: dict[str, Any] = {}

    # -- forward / sampling -------------------------------------------------

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor | None = None) -> torch.Tensor:
        if positions is None:
            positions = torch.arange(input_ids.shape[0], device=input_ids.device)
        return self.model(input_ids, positions)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return _linear(hidden_states, self.lm_head, self.dtype_policy.logits_dtype)

    def sample_token(self, input_ids: torch.Tensor, positions: torch.Tensor | None = None) -> torch.Tensor:
        """Forward + greedy (argmax) sample of the last position's next token."""
        hidden = self.forward(input_ids, positions)
        logits = self.compute_logits(hidden)
        return logits[-1].argmax()

    # -- KV-cache spec materialization (E2.2) -------------------------------

    def get_kv_cache_groups(self) -> list:
        """Materialize the hybrid KV-cache groups from the E2.2 per-layer plan."""
        plan = build_deepseekv41_layer_plan(self.config, dtype_policy=self.dtype_policy)
        return build_deepseekv41_kv_cache_groups(plan)

    def kv_group_report(self) -> dict[str, Any]:
        """Semantic KV report: MLA-latent layers, indexer layers, group sizes.

        Every layer carries the MLA latent cache; ratio-{1,2} layers additionally
        carry an indexer compressed-history cache. Identical specs merge into one
        ``KVCacheGroupSpec``, so the group count per class is <= the layer count.
        """
        plan = build_deepseekv41_layer_plan(self.config, dtype_policy=self.dtype_policy)
        groups = build_deepseekv41_kv_cache_groups(plan)
        mla_latent_layers = len(plan)
        indexer_layers = sum(1 for entry in plan if entry.has_indexer)
        spec_class_layers: dict[str, int] = {}
        for group in groups:
            key = type(group.kv_cache_spec).__name__
            spec_class_layers[key] = spec_class_layers.get(key, 0) + len(group.layer_names)
        return {
            "num_layers": len(plan),
            "mla_latent_layers": mla_latent_layers,
            "indexer_layers": indexer_layers,
            "num_groups": len(groups),
            "group_block_sizes": sorted(group.kv_cache_spec.block_size for group in groups),
            "spec_class_layers": spec_class_layers,
        }

    # -- weight load (E3.4): experts -> E1.3 layout, FP16 by name, Engram -> host

    def _expert_geometry(self) -> dict[str, int]:
        """Frozen W2 expert geometry driven from the model config (E3.4 schema)."""
        config = self.config
        hidden = int(config.hidden_size)
        return {
            "hidden_size": hidden,
            "moe_intermediate_size": int(config.moe_intermediate_size),
            "num_hidden_layers": int(config.num_hidden_layers),
            "n_routed_experts": int(config.n_routed_experts),
            "num_nextn_predict_layers": int(getattr(config, "num_nextn_predict_layers", 0)),
            "dspark_n_routed_experts": int(getattr(config, "dspark_n_routed_experts", config.n_routed_experts)),
        }

    def _ensure_expert_bank(self, block: str, geometry: dict[str, int]) -> dict[str, torch.Tensor]:
        """Allocate (once) the E1.3 fused param bank for one MoE block.

        Layout mirrors ``AscendW2DynamicFusedMoEMethod310.get_weight`` /
        ``get_dynamic_quant_param`` without importing the ``torch_npu`` method:
        ``w13_codes[E, 2*inter, hidden//4]`` (uint8), ``w2_codes[E, hidden,
        inter//4]`` (uint8), and the per-``[32, 32]`` block scales.
        """
        bank = self._expert_param_banks.get(block)
        if bank is not None:
            return bank
        # Draft (MTP) blocks carry dspark_n_routed_experts; dense layers n_routed.
        num_experts = geometry["dspark_n_routed_experts"] if block.startswith("mtp.") else geometry["n_routed_experts"]
        hidden = geometry["hidden_size"]
        inter = geometry["moe_intermediate_size"]
        bank = {
            "w13_codes": torch.zeros(num_experts, 2 * inter, hidden // W2_CODES_PER_BYTE, dtype=torch.uint8),
            "w2_codes": torch.zeros(num_experts, hidden, inter // W2_CODES_PER_BYTE, dtype=torch.uint8),
            "w13_scale": torch.zeros(
                num_experts, (2 * inter) // W2_BLOCK_ROWS, hidden // W2_BLOCK_COLS, dtype=torch.float32
            ),
            "w2_scale": torch.zeros(num_experts, hidden // W2_BLOCK_ROWS, inter // W2_BLOCK_COLS, dtype=torch.float32),
        }
        self._expert_param_banks[block] = bank
        return bank

    def _place_expert_tensor(self, raw_name: str, weight: torch.Tensor, geometry: dict[str, int]) -> str | None:
        """Validate + copy one artifact expert tensor into its fused E1.3 slot.

        Rejects a dtype/shape mismatch with the E3.4 taxonomy before any copy,
        then places the source rows at ``[expert_id, row_offset : row_offset +
        rows]`` of the (possibly gate/up-fused) target param. An expert id beyond
        the frozen schema is left unplaced (returns ``None``): the end-of-pass
        :func:`validate_weight_map` rejects it as an :class:`ExtraTensorError`,
        rather than an opaque index error here.
        """
        mapping = map_expert_tensor(raw_name, geometry)
        expected_shape = expected_expert_shape(mapping.slot, mapping.kind, geometry)
        if tuple(weight.shape) != expected_shape:
            raise ShapeMismatchError(
                f"{raw_name}: shape {tuple(weight.shape)} != expected {expected_shape} "
                f"(slot={mapping.slot}, kind={mapping.kind})"
            )
        expected_dtype = torch.uint8 if mapping.kind == "codes" else torch.float32
        if weight.dtype != expected_dtype:
            raise DtypeMismatchError(
                f"{raw_name}: dtype {weight.dtype} != expected {expected_dtype} (kind={mapping.kind}); "
                f"E1.1 pack token {expected_expert_dtype(mapping.kind)!r}"
            )
        bank = self._ensure_expert_bank(mapping.block, geometry)
        target = bank[mapping.target_param]
        if mapping.expert_id >= target.shape[0]:
            return None  # extra expert: deferred to validate_weight_map
        rows = weight.shape[0]
        with torch.no_grad():
            target[mapping.expert_id, mapping.row_offset : mapping.row_offset + rows].copy_(weight)
        return f"{mapping.block}.{mapping.target_param}"

    def _fp16_named_params(self) -> dict[str, nn.Parameter]:
        """The by-name FP16 destinations (embeddings / lm-head / final norm)."""
        return {
            "embed.weight": self.model.embed_tokens,
            "head.weight": self.lm_head,
            "norm.weight": self.model.norm,
        }

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Stream a (synthetic real-shaped) W2 artifact into the assembled model.

        Three tensor namespaces are handled in one streaming pass -- no full
        384-expert bank (or the 258 GB checkpoint) is ever materialized:

        * ``W2_EXPERT`` (``.ffn.experts.``) -> mapped through the E3.4
          :func:`map_expert_tensor` into the fused ``w13_*`` / ``w2_*`` E1.3
          layout, shape/dtype-validated (rejections fire).
        * ``ENGRAM_HOST`` (the ~W4 ``embed_*`` / ``wkv_*`` rows) -> recorded as the
          single shared host copy (never a device param).
        * ``FP16`` -> loaded by name (embeddings / lm-head / norm) after shape
          check; everything else recorded as classified-but-unplaced.
        * ``EXCLUDE`` (vision tower) -> dropped (text-only deployment).

        After the pass the routed-expert coverage is validated against the frozen
        geometry (:func:`validate_weight_map`) -- missing / extra / duplicate
        expert tensors are rejected.
        """
        geometry = self._expert_geometry()
        fp16_targets = self._fp16_named_params()
        loaded: set[str] = set()
        seen_names: list[str] = []
        report = {
            "expert_tensors_placed": 0,
            "engram_host": set(),
            "fp16_loaded": set(),
            "fp16_unplaced": set(),
            "excluded": set(),
        }

        for raw_name, weight in weights:
            seen_names.append(raw_name)
            lane = classify_tensor(raw_name)
            if lane is WeightClass.EXCLUDE:
                report["excluded"].add(raw_name)
                continue
            if lane is WeightClass.W2_EXPERT:
                target = self._place_expert_tensor(raw_name, weight, geometry)
                if target is not None:
                    report["expert_tensors_placed"] += 1
                    loaded.add(target)
                continue
            if lane is WeightClass.ENGRAM_HOST:
                # ~W4 Engram rows live in the single shared host table (E2.3),
                # not a device parameter -- record the host placement.
                report["engram_host"].add(raw_name)
                loaded.add(raw_name)
                continue
            # FP16 device lane: load by name where a destination exists.
            param = fp16_targets.get(raw_name)
            if param is not None and tuple(param.shape) == tuple(weight.shape):
                with torch.no_grad():
                    param.copy_(weight.to(param.dtype))
                report["fp16_loaded"].add(raw_name)
                loaded.add(raw_name)
            else:
                report["fp16_unplaced"].add(raw_name)

        # Reject an incomplete / malformed routed-expert set.
        validate_weight_map(seen_names, geometry)
        self.load_report = report
        return loaded

    def close(self) -> None:
        self.model.close()
