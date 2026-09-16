# DeepSeek V4.1 W2 / 310P adaptation seams — architecture delta review

**Scope:** READ-ONLY delta review to scope the 310P DeepSeek V4.1 (552B, 2-bit / W2 experts)
adaptation. Determines how much of the model already exists in vllm-ascend's shipped
`deepseek_v4` (910/Triton path) and the vLLM fork's `deepseek_v41` reference, and turns the
plan's green-field E2–E4 tasks into concrete "adapt X in file Y" work.

**Reviewed trees**

- vllm-ascend shipped: `vllm_ascend/models/deepseek_v4/` (`model.py`, `compressor.py`,
  `indexer.py`, `dspark.py`, `mtp.py`, `vision.py`, `vl_model.py`, `mm_preprocess.py`),
  `vllm_ascend/ops/{dsa,mla,rope_dsv4}.py`, `vllm_ascend/attention/{mla_v1,sparse_flash_mla,dsa_v1}.py`,
  `attention/context_parallel/mla_cp.py`, `vllm_ascend/models/__init__.py`.
- vLLM fork reference: `vllm/models/deepseek_v41/` (`common/engram.py`, `nvidia/{model,model_state,engram}.py`,
  `attention.py`, `sparse_mla.py`, `compressor.py`, `quant_config.py`, `__init__.py`).
- Already-built W2 kernel: `vllm_ascend/models/deepseek_v41/w2_unpack.py` (E1.2),
  `vllm_ascend/_310p/quantization/methods/w2_dynamic.py` (E1.3, registered `(W2A8_DYNAMIC, moe)`).
- Triton-free precedent: `vllm_ascend/models/qwen4_exp/` + `vllm_ascend/_310p/`.

---

## 0. Headline findings

1. **The shipped `deepseek_v4` is config-driven and already covers most of V4.1's *shape*.**
   `DeepseekV4Attention`/`DeepseekV4MoE`/`DeepseekV4DecoderLayer`/`DeepseekV4Model` read
   `compress_ratios`, `q_lora_rank`, `o_lora_rank`, `o_groups`, `num_experts_per_tok`,
   `n_routed_experts`, `moe_intermediate_size`, `dspark_*`, `use_index_cache`,
   `index_topk_pattern` off `DeepseekV2Config | DeepseekV3Config | DeepseekV4Config`. So V4.1's
   40-layer / 5120 / 384-top-6 / q_lora 1280 geometry is mostly *data*, not new code.

2. **Three genuine architecture deltas separate shipped `deepseek_v4` (V4, Fable: 256 experts /
   43 layers / hash-MoE) from V4.1 (384 / 40 / Engram):**
   - **Engram** (`engram_layer_ids=[1,14]`, ~384M-row n-gram tables). The fork has it
     (`common/engram.py`, `nvidia/engram.py`) and it is **heavily Triton**. The shipped
     `deepseek_v4` has **no Engram** — it uses `num_hash_layers` hash-MoE routing (`gate.tid2eid`),
     a different mechanism. Engram is the one genuinely-new decoder sub-block for the 310P path.
   - **`compress_ratios` = {0,1,2}** in V4.1 vs **{0,4,128}** in V4. The shipped indexer keys off
     `compress_ratio == 4` (`DeepseekV4Attention.__init__`); the fork's `sparse_mla.py` maps
     0→SWA-only, 1→C1A, 2→C2A. This changes indexer/compressor instantiation and the SWA/compressed
     KV plans.
   - **Experts are 2-bit (W2) on 310P.** The fork ships **FP8/FP4 block experts**
     (`quant_config.py`: `weight_block_size=[32,32]` MXFP8 or `[128,128]` FP8). W2 is *not* in
     either upstream — it is the 310P memory-fit path, already built as E1.2/E1.3. So the MoE
     adaptation is "swap the general `FusedMoEFactory` for the E1.3 `AscendW2DynamicFusedMoEMethod310`",
     not a port.

3. **Only one explicit Triton op sits in shipped `deepseek_v4/model.py`:** `muls_add_triton`
   (`vllm_ascend/ops/triton/mul_add.py`) computing `x*scale + y`. Trivial eager replacement.
   The rest of the shipped attention path is `torch_npu` native ops (device eager), not Triton.
   The Triton mass lives in the **fork** Engram (`common/engram.py`, `nvidia/engram.py`) — which is
   exactly the code E2.3 must replace with the Qwen host-table pattern rather than port.

4. **The plan's decision to create a new `vllm_ascend/models/deepseek_v41/` package is correct** —
   `w2_unpack.py` already lives there, and the Qwen4Exp precedent is a *sibling* Triton-free package,
   not an edit of the 910 model. The new package should **import and reuse** shipped `deepseek_v4`
   classes where they are already `torch_npu`-eager and config-driven, and only override the three
   deltas above plus the `muls_add_triton` / `FusedMoEFactory` seams.

---

## 1. Inventory

Legend — **Exists-as-is**: reuse the shipped/asc class unchanged (import).
**Adapt**: shipped code exists but needs a 310P-specific override (Triton removal, W2 wiring, or
config-ratio change). **New**: no 310P-safe implementation exists; must be written (reusing a
different-model precedent where noted).

| Component | Shipped `deepseek_v4` (asc)? | Triton-based? | Fork `deepseek_v41` delta | Verdict |
|---|---|---|---|---|
| **Config / geometry** (40L, 5120, 384 top-6, q_lora 1280, moe_inter 2304, MTP-3) | Yes — all read off `DeepseekV*Config` | No | V4.1 values differ from Fable-V4 (256/43/1024); `compress_ratios={0,1,2}` not `{0,4,128}` | **Adapt** (data + ratio-branch) |
| **MLA / attention core** (`wq_a/q_norm/wq_b/wkv/kv_norm/wo_a/wo_b`, decoupled RoPE) | Yes — `DeepseekV4Attention` + `ops/dsa.py` (`AscendDeepseekSparseAttention`) + `ops/rope_dsv4.py` | No (torch_npu native + `ComplexExpRotaryEmbedding`) | Fork uses `sparse_mla.py` FlashMLA backend; asc already has a working DSA/MLA NPU path | **Adapt** (eager/310P backend; reuse `ops/mla.py`, `attention/mla_v1.py`, `mla_cp.py` where the 910 DSA path is unavailable on 310P) |
| **Indexer / CSA2 sparse-attn selection** | Yes — `deepseek_v4/indexer.py` (`DeepseekV4Indexer`, `AscendIndexerOps`) | No, but **hard `import torch_npu`** + `npu_*` select-topk/quant ops | Fork `attention.py` uses `SparseAttnIndexer` + short-ctx Triton; V4.1 `compress_ratio∈{1,2}` (asc keys on `==4`) | **Adapt** (ratio branch + torch/`_310p` fallback for `npu_*` ops unavailable on 310P; mirror Qwen `indexer_qsa.py` fallback style) |
| **Compressor** (compressed-KV state) | Yes — `deepseek_v4/compressor.py` (`Compressor`, `AscendCompressorStateCache`) | No | Fork `compressor.py` is Triton (`@triton.jit`); asc compressor is torch/`torch_npu` | **Adapt** (ratio {1,2} plumbing; reuse asc torch compressor) |
| **Engram** (n-gram tables, layers 1/14) | **No** (asc has hash-MoE `gate.tid2eid`, a different routing mechanism) | Fork Engram is **heavily Triton** (`common/engram.py`, `nvidia/engram.py`: hash-cache, lookup, post-wkv, select-rows kernels) | Present only in fork; ~W4 quantized 384M-row host tables | **New** (reuse Qwen `ngram_embedding.py` host-table `AscendPLEEmbeddingMethod` + `ple_prefetch.py`; T4.1/T4.4 pattern) |
| **DSpark draft (MTP-3)** | Yes — `deepseek_v4/dspark.py` (`DSparkDeepseekV4ForCausalLM`, default `dspark_num_mtp_layers=3`) reuses `DeepseekV4DecoderLayer`/`DeepseekV4MoE` | Inherits `muls_add_triton` via reused MoE | Fork `nvidia/dspark.py`; V4.1 `dspark_target_layer_ids=[37,38,39]`, top-3/128 | **Adapt** (reuse asc DSpark; inherits the same MoE/Triton seam fixes) |
| **Serial MTP** | Yes — `deepseek_v4/mtp.py` (`DeepSeekV4MTP`) | Inherits reused MoE | Fork MTP-3 | **Adapt** (register for `num_nextn_predict_layers=3`; stub-wire per E4.2) |
| **Routed MoE + shared expert** | Yes — `DeepseekV4MoE` via `FusedMoEFactory` + `QuantizationConfig`; shared = `DeepseekV2MLP` | `muls_add_triton` in `forward` (shared/routed combine) | Fork uses FP8/FP4 `FusedMoE`; **W2 not upstream** | **Adapt** (swap `FusedMoEFactory` → E1.3 `AscendW2DynamicFusedMoEMethod310`; eager combine) |
| **W2 MoE kernel** (unpack + grouped QDQ) | **Built** — `deepseek_v41/w2_unpack.py` (E1.2) + `_310p/.../w2_dynamic.py` (E1.3) | No (host math + guarded `torch_npu` device path) | Neither upstream (both are FP8) | **Exists-as-is** (E1.2/E1.3 done; E3.3 just wires) |
| **Model assembly** (`*Model`, decoder stack, `*ForCausalLM`) | Yes — `DeepseekV4Model`, `AscendDeepseekV4ForCausalLM`, `DeepseekV4DecoderLayer` | `muls_add_triton` (via MoE) | Fork `nvidia/model.py` adds Engram injection between sublayers (layers 1/14) | **Adapt** (subclass/compose; add Engram-inject hook at engram layer ids) |
| **KV / model-state specs** (latent MLA KV, SWA, indexer cache) | Yes — `AscendDeepseekV4SWACache`, `AscendSlidingWindowMLASpec`, `AscendMLAAttentionSpec`, `AscendDeepseekV4IndexerCache` | No | Fork `nvidia/model_state.py`; V4.1 ratio-{0,1,2} SWA/compressed plans | **Adapt** (reuse specs; recompute per-layer plan for {0,1,2}; reuse Qwen `_310p/worker/v2` state + `kv_cache.py`) |
| **Registration** | Yes — `models/__init__.py` registers `DeepseekV4ForCausalLM`, `DeepSeekV4MTPModel`, DSpark | n/a | Fork arch name `DeepseekV41ForCausalLM` | **New** (add `DeepseekV41ForCausalLM` + `Conditional`/MTP-3 rows pointing at the new package) |
| **Vision / VL / mm_preprocess** | Yes — `deepseek_v4/{vision,vl_model,mm_preprocess}.py` | No | Fork `common/mm_preprocess.py`, `nvidia/vl_model.py` | **Out of scope for W2 text path** (ConditionalGeneration alias rejects MM at first gate per E2.1) |

---

## 2. 310P adaptation seams (concrete)

### (a) Remove Triton — `muls_add_triton` and the fork Engram kernels
- **`muls_add_triton`** is the *only* Triton import in shipped `deepseek_v4/model.py`
  (`from vllm_ascend.ops.triton.mul_add import muls_add_triton`, used at
  `model.py:425` and `:432` in `DeepseekV4MoE.forward`). It computes `x*scale + y`
  (see `ops/triton/mul_add.py`). Replace with eager `final_hidden_states * scale + shared_output`
  (or `torch.addcmul`) in the V4.1 MoE override. Nothing else in the shipped decoder/attention/indexer
  imports Triton — they are `torch_npu` native ops (device-eager), which the 310P worker already runs.
- The **fork** Engram (`common/engram.py`, `nvidia/engram.py`) is the real Triton mass
  (`_write_hash_cache_kernel`, `_hash_ids_kernel`, `_engram_lookup_kernel`,
  `_fused_engram_post_wkv_kernel`, `_engram_select_rows_kernel`). **Do not port these.** Replace with
  the Qwen host-table lookup (seam (d)).
- **Grep-gate:** the new `deepseek_v41` package must contain zero `triton` imports (mirror the Qwen4Exp
  `__init__.py` docstring gate). The shipped `ops/dsa.py`, `ops/mla.py`, `indexer.py` `torch_npu` calls
  are device ops, not Triton, but the `npu_*` selection/quant ops in `indexer.py` must be checked for
  310P availability and given a torch fallback (seam (e)).

### (b) Swap MoE quant method → E1.3 `AscendW2DynamicFusedMoEMethod310`
- Shipped `DeepseekV4MoE.__init__` builds `self.experts = FusedMoEFactory(... quant_config=quant_config ...)`
  and `self.shared_experts = DeepseekV2MLP(...)`. For 310P V4.1, the routed experts must instead resolve to
  the E1.3 method `(W2A8_DYNAMIC, moe)` via the 310P registry
  (`_310p/quantization/methods/registry.py::get_scheme_class("W2A8_DYNAMIC","moe")`).
- The E1.3 method already exposes the W8A8-parallel param surface
  (`get_weight`/`get_dynamic_quant_param`/`get_shared_expert_*`) so the E3.4 loader reuses the W8A8 hooks;
  `apply(...)` consumes router-selected `topk_ids`/`topk_weights` and carries the packed W2 bank on
  `layer.w2_experts` (+ `layer.w2_shared_expert`). The V4.1 MoE override must therefore: run the router
  (softmax→top-6→renorm, `norm_topk_prob=True`, `routed_scaling_factor=1.5`) and hand `topk_ids/weights`
  to the method, rather than delegating routing to `FusedMoEFactory`'s internal router.
- Host-math parity path is already `w2_active_moe_forward` / `moe_forward` (E1.2) — E3.3 reuses it for CPU UT.

### (c) V4.1 config deltas (authoritative table)
Confirmed from `DeepSeek-V4.1-Flash-UNCENSORED-FP8/config.json` (nested under `text_config`):

| field | V4.1 (target) | shipped-V4 (Fable) |
|---|---|---|
| `num_hidden_layers` | 40 | 43 |
| `hidden_size` | 5120 | 4096 |
| `n_routed_experts` / `num_experts_per_tok` | 384 / 6 | 256 / 6 |
| `n_shared_experts` | 1 | 1 |
| `moe_intermediate_size` | 2304 | 2048 |
| `q_lora_rank` / `o_lora_rank` | 1280 / 1024 | 1024 / 1024 |
| `head_dim` / `qk_rope_head_dim` / `num_attention_heads` / `o_groups` | 512 / 64 / 64 / 8 | 512 / 64 / 64 / 8 |
| `compress_ratios` | {0,1,2} pattern | {0,4,128} pattern |
| `scoring_func` / `norm_topk_prob` / `routed_scaling_factor` | sqrtsoftplus / True / 1.5 | same |
| `engram_layer_ids` | [1, 14] | (none; `num_hash_layers=3`) |
| `engram_num_embeddings` | [384006168, 384016682] | — |
| `engram_max_ngram_size` / `engram_vocab_size` / `engram_n_heads` / `engram_head_dim` | 4 / 16000000 / 8 / 256 | — |
| `engram_compressed_vocab_size` | 99092 | — |
| `num_nextn_predict_layers` (MTP) | 3 | 1 |
| `dspark_target_layer_ids` / `dspark_block_size` / `dspark_markov_rank` | [37,38,39] / 5 / 256 | (Fable: n/a) |
| `dspark_n_routed_experts` / `dspark_num_experts_per_tok` | 128 / 3 | — |
| experts quant (upstream) | fp8 `weight_block_size=[32,32]` | fp8 block | (310P overrides to **W2 2-bit**) |

The most load-bearing code deltas: **`compress_ratio ∈ {1,2}`** (shipped indexer instantiation gates on
`compress_ratio == 4`, `DeepseekV4Attention.__init__`; must become "indexer on ratio 1 and 2"), and
**Engram vs hash-MoE** — V4.1 has no `num_hash_layers`, so `DeepseekV4MoE.hash` must be off and Engram
layers added instead.

### (d) Engram → Qwen host-table pattern (T4.1/T4.4)
- No 310P Engram exists; the fork's is Triton. Reuse `vllm_ascend/models/qwen4_exp/ngram_embedding.py`
  (`AscendPLEEmbeddingMethod` with the dual transport: `AscendPLEPinnedHostEmbeddingMethod` UVA +
  `AscendPLESharedMmapEmbeddingMethod` `/dev/shm`, selected by `create_ple_embedding_method`) and
  `ple_prefetch.py`. The Qwen PLE table is *one* shared host FP16 copy across the 4 × 310P workers,
  never `×rank` — exactly the ~W4 384M-row Engram requirement.
- The new `deepseek_v41/engram.py` subclasses/wraps the Qwen host-table method: quantized row gather
  (dequant on gather), batched dedup + async prefetch, fail-fast host-byte accounting (reuse
  `observability/qwen38_mem_accounting.py`). The n-gram *hashing* (fork `compute_hash_multipliers`,
  `EngramLayout`, `NgramHashState`) is pure-numpy/torch host math and can be ported directly (the Triton
  `_hash_ids_kernel` is only the batched device fast-path — the torch equivalent is the reference).
- Injection point: the fork injects Engram into the residual between sublayers at
  `engram_layer_ids=[1,14]` (`nvidia/model.py` `DeepseekV4DecoderLayer.forward` `residual = self.engram(...)`).
  The V4.1 decoder override adds the same hook at those layer ids.

### (e) MLA reuse vs eager on 310P
- Shipped `deepseek_v4` attention runs the **DSA** path (`ops/dsa.py::AscendDeepseekSparseAttention` →
  `DSAAttention`, `attention/dsa_v1.py`), which is a 910/Atlas NPU sparse-MLA backend. On 310P the DSA
  backend and several `npu_*` ops in `indexer.py` (`import torch_npu`, `npu` select-topk/quant) may be
  unavailable. Two reuse tiers:
  1. **Prefer the generic MLA wrapper** `vllm_ascend/ops/mla.py`
     (`MultiHeadLatentAttentionWrapper` / `MLAModules`) + `attention/mla_v1.py` for the dense MLA math
     (q down/up q_lora 1280, kv down/up, decoupled rope `qk_rope_head_dim=64`, latent KV write/read),
     matching E3.1. `attention/context_parallel/mla_cp.py` is the CP precedent to consult.
  2. **Eager torch fallback** where a 310P `npu_*` op is missing (indexer top-k selection, per-token
     quant), mirroring the Qwen `indexer_qsa.py` / `ops/qsa_cache.py` Triton-free deterministic top-k
     and `_310p/ops/*` eager kernels. Deterministic (stable descending sort) selection is required for
     E0.4 parity.
- The RoPE (`ops/rope_dsv4.py::ComplexExpRotaryEmbedding`, `get_cos_and_sin_dsa`) is torch/`torch_npu`
  and reusable as-is subject to op availability.

### (f) Reusable-as-is files
- **`vllm_ascend/models/deepseek_v41/w2_unpack.py`** (E1.2) and
  **`vllm_ascend/_310p/quantization/methods/w2_dynamic.py`** (E1.3) — the whole W2 path. **No change.**
- **`vllm_ascend/models/qwen4_exp/moe.py`** — the T3.3 grouped W8A8 host-math *shape* that
  `w2_unpack.py` already mirrors; the parity reference for E3.3.
- **`vllm_ascend/models/qwen4_exp/ngram_embedding.py`** + **`ple_prefetch.py`** — host Engram transport (E2.3).
- **`vllm_ascend/models/qwen4_exp/{indexer_qsa,qsa,ops/qsa_cache}.py`** — Triton-free indexer/top-k precedent (E3.2).
- **`vllm_ascend/_310p/`** worker/quantization/fused_moe/ops scaffolding (registry, `w8a8_dynamic.py`
  surface, `worker/v2` state, `kv_cache` plumbing) — the 310P host lane the new package plugs into.
- **Shipped `deepseek_v4/{compressor,indexer,dspark,mtp,model}.py`** — imported and subclassed by the new
  `deepseek_v41` package for everything except the three deltas (Engram, ratio {1,2}, W2/`muls_add`).

---

## 3. Rewritten E2–E4 task list (paste into plan)

These replace the current E2.1/E2.2/E2.3/E3.1/E3.2/E3.3/E3.4/E4.1/E4.2. Each task now names the file to
adapt and whether it **reuses** (import unchanged), **adapts** (override a shipped class), or **creates**
(new, reusing a named precedent). Dependencies preserved from the plan graph.

### E2.1 [asc]: DeepseekV41 package + registration + precision policy — **ADAPT/CREATE**
- **depends_on:** []
- **do:** Create `vllm_ascend/models/deepseek_v41/{__init__,model,mla,indexer,engram,moe,mtp}.py`
  as a *thin* package that **imports** shipped `deepseek_v4` classes and overrides only the deltas.
  Add a `dtype_policy.py` (mirror `qwen4_exp/dtype_policy.py`): W2 experts / INT8 act, ~W4 Engram,
  FP16 MLA/indexer/dense/LM-head, FP32 accum — one object every module reads (no local literals).
  Register `DeepseekV41ForCausalLM` (+ `DeepseekV41ForConditionalGeneration` alias that rejects MM at
  the first gate) in `vllm_ascend/models/__init__.py`.
- **reuse:** shipped `deepseek_v4` classes, `qwen4_exp/dtype_policy.py` shape.
- **validate:** import under faked NPU platform; tiny random config constructs on meta; grep-gate
  **zero `triton`** in `deepseek_v41/`; every module reads the policy object.

### E2.2 [asc]: Model state + MLA latent-KV specs — **ADAPT**
- **depends_on:** [E2.1]
- **do:** Reuse `AscendSlidingWindowMLASpec` / `AscendMLAAttentionSpec` / `AscendDeepseekV4SWACache` /
  `AscendDeepseekV4IndexerCache` from shipped `deepseek_v4`. **Recompute the per-layer plan for
  `compress_ratios ∈ {0,1,2}`** (0→SWA-only, 1→C1A, 2→C2A per fork `sparse_mla.py`), not the V4
  `{1,4,128}` classification. Package hybrid MLA latent + indexer compressed history; per-chip byte
  table. Reuse `_310p/worker/v2/` state + `qwen4_exp/kv_cache.py` plumbing. PP=1.
- **reuse:** asc KV specs + Qwen `_310p` state; **adapt:** ratio-{0,1,2} plan.
- **validate:** specs from tiny config yield valid KVCacheConfig; 8K + long-context block math exact.

### E2.3 [asc]: Engram host lookup (~W4) — **CREATE (reuse Qwen T4.1/T4.4)**
- **depends_on:** [E2.1]
- **do:** New `deepseek_v41/engram.py`: subclass Qwen `AscendPLEEmbeddingMethod`
  (`ngram_embedding.py`) + reuse `ple_prefetch.py` for one shared ~W4 host table across 4 workers,
  quantized row gather (dequant on gather), batched dedup + async prefetch, fail-fast host accounting,
  never `×rank`. Port the fork's **torch/numpy** n-gram hashing (`EngramLayout`, `NgramHashState`,
  `compute_hash_multipliers`) — **not** the Triton kernels. Inject at `engram_layer_ids=[1,14]`.
- **reuse:** Qwen host-table dual transport; **create:** the DeepSeek n-gram hash + injection hook.
- **validate:** UT with `/dev/shm` ~W4 table: 4 procs share one copy; dequant row reads correct; host
  bytes exact; no full-table pin by default; zero Triton.

### E3.1 [asc]: MLA attention on 310P + parity — **ADAPT**
- **depends_on:** [E2.1, E0.4]
- **do:** `deepseek_v41/mla.py`: reuse shipped `DeepseekV4Attention` linear stack
  (`wq_a/q_norm/wq_b/wkv/kv_norm/wo_a/wo_b`, `ops/rope_dsv4.py`). On 310P, drive the dense MLA through
  the generic `vllm_ascend/ops/mla.py` (`MultiHeadLatentAttentionWrapper`) + `attention/mla_v1.py`
  instead of the 910 DSA backend (`ops/dsa.py`) where that backend/`npu_*` op is unavailable; eager
  torch fallback otherwise. Consult `attention/context_parallel/mla_cp.py`. q_lora 1280, decoupled rope 64.
- **reuse:** asc attention linears + RoPE; **adapt:** 310P MLA backend selection.
- **validate:** CPU parity vs E0.4 MLA reference at boundary lengths; latent-KV shape/dtype asserts.

### E3.2 [asc]: Sparse-attention indexer / CSA2 + parity — **ADAPT**
- **depends_on:** [E2.1, E0.4]
- **do:** `deepseek_v41/indexer.py`: reuse shipped `DeepseekV4Indexer` / `Compressor` structure but
  (i) instantiate the indexer on `compress_ratio ∈ {1,2}` (shipped gate is `== 4`), and (ii) give every
  `torch_npu`-only op in `AscendIndexerOps` (`select_topk`, per-token quant, cache update) a
  deterministic torch fallback for 310P, mirroring `qwen4_exp/indexer_qsa.py` / `qsa.py` /
  `ops/qsa_cache.py`. Reuse asc `compressor.py` torch state cache.
- **reuse:** asc indexer/compressor + Qwen QSA torch fallbacks; **adapt:** ratio gate + op fallbacks.
- **validate:** CPU parity vs E0.4 indexer reference; deterministic (stable-sort) selection.

### E3.3 [asc]: W2 MoE forward wiring — **ADAPT (W2 kernel already exists)**
- **depends_on:** [E1.3, E2.1]
- **do:** `deepseek_v41/moe.py`: override shipped `DeepseekV4MoE` to (i) drop hash-MoE
  (`num_hash_layers`/`gate.tid2eid` off for V4.1), (ii) run the router (softmax→top-6→renorm,
  `routed_scaling_factor=1.5`) and hand `topk_ids/topk_weights` to the E1.3
  `AscendW2DynamicFusedMoEMethod310` resolved via `get_scheme_class("W2A8_DYNAMIC","moe")` — **not**
  `FusedMoEFactory`, and (iii) **replace `muls_add_triton`** (`model.py:425/432`) with eager
  `routed*scale + shared`. Shared expert stays higher-precision `DeepseekV2MLP`. CPU path reuses
  E1.2 `w2_active_moe_forward` / `moe_forward`.
- **reuse:** E1.2/E1.3 (unchanged); **adapt:** router + combine + method swap.
- **validate:** CPU parity vs E0.4/E1.2 W2 reference (with/without shared); renorm correct; no Triton.

### E3.4 [asc]: Weight mapping + streamed W2 load — **ADAPT**
- **depends_on:** [E1.1, E2.1, E0.5]
- **do:** `deepseek_v41/weight_mapping.py` (model this on `qwen4_exp/weight_mapping.py`): map W2 expert
  code tensors + per-`[32,32]` block scales → the E1.3 `get_weight`/`get_dynamic_quant_param` layout
  (reusing the W8A8 loader hooks the method mirrors); FP16 MLA/indexer/dense/LM-head by name; Engram →
  host (E2.3). Stream by TP4 (EP4-ready) via `_310p/sharded_state_loader_310p.py`, no full-bank
  materialization; integrate E0.5 accounting; reject missing/extra/duplicate/wrong-shape/dtype.
- **reuse:** E1.3 param surface + `_310p` sharded loader + Qwen weight_mapping shape.
- **validate:** CPU sim: per-rank bytes ≤ HBM target; no double instantiation; bounded RSS; all
  rejection classes fire.

### E4.1 [asc]: Full-model assembly + dummy-weight boot — **ADAPT**
- **depends_on:** [E2.2, E2.3, E3.1, E3.2, E3.3, E0.4]
- **do:** `deepseek_v41/model.py`: compose/subclass shipped `DeepseekV4Model` /
  `AscendDeepseekV4ForCausalLM` / `DeepseekV4DecoderLayer` to assemble the 40-layer stack with the
  adapted MLA (E3.1), indexer (E3.2), W2 MoE + shared (E3.3), **two Engram layers injected at ids
  [1,14]** (E2.3), DSpark, norms, LM-head. No `FusedMoEFactory`, no `muls_add_triton`, no DSA-910
  backend on 310P. Dummy-weight CPU/meta boot exercises control flow + KV-spec materialization.
- **reuse:** shipped model assembly; **adapt:** Engram hook + component swaps.
- **validate:** tiny random model loads dummy weights, forwards, samples; KV-group report matches E2.2;
  two fixed-seed forwards identical; grep-gate no CUDA/Triton on 310P path.

### E4.2 [asc]: MTP-3 registration (DSpark) — **ADAPT**
- **depends_on:** [E2.1]
- **do:** Register the `num_nextn_predict_layers=3` MTP/DSpark class for V4.1 by reusing shipped
  `deepseek_v4/dspark.py` (`DSparkDeepseekV4ForCausalLM`, default `dspark_num_mtp_layers=3`,
  `dspark_target_layer_ids=[37,38,39]`) and/or `mtp.py` (`DeepSeekV4MTP`) — which reuse
  `DeepseekV4DecoderLayer`/`DeepseekV4MoE` and therefore inherit the E3.3 W2/`muls_add` fixes
  automatically. Add the registration rows; not wired into decode (follow-on).
- **reuse:** asc DSpark/MTP (inherits MoE seam fixes); **create:** registration rows only.
- **validate:** class registers under the V4.1 arch/MTP names; constructs on tiny config; no Triton.

---

## 4. Net effort delta vs the current plan

- **Downgraded from "port" to "adapt/reuse":** E2.2, E3.1, E3.2, E3.4, E4.1, E4.2 — all have a shipped
  `deepseek_v4` (asc) or Qwen `_310p` precedent to subclass/import. The only *new decoder sub-block* is
  **Engram (E2.3)**, and even that reuses the Qwen host-table transport and the fork's torch hashing.
- **Already done (no work):** the W2 arithmetic — E1.2 (`w2_unpack.py`) and E1.3 (`w2_dynamic.py`,
  registered `(W2A8_DYNAMIC, moe)`). E3.3 is a wiring/override task, not a kernel task.
- **The concentrated risk** is the three deltas: (1) Engram host table + injection, (2) `compress_ratios
  {0,1,2}` indexer/KV plan, (3) 310P availability of the `torch_npu` `npu_*` indexer ops (torch fallback
  needed). Everything else is import-and-override of shipped, already-`torch_npu`-eager code, with
  `muls_add_triton` the single trivial Triton removal in the shipped model.
