# GLM-5.3-Flash W2-310p — HBM 2× (really 4–8×) bloat at TP4: root cause & fix

Author: matteius@gmail.com · Date: 2026-09-19
Scope: **CODE + DATA analysis only.** No NPU execution. Safetensors *headers* were
read read-only over SSH (8-byte length + JSON header, never tensor data). The
production server on 192.168.53.187 was not touched.

---

## TL;DR

- **On-disk expert dtype (VERIFIED from safetensors headers):** routed experts are
  stored as **2-bit packed codes in `uint8`** (`*_proj_codes`, 4 codes/byte) plus
  `float32` block scales — NOT fp8, NOT int8. The name "W2" is correct; the
  `config.json` `quant_method: "fp8"` is a leftover from the source checkpoint and
  is **misleading**.
- **Root cause (VERIFIED):** `config.json` declares
  `architectures: ["Glm5NextForConditionalGeneration"]`, which vLLM dispatches to
  the **fp8** package `vllm_ascend/models/glm5next`. Its MoE selects
  `AscendFp8BlockFusedMoEMethod`, whose `get_weight` allocates `w13_weight` as
  **`float8_e4m3fn` at full logical shape** — i.e. it **up-casts the 2-bit codes to
  1 byte/element (4×)** at allocation, then `process_weights_after_loading` would
  resolve to the model dtype **fp16 (2 bytes, 8×)**. It never gets that far: it
  OOMs while *allocating* the fp8 buffers.
- **Biggest bloat contributor (VERIFIED arithmetic):** the routed expert weights.
  Kept 2-bit-packed they are **~19 GB/rank at TP4**; materialized as fp8 they are
  **~76 GB/rank** (fp16 target would be ~152 GB/rank). The fp8 `w13` allocation is
  exactly **1.125 GiB/layer/rank**, matching the OOM's "Tried to allocate 1.13 GiB".
- **Fixable without offload? YES.** The 2-bit model shards to **~24–27 GB/rank** at
  TP4, which fits 4×44 GB with ~13 GB left for KV cache at `gpu_memory_utilization=0.90`.
  The OOM is a **wrong-loader bug**, not a genuine capacity problem.
- **BUT the fix is not a config/engine-arg flip today.** The correct 2-bit loader
  package `vllm_ascend/models/glm5next_w2` exists but is a **G3-gate scaffold**: its
  MoE→W2 swap (`_swap_moe_to_w2`, G6), KDA-eager (G4) and DSA-indexer (G5) hooks are
  **no-ops**. Routing to it today would fall back to the same fp8 MoE and OOM
  identically. The fix requires **completing G6** (wire `Glm5NextW2MoE` +
  `AscendW2DynamicFusedMoEMethod310` and the `*_codes`→`w13_codes/w2_codes` weight
  loader into the model constructor / `load_weights`).

---

## 1. Ground-truth tensor sizes (VERIFIED — safetensors headers, no data loaded)

Method: parsed each of the 19 shard headers + `model.safetensors.index.json`; bytes
computed from `data_offsets`, dtype/shape from the header JSON.

**Expert weight samples (settles fp8 vs 2-bit vs int8):**

| tensor | dtype | shape | logical shape | packing |
|---|---|---|---|---|
| `...experts.0.gate_proj_codes` | **U8** | `[2048, 1024]` | `[2048, 4096]` | 4096/1024 = **4 codes/byte = 2-bit** |
| `...experts.0.down_proj_codes` | **U8** | `[4096, 512]` | `[4096, 2048]` | 2048/512 = **4 codes/byte = 2-bit** |
| `...experts.0.down_proj_scale` | F32 | `[128, 64]` | — | per-block scale |

**Total on-disk = 97.34 GB (VERIFIED), by category × dtype:**

| Category | dtype | GB | notes |
|---|---|---:|---|
| MoE expert w13 (gate+up `_codes`) | U8 (2-bit) | **51.942** | dominant |
| MoE expert w2 (down `_codes`) | U8 (2-bit) | **25.971** | |
| MoE expert scales (`_scale`) | F32 | 1.217 | [32,32]-block |
| MLA/DSA attention (q/kv/o) | F16 | 7.382 | full-attn layers |
| KDA linear-attn (`self_attn.*`: A_log, dt_bias, k_conv, *_a/b_proj) | F16 | 4.982 | hybrid attn |
| shared_expert | F16 | 2.164 | |
| lm_head | F16 | 1.269 | |
| embedding | F16 | 1.269 | |
| router gate (+e_score bias) | F16 | 0.403 | |
| dense MLP (first_k_dense_replace=3) | F16 | 0.604 | |
| misc (eh_proj, hc_* gates, norms, bias) | F16 | ~0.14 | |
| **TOTAL** | | **97.34** | |

dtype totals: **U8 77.913 GB · F16 18.212 GB · F32 1.217 GB.**

**Ideal TP4 shard of the 2-bit model:** experts (77.9 GB U8) shard 1/4 →
**~19.5 GB/rank**; total ≈ **24–27 GB/rank** (see §4). Comfortably < 44 GB.

Geometry (VERIFIED from `config.json`): 45 layers, `first_k_dense_replace=3` →
**42 MoE layers**; hidden 4096; `moe_intermediate_size=2048`; `n_routed_experts=288`
top-8; `n_shared_experts=1`; hybrid attention (34 KDA linear-attn + 11 DSA
full-attn); `vocab_size=154880`.

---

## 2. Loader trace — the up-cast (VERIFIED from code)

The OOM traceback runs through the **fp8** MoE method:

- `vllm_ascend/quantization/methods/w8a8/fp8_block.py:288-298` —
  `AscendFp8BlockFusedMoEMethod.get_weight` allocates
  ```python
  "w13_weight": torch.empty(num_experts, 2*intermediate_size_per_partition,
                            hidden_sizes, dtype=BLOCK_FP8_WEIGHT_DTYPE)  # float8_e4m3fn
  ```
  `BLOCK_FP8_WEIGHT_DTYPE = torch.float8_e4m3fn` (`fp8_block.py:59`) = **1 byte per
  *logical* element**. For 288 experts, `2*inter/4 = 1024`, hidden 4096:
  `288 × 1024 × 4096 = 1,207,959,552 B = 1.125 GiB` — **exactly the OOM's
  "Tried to allocate 1.13 GiB"** (`w2_weight` is another 0.5625 GiB/layer/rank).
  → This is the **4× up-cast**: on disk the same data is `uint8[..., hidden//4]`
  (0.25 B/element); here it is allocated at 1 B/element.
- `fp8_block.py:318-356` — `process_weights_after_loading` → `_resolve_experts`
  (`:347-356`) allocates a **new** `torch.empty(weight.shape, dtype=self.model_dtype)`
  (fp16, **2 B/element**) and `replace_parameter`s it in. On non-950 (310P,
  `is_950()` False, `:270`) there is **no MXFP8 requantize** — steady state is the
  **fp16 (8× on-disk) resolved matrix**. Transient peak = fp8 buffer + fp16 resolved.
- **Duplication note:** the fp8 path does not keep both a quantized *and* a
  dequantized copy at steady state (`replace_parameter` frees the fp8 buffer). The
  bloat is the **up-cast**, not a duplicate buffer. Per-block scales are sized
  reasonably (`get_dynamic_quant_param`, `:300-316`). The failure is purely the fp8
  allocation being 4× the packed disk size — it OOMs before any resolve.

Adapters that drive this: `vllm_ascend/quantization/method_adapters.py:286-317`
(`AscendFusedMoEMethod.create_weights` → `get_weight`) ←
`vllm_ascend/_310p/fused_moe/fused_moe.py:143` ←
`vllm_ascend/ops/fused_moe/routed_experts.py:351`.

**The matching W2 (packed) method already exists but is not selected for GLM:**
`vllm_ascend/_310p/quantization/methods/w2_dynamic.py:148-163`
(`AscendW2DynamicFusedMoEMethod310.get_weight`) allocates
`w13_codes = uint8[E, 2*inter, hidden//4]` and `w2_codes = uint8[E, hidden, inter//4]`
— **0.25 B/element, matching disk**. It is registered as `W2A8_DYNAMIC`/`moe`.

---

## 3. Replicated FP16 modules (`modules_to_not_convert`)

`config.json` lists 1509 `modules_to_not_convert` (kept FP16). Per category (from
config regex): 46× `input_layernorm`, 46× `post_attention_layernorm`, 45× each of
`hc_attn_{base,fn,scale}` / `hc_ffn_{base,fn,scale}` (hyper-connection gates), 43×
`mlp.gate` (+ `e_score_correction_bias`), 34× each KDA `self_attn.*` (A_log, b_proj,
dt_bias, f_a/f_b/g_a/g_b/qkvbfg_a proj, k_conv), plus 48× `visual.*` (excluded —
text-only). Total FP16 on disk = 18.2 GB.

Sharding (INFERRED from the shipped `glm5next` model, `vllm_ascend/models/glm5next/model.py`):
- Attention q/kv/o and dense/shared MLP use vLLM column/row-parallel linears → **TP-sharded 1/4**.
  Shared expert is a `Glm5NextMLP(intermediate_size = moe_intermediate_size × n_shared_experts = 2048)`
  (`model.py:195-202`) built from parallel linears → **sharded, NOT replicated full**.
- Embedding + lm_head → vocab-parallel → **sharded 1/4**.
- Router gate (0.4 GB), layernorms, hc_* gates, per-head KDA state (A_log/dt_bias/
  conv) → small, **replicated per rank** (~0.6–1.0 GB/rank total, INFERRED).

None of the replicated set is a material contributor; the replicated bytes are
< ~1 GB/rank. **The bloat is entirely the expert up-cast, not TP replication.**

---

## 4. Per-chip HBM byte budget (TP4)

Experts confirmed to shard 1/4 (fp8 `w13` alloc = 1.125 GiB/layer/rank = the /4
figure). "current (fp8)" = VERIFIED arithmetic from `get_weight`. "target (W2)" =
computed from `w2_dynamic.get_weight` shapes; non-expert = INFERRED sharding.

| Category | current: fp8 path (GB/rank) | target: W2 packed path (GB/rank) |
|---|---:|---:|
| Expert w13 | 50.7 (fp8) → 101 (fp16 resolve) | 12.7 (uint8) |
| Expert w2 | 25.4 (fp8) → 51 (fp16 resolve) | 6.3 (uint8) |
| Expert scales (F32) | 0.3 | 0.3 |
| Attention (MLA/DSA + KDA), sharded | ~3.1 | ~3.1 |
| Embedding + lm_head, sharded | ~0.6 | ~0.6 |
| Dense MLP + shared expert, sharded | ~0.7 | ~0.7 |
| Router/norms/hc gates, replicated | ~0.7 | ~0.7 |
| **Weight subtotal / rank** | **~81 GB (fp8) — cannot even allocate** | **~24.4 GB** |
| Headroom under 44 GB (util 0.90 ≈ 39.6 GB) | **−41 GB (OOM)** | **~+15 GB for KV cache** |

**Observed OOM reconciliation:** the fp8 weight subtotal (~76 GB experts alone)
exceeds capacity, so allocation dies partway — the log shows ~41 GB already
allocated, 721 MiB free, failing the next 1.13 GiB `w13` allocation. Consistent.

**Single biggest contributor:** `w13` expert weights up-cast from 2-bit to fp8 —
**+38 GB/rank** (12.7 → 50.7). w13+w2 together are the entire OOM (~19 GB packed vs
~76 GB fp8).

**Fix classification (per task 4):** primarily **(a) select the W2/native packed
loader instead of fp8 (stop up-casting)**. Not (b) — replication is negligible; not
(c) — no duplicate buffer at steady state; not (d) — it fits, no offload needed.

---

## 5. Concrete fix

**The model fits in 4×44 GB HBM with room for KV cache once the experts are kept
2-bit-packed. No offload is required.** The number that would otherwise have to
change is the ~57 GB/rank of fp8 up-cast that must instead stay as ~19 GB/rank of
packed uint8.

### Why an engine-arg alone is NOT enough (gating)

The right architecture is registered:
`vllm_ascend/models/__init__.py:92-95` maps
`Glm5NextW2ForConditionalGeneration` → `glm5next_w2.model:...W2ForConditionalGeneration`,
and the W2 MoE (`glm5next_w2/moe.py:Glm5NextW2MoE`, using
`AscendW2DynamicFusedMoEMethod310`) plus the `*_codes`→`w13_codes/w2_codes` mapping
(`glm5next_w2/weight_mapping.py:154,166`) are written.

**But `glm5next_w2/model.py` is a G3 scaffold:** `AscendGlm5NextW2ForCausalLM`
subclasses the *shipped fp8* `Glm5NextForCausalLM` and its override hooks are
**no-ops** (`model.py:176-207`):
`_swap_moe_to_w2()` (**G6**), `_swap_kda_to_eager()` (G4), `_override_dsa_indexer()`
(G5) have docstring-only bodies. There is **no `load_weights` override** installing
the packed `*_codes` params. So dispatching to `Glm5NextW2ForConditionalGeneration`
today would build the same fp8 `FusedMoEFactory` MoE and OOM identically.

### The fix (CODE — required, must be tested on the box later)

Complete **G6** so the registered W2 arch actually installs the packed path:
1. In `glm5next_w2/model.py::_swap_moe_to_w2`, replace each layer's
   `Glm5NextMoE` routed path with `Glm5NextW2MoE` (`glm5next_w2/moe.py:339+`),
   which resolves `AscendW2DynamicFusedMoEMethod310` (`W2A8_DYNAMIC`/`moe`) and
   allocates `w13_codes/w2_codes` uint8 (`w2_dynamic.py:148-163`).
2. Add a `load_weights` that maps checkpoint `...experts.{E}.{gate,up,down}_proj_codes`
   → `w13_codes`/`w2_codes` and `..._scale` → `w13_scale`/`w2_scale` using
   `glm5next_w2/weight_mapping.py`, populating `layer.w2_experts` (required by
   `w2_dynamic.apply`, `w2_dynamic.py:246-251`).
3. Also complete G4 (KDA eager) and G5 (DSA indexer) so the 310P forward is
   Triton-free, or the model will fail later in the forward even if it now fits.

This is a substantive, multi-file change against a scaffold. It was **not applied
here** because it cannot be validated without running on the NPU (explicitly
out of scope, and the box is serving production).

### Launch command to test the fix (UNVERIFIED — for you to run later)

Once G6 is wired, dispatch the checkpoint to the W2 arch. Cleanest without editing
the read-only model dir is an `hf-overrides` architecture override:

```bash
# On 192.168.53.187, venv /srv/ai/venvs/fork028 — ONLY when the box is free.
VLLM_USE_V1=1 \
python -m vllm.entrypoints.openai.api_server \
  --model /srv/ai/models/GLM-5.3-Flash-W2-310p \
  --tensor-parallel-size 4 \
  --gpu-memory-utilization 0.90 \
  --dtype float16 \
  --max-model-len 8192 \
  --hf-overrides '{"architectures": ["Glm5NextW2ForConditionalGeneration"]}'
```

Expected after the fix: ~24–25 GB/rank of weights, load completes, ~13–15 GB/rank
left for KV cache. If G6 is *not* yet wired, this command OOMs exactly as today
(it falls back to the fp8 MoE) — that is the gating check.

---

## Verified vs inferred

- **VERIFIED (safetensors headers / config.json / source code):** on-disk expert
  dtype = U8 2-bit packed; 97.34 GB total & category breakdown; 42 MoE layers;
  fp8 `get_weight` allocates `float8_e4m3fn` at 1.125 GiB/layer/rank (= OOM figure);
  the W2 packed method exists (`w2_dynamic.py`) but GLM's `architectures` selects the
  fp8 `glm5next` package; the `glm5next_w2` G6/G4/G5 hooks are no-ops.
- **INFERRED (computed / not run):** exact non-expert per-rank sharded bytes
  (~5 GB/rank); shared-expert is TP-sharded (from `Glm5NextMLP` construction);
  W2-path total ≈ 24–27 GB/rank and the resulting KV headroom; the launch command.
