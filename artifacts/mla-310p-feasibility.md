# MLA on Ascend 310P — Feasibility & Bring-up Draft (GLM-5.3-Flash)

Author: matteius@gmail.com · Date: 2026-09-19
Scope: analysis + code draft only. **No NPU execution was performed.** All edits
are uncommitted in the local repo `/run/media/matteius/20TB-drive/vllm-ascend`.

Legend: **[verified in code]** = read directly in a file (cited `file:line`).
**[inferred]** = reasoned from code/config but not executed on hardware.

---

## 0. Repo divergence note (ground truth)

- Local HEAD `c55ad3a6b` is **strictly ahead** of deployed HEAD
  `0cc120dd0` — deployed HEAD is an ancestor of local (`git merge-base --is-ancestor` = yes).
  Local = deployed GLM work (G3–G7) **plus** extra `system_one` commits (calibration/serving)
  that do not touch the MLA files. So the MLA source is identical between local and deployed. **[verified in code]**
- Deployed working tree has 4 uncommitted edits (`CMakeLists.txt`,
  `attention/utils.py`, `compilation/compiler_interface.py`,
  `patch/worker/patch_mamba_utils.py`) — all fork-drift shims, **none MLA-related**. **[verified in code]**
- Conclusion: the local repo is a safe place to draft; MLA logic matches the box.

---

## 1. Verdict

**Feasible with caveats — likely feasible *now* if `npu_fused_infer_attention_score`
(FIA) runs on ascend310p1; otherwise feasible with one new AscendC paged-latent-MQA
kernel.** MLA is *not* fundamentally blocked on 310P: it is blocked by (a) three hard
config-time guards and (b) missing backend wiring. The heavy optimized kernels
(`mla_preprocess`, `npu_mla_prolog_v3`/MlaPrologV3) are **not required** — they are an
optimization the fork already bypasses on 310P.

The single decisive unknown that only NPU testing can resolve: **does the fused
attention core (`npu_fused_infer_attention_score[_v2]`) execute on ascend310p1?**
Everything else in the MLA path is either a plain matmul/bmm, a generic torch_npu op,
or already-working 310P infrastructure.

Why this model is unusually tractable:
- GLM-5.3-Flash is **NoPE MLA** (`qk_rope_head_dim=0`, `mla_use_nope=True`) — no
  decoupled-rope kernels needed on the attention path. **[verified in code]**
- Only **11 of 45 layers** are attention (`deepseek_sparse_attention`); the other 34
  are `linear_attention` (KDA/GDN) which **already work on 310P**. **[verified in code]**
- The sparse indexer runs in **`full` mode** (`indexer_types` all `"full"`), so no
  top-k sparsification/SFA kernel is required. **[verified in code]**
- MoE (288 experts, W2) and MTP-1 are already done (G6/G7). **[verified in code]**

---

## 2. Target model shape (from `/srv/ai/models/GLM-5.3-Flash-W2-310p/config.json`) [verified in code]

| Field | Value |
|---|---|
| arch / model_type | `Glm5NextForConditionalGeneration` / `glm5_next_text` |
| num_hidden_layers | 45 |
| layer_types | 34 × `linear_attention` (KDA), 11 × `deepseek_sparse_attention` at idx 3,7,…,43 |
| mlp_layer_types | 42 × `sparse` (MoE), 3 × `dense` (`first_k_dense_replace=3`) |
| MoE | 288 routed + 1 shared, top-8 |
| kv_lora_rank | 512 |
| q_lora_rank | 1536 |
| qk_nope_head_dim | 256 |
| **qk_rope_head_dim** | **0  (NoPE)** |
| qk_head_dim | 256 |
| v_head_dim | 256 |
| num_attention_heads / num_key_value_heads | 64 / 64 |
| mla_use_nope | True |
| attention_bias | False |
| indexer_types | all `"full"` (index_topk=2048, index_kpool=4) |
| num_nextn_predict_layers (MTP) | 1 |

MLA latent-cache head size = `kv_lora_rank + qk_rope_head_dim` = **512**.
Prefill (materialized) head size = `qk_nope_head_dim` = `v_head_dim` = **256**.

---

## 3. Where MLA is blocked (the three guards) [verified in code]

1. `vllm_ascend/patch/platform/patch_mamba_config_310.py:42` (config time, fires first)
   `if model_config.use_mla: raise RuntimeError("MLA is not supported on 310P currently.")`
2. `vllm_ascend/_310p/worker/v2/model_runner.py:87` (MRv2 `_validate_config`)
   `raise NotImplementedError("MLA is not supported by model runner v2 on 310P.")`
3. `vllm_ascend/_310p/model_runner_310p.py:709` (MRv1 KV-cache init)
   `raise ValueError("MLAAttention is not supported for 310P.")`
   (Note a sibling guard at `:704` blocks `use_sparse`; **not triggered for GLM** — see §5.)

Backend selection: `vllm_ascend/platform.py:227 get_attn_backend_cls`. For the 310P
COMPATIBILITY family it uses `compatibility_backend_map` which only defined
`(False,False)` and left `# (True, False): "...AscendMLABackend310"` as a TODO
placeholder (`platform.py:258`). **[verified in code]** That placeholder is exactly the
intended shape of the fix.

---

## 4. Runtime op requirements for MLA on Ascend

### 4a. The optimized front-end is NOT available on 310P — and NOT needed [verified in code]
- `csrc/torch_binding.cpp`: `mla_preprocess` (def `:2857`) sits inside
  `#ifdef VLLM_ENABLE_ATB_AND_DIRECT_KERNELS` which is inside the **`#else`
  (non-310P) branch** of `#ifdef ASCEND_PLATFORM_310P` (`:2721 #ifdef`, `:2778 #else`).
  `npu_mla_prolog_v3` (def `:2880`) is also in that `#else` branch and its own comment
  says the underlying aclnn op is **950-only**. => Neither custom op is compiled for
  ascend310p1.
- This does not matter: `enabling_mlapo()` (`attention/utils.py:555`) returns **False**
  on 310P (no `UNRESTRICTED_MLAPO` capability in the 310P hardware profile;
  KV-transfer decode is disabled on 310P). So `AscendMLAImpl.forward` (`mla_v1.py:2076`)
  takes the **decomposed** `_mla_preprocess` branch, not the `mla_preprocess_only_decode`
  branch. **[verified in code]**

### 4b. The decomposed front-end uses only portable/generic ops [verified in code]
`_mla_preprocess` → `mla_preprocess_decode`/`mla_preprocess_prefill`
(`mla_v1.py:1909-1966`) use:
- projection matmuls `q_proj`, `kv_b_proj` (W2-quantized — 310P quant methods exist),
- weight-absorption bmm: `_q_proj_and_k_up_proj` (`torch.bmm`, `mla_v1.py:996`) and
  `_v_up_proj` (`torch_npu.npu_transpose_batchmatmul`, `mla_v1.py:972`),
- `rope_single` (`mla_v1.py:1550`, uses `npu_interleave_rope`) — **no-op for NoPE**,
- `exec_kv_decode`/`exec_kv_prefill` (`mla_v1.py:1430/1468`) which write the latent KV
  cache via `torch_npu.npu_kv_rmsnorm_rope_cache`.

### 4c. The attention core [verified in code]
- `_forward_prefill` (`mla_v1.py:1360`) → `torch_npu.npu_fused_infer_attention_score` (FIA).
  For NoPE it already branches to `query,key = q_nope,k_nope` with no rope concat
  (`mla_v1.py:1355`).
- `_forward_decode` (`mla_v1.py:1566`) → `torch_npu.npu_fused_infer_attention_score_v2`,
  with explicit `qk_rope_head_dim == 0` handling throughout (`mla_v1.py:1590,1601,1610`).

### 4d. torch_npu availability on the box (deployed torch_npu 2.13.0rc1) [verified in code]
Grep of the venv's torch_npu shows the Python API entries exist for:
`npu_fused_infer_attention_score`(+`_v2`), `npu_kv_rmsnorm_rope_cache`,
`npu_interleave_rope`, `npu_transpose_batchmatmul`, `npu_mla_prolog_v3`,
`_npu_flash_attention`, `_npu_paged_attention`(+`_splitfuse`).
**API presence ≠ ascend310p1 aclnn support** — the latter is the untestable-here gap.

### Summary table

| Op | Needed for GLM NoPE MLA | On 310P? |
|---|---|---|
| `mla_preprocess` (csrc) | No (mlapo off) | **No** (non-310P `#else` branch) [verified] |
| `npu_mla_prolog_v3` (csrc/torch_npu) | No | **No** (950-only) [verified] |
| projection matmuls / `torch.bmm` / `npu_transpose_batchmatmul` | Yes | Yes (generic) [inferred] |
| `npu_interleave_rope` (rope_single) | No (NoPE ⇒ no-op) | n/a on attn path [verified NoPE] |
| `npu_kv_rmsnorm_rope_cache` (KV write) | Yes | API present; **runtime unverified** [inferred] |
| **`npu_fused_infer_attention_score[_v2]`** (FIA) | **Yes (decisive)** | API present; **runtime unverified** [inferred] |
| `_npu_flash_attention` / `_npu_paged_attention` | Only for naive fallback | **Yes, known-working** (dense 310P backend) [verified] |

---

## 5. DeepSeek sparse attention / indexer assessment [verified in code]

- GLM has `index_kpool` in config ⇒ `model_uses_kpool_indexer()` = True
  (`utils.py:121`) ⇒ `enable_sfa()` returns **False** (`utils.py:1628-1636`) ⇒
  `self.use_sparse = False` (`worker/model_runner_v1.py:416`).
- Therefore GLM's attention key in `get_attn_backend_cls` is `(use_mla=True,
  use_sparse=False)` — it uses the **MLA backend, not the SFA/DSA backend**
  (`platform.py:231` comment: "index_kpool GLM is not DeepSeek SFA; keep MLA backend").
- Consequence: the `dsa_v1.py`/`sfa_v1.py`/`sparse_flash_mla.py` backends and the
  DSA-only guard (`model_runner_310p.py:704`) are **not on GLM's path**. The
  `Failed to import vllm._deepselect_C` log is benign for GLM.
- **Indexer is still model-side code.** The `deepseek_sparse_attention` layers construct
  a lightning indexer (`attention/indexer.py`) that computes top-k with generic torch_npu
  ops (`npu_quant_matmul`, `npu_rotary_mul`, `npu_dynamic_quant`, `npu_scatter_nd_update_`
  — `indexer.py:230-433`). With `indexer_types="full"` + `index_topk=2048`, for served
  contexts ≤ 2048 tokens the top-k selects the full context ⇒ **behaves as dense MLA**.
  This is the natural dense fallback. **Risk:** the indexer forward still executes and its
  ops must run on ascend310p1; validate separately (it is independent of the 3 guards).

---

## 6. Minimal-change plan (implemented as a gated fall-through)

Design principle: **never remove the guards blindly**; gate them behind a default-off
bring-up flag so production 310P behavior is byte-for-byte unchanged, and wire the
already-anticipated `AscendMLABackend310` placeholder.

Flag: `VLLM_ASCEND_310P_ENABLE_MLA` (default `0`), added to `vllm_ascend/envs.py`
per AGENTS.md (centralized `env_variables`).

---

## 7. Files edited / added (all uncommitted)

1. **`vllm_ascend/envs.py`** — added `VLLM_ASCEND_310P_ENABLE_MLA` (default 0) with
   documentation of why MLA is off by default on 310P.
2. **`vllm_ascend/patch/platform/patch_mamba_config_310.py`** (guard 1) — raise only when
   the flag is off; when on, fall through so `FullAttentionSpec` uses the MLA head size.
   Comment notes the `block_size*head_size<=16384` constraint (⇒ block_size ≤ 32 for the
   512-wide latent head).
3. **`vllm_ascend/_310p/worker/v2/model_runner.py`** (guard 2) — MRv2 raise gated by the
   flag (MLA bring-up is wired through MRv1 for now).
4. **`vllm_ascend/_310p/model_runner_310p.py`** (guard 3) — MRv1 KV-cache-init raise gated
   by the flag; logs a warning when the experimental path is taken.
5. **`vllm_ascend/platform.py`** — `get_attn_backend_cls`: when the flag is on, register
   `(True, False) -> vllm_ascend._310p.attention.mla_v1_310.AscendMLABackend310` in the
   310P `compatibility_backend_map` (the fork's own TODO placeholder).
6. **`vllm_ascend/_310p/attention/mla_v1_310.py`** (new) — `AscendMLABackend310` +
   `AscendMLAImpl310`:
   - Subclasses the standard MLA backend/impl; **reuses the decomposed NoPE front-end**.
   - Enforces invariants (`enable_mlapo==False`, `qk_rope_head_dim==0`) so a run can never
     silently hit a 310P-absent op.
   - `get_supported_kernel_block_sizes()` = `[32, 16]` (the 512-head tiling constraint).
   - Attention core: **inherits the FIA path by default** (to be tried first on hardware),
     and provides `_forward_prefill_naive`/`_forward_decode_naive` as **precise
     kernel-spec stubs** (raise `NotImplementedError` with the exact contract) for the
     case where FIA is unsupported on ascend310p1.

All six files pass `python -m py_compile`. **[verified locally]**

---

## 8. Remaining kernel / testing work

Ordered by likelihood of being the blocker:

1. **Validate FIA on ascend310p1.** Run one MLA layer decode+prefill through the inherited
   FIA path. If it works, GLM MLA likely runs end-to-end with the current draft (modulo
   block-size and KV-cache-shape tuning). If it raises "not supported on this SoC", go to (3).
2. **Validate `npu_kv_rmsnorm_rope_cache` on 310P** (the latent KV-cache write in
   `exec_kv_*`). Same all-or-nothing risk as FIA. If unsupported, replace with explicit
   RMSNorm + a scatter into the paged latent cache (portable ops).
3. **If FIA is unsupported: implement the naive core.**
   - Prefill (`_forward_prefill_naive`): materialized MHA, head_size 256 via
     `torch_npu._npu_flash_attention` (as the dense 310P backend does). Requires wiring the
     MLA metadata builder to emit the packed seq_len/mask layout that op expects. **Verify**
     `_npu_flash_attention` accepts head_size 256 on 310P (e8f7b2e3f relaxed the paged path;
     confirm the non-paged flash op).
   - Decode (`_forward_decode_naive`): **paged latent MQA**, head_size 512, num_kv_heads 1,
     block_size ≤ 32. If `_npu_paged_attention` does not accept head_size 512 on 310P, a
     **new AscendC kernel** is required with contract:
     `out[t,h,:512] = softmax(scale · q[t,h,:512] · K_ctx^T) · K_ctx`, K_ctx = latent
     vectors gathered by the block table of request(t), causal over `context_lens`, for
     h∈[0,64). Output is latent (512); `_v_up_proj` (inherited) projects to v_head_dim 256.
4. **Block-size resolution.** Confirm the KV-cache manager selects block_size ∈ {32,16} for
   the MLA latent cache and that the mamba/attention page-size reconciliation in
   `patch_mamba_config_310.py` produces a consistent size across the hybrid (KDA + MLA) model.
5. **Indexer forward on 310P** (§5) — validate the lightning-indexer torch_npu ops run;
   with `indexer_types="full"` it should reduce to dense selection.
6. **Numerical parity** vs a 910B reference (or CPU) for a single MLA layer, then full model.
7. Unit tests in `tests/ut/` for `AscendMLABackend310` selection + guard gating (per AGENTS.md).

---

## 9. Step-by-step NPU test plan (for the main session — coordinate to avoid the running benchmark)

> Do NOT run while the throughput benchmark holds the NPUs. Killing NPU processes mid-op
> wedges the chips (physical power cycle required). Wait for the box to be idle.

Environment:
```bash
source /srv/ai/bin/ascend-env.sh
source /srv/ai/venvs/fork028/bin/activate
export SOC_VERSION=ascend310p1
cd /srv/ai/src/vllm-ascend
```

Step 0 — sync the draft to the box (the edits were made in the local repo). Either
`git fetch`/apply the local changes onto the deployed tree, or re-apply the 6 edits from §7.
Rebuild is **only** needed if csrc changed — **it did not**, so no recompile is required for
this draft (pure-Python change).

Step 1 — guards no longer fire at config time (fast, CPU-ish init):
```bash
VLLM_ASCEND_310P_ENABLE_MLA=1 \
python -c "from vllm import LLM; LLM(model='/srv/ai/models/GLM-5.3-Flash-W2-310p', \
  tensor_parallel_size=4, trust_remote_code=True, max_model_len=2048, \
  gpu_memory_utilization=0.9, enforce_eager=True)"
```
Expect: passes the three MLA guards, reaches KV-cache allocation, selects
`AscendMLABackend310`. Watch for the first op that raises (this pinpoints FIA vs
kv_rmsnorm_rope_cache vs block-size).

Step 2 — single short generation, eager, TP4:
```bash
VLLM_ASCEND_310P_ENABLE_MLA=1 \
python examples/offline_inference/... \
  --model /srv/ai/models/GLM-5.3-Flash-W2-310p --tensor-parallel-size 4 \
  --max-model-len 2048 --enforce-eager --max-num-seqs 1 \
  --prompt "Hello" --max-tokens 8
```
Expect: coherent tokens ⇒ prefill+decode MLA core works on 310P.

Step 3 — if Step 1/2 raises on FIA or kv_rmsnorm_rope_cache, capture the exact op name +
SoC error and report back; that selects the naive-core work in §8(3).

Step 4 — parity: compare logits/first-token against a 910B run (or reduce to a tiny config)
for the 11 MLA layers.

Step 5 — only after eager parity: try graph mode and larger `max-num-seqs`, then longer
contexts (>2048) to exercise the indexer's actual top-k path.

---

## 10. Honest limitations of this draft

- The attention **core** is inherited (FIA); the naive fallback is a **spec stub**, not a
  tested implementation — deliberately, because correct packed-layout/paged-latent tensor
  code cannot be validated without the NPU and would risk silent numerical errors.
- FIA and `npu_kv_rmsnorm_rope_cache` support on ascend310p1 is **[inferred]** from API
  presence only; it is the primary thing NPU testing must resolve.
- Block-size/KV-cache-shape interplay for the 512-wide latent head under the hybrid
  (KDA+MLA) page-size reconciliation is analyzed but **not executed**.
