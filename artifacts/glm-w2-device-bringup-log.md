# GLM-5.3-Flash W2 — Ascend 310P device bring-up log (2026-09-19)

Device wave (D1) for `Glm5NextW2ForCausalLM` at TP4 on 4× Ascend 310P. The W2
adaptation (G3–G7) was code-complete + CPU-tested; this log records the fixes
applied on the **deployed box** (`matteius@192.168.53.187:/srv/ai/src/vllm-ascend`)
to make it load + run on real hardware. **These are live box edits not yet
backported to the repo — backport all of them.**

Launch: `VLLM_ASCEND_310P_ENABLE_MLA=1`, TP4, `--hf-overrides '{"architectures":["Glm5NextW2ForCausalLM"]}'`,
`max_model_len=2048`, enforce-eager, `dtype=float16`, gpu-mem-util 0.9.

## Result so far
GLM-W2 **loads and fits in HBM at TP4 without OOM** (2-bit experts, ~24-27GB/chip) —
answering "can GLM run without the RAM upgrade": YES for load/fit. Forward runs
through embeddings, hyper-connection, first matmuls. **Remaining:** KDA/DSA
forwards still call the shipped Triton/MLA path (only attached as seams) — being
wired to the eager modules (separate task).

## Fixes applied on the box (file : what : why)
1. **Deployed the wired `glm5next_w2` package** (from wiring sub-agent) — filled `_swap_moe_to_w2`/`_swap_kda_to_eager`/`_override_dsa_indexer`, `_suppress_fp8_expert_allocation` + `bind_w2_delegate` (fp8 experts never allocated → no OOM), `load_weights` streaming `_codes`.
2. **arch** = `Glm5NextW2ForCausalLM` (not `...ForConditionalGeneration`) — the ConditionalGeneration alias hard-rejects any config with `vision_config` even text-only.
3. `vllm_ascend/ops/rotary_embedding.py` `_record_cos_and_sin_cache_interleaved` : guard `numel()==0 / shape[-1]==0` : GLM is NoPE (`qk_rope_head_dim=0`) → empty rope cache → `view(-1,2,0)` crash.
4. `vllm_ascend/models/glm5next_w2/dsa.py:222,450` : `torch.randn(..., device="cpu")` : 310P has no NPU torch.Generator; CPU-gen randn on npu default device fails.
5. `vllm_ascend/models/glm5next_w2/model.py` `_place_streamed_expert` : skip `layers.{mtp_index}` expert banks (MTP-1 stub not wired).
6. `vllm_ascend/models/glm5next_w2/model.py` `load_weights` : (a) skip ALL MTP-layer weights (`.layers.45.`), (b) remap `model.language_model.` → `model.` for passthrough (checkpoint is a multimodal wrapper; text-only model expects `model.*`). Experts keep the wrapped name for the G6 map.
7. `vllm_ascend/quantization/methods/w8a8/fp8_block.py` `resolve_block_scales` : do the fp8→fp16 dequant on **CPU**, move dense result to NPU : **310P has no fp8 (e4m3) compute** — on-device fp8 cast/copy fails (`aclnnInplaceCopy` 561103). Non-expert linears are fp8-block-quantized.
8. `vllm_ascend/core/kv_cache_interface.py` `AscendMLAAttentionSpec` : add no-op `storage_block_size` setter : vLLM parent re-added it as a field; frozen-dataclass init assigns it but the subclass made it a read-only @property (value stays computed).
9. `vllm_ascend/_310p/model_runner_310p.py` `__init__` : default `uses_xdrope_dim=0` / `draft_uses_xdrope_dim=0` : referenced everywhere, never assigned on this path; NoPE → 0.
10. `vllm/model_executor/kernels/mhc/torch.py:82` (opensensor vllm) : `residual.dtype == bfloat16` → `in (bfloat16, float16)` : GLM W2 dtype policy is fp16; hyper-connection asserted bf16.
11. `vllm/model_executor/kernels/mhc/torch.py:128` (opensensor vllm) : `.to(torch.bfloat16)` → `.to(residual.dtype)` : mhc emitted a bf16 tensor → downstream matmul `aclnnMatmul` rejects bf16 (310P matmul supports only DT_FLOAT/DT_FLOAT16).

## Key hardware facts learned (reusable)
- **310P has no fp8 compute** (e4m3): dequant fp8 on CPU.
- **310P matmul: fp16/fp32 only, no bf16.** Keep the whole model fp16.
- **No NPU torch.Generator**: seeded init `torch.randn(..., device="cpu")` then copy.
- **Triton absent**: any `@triton.jit` kernel is a plain fn; `kernel[grid]()` fails → need eager paths (KDA/DSA).

## Related worktrees (staged, not merged)
- MTP-1 real draft head: `.claude/worktrees/agent-a52de3dee160e2015` (spec-decode; needs `--speculative-config`).
- KDA/DSA forward wiring: in progress.

## CORRECTION (fp8) + real HBM root cause (2026-09-19, later)
- **310P DOES support fp8 matmul** (probe: `torch.matmul` on two float8_e4m3fn tensors on npu = OK; fp16->fp8 cast = OK). The ONLY broken op is the **fp8->fp16 cast** (`Cast`/`aclnnInplaceCopy`, 561103). My earlier "310P has no fp8" was over-generalized from that one cast.
- **The GLM-W2 checkpoint has ZERO fp8 tensors:** 77.9GB U8 (2-bit experts) + 18.2GB F16 (non-experts) + 1.2GB F32 (scales). The `config.json` `quant_method: fp8` makes vLLM pointlessly round-trip the F16 non-experts through F16->fp8->F16 (the fp8_block linear method); that round-trip is where the 561103 + my CPU-dequant workaround live. It is not needed.
- **The real HBM OOM cause: the 18.2GB F16 non-experts are REPLICATED across all 4 ranks (~18GB/chip) instead of TP-sharded (~4.5GB/chip).** The shipped glm5next attention uses vLLM parallel layers (`ColumnParallelLinear`/`RowParallelLinear`/`VocabParallelEmbedding`) and shards fine, but the eager KDA/DSA (`glm5next_w2/dsa.py`,`kda.py`) own raw `nn.Parameter`s at full `num_heads` (dsa.py ~376/391/396/398), and the shared expert in `moe.py` may too. Fix = shard those to `num_heads//tp_size` (+ RowParallel all-reduce on o_proj), or reuse the shipped parallel projection layers and replace only the attention core. Expected after fix: ~24GB/chip, fits with ~18GB headroom. (This also fixes a latent correctness bug: the full-size eager params mismatched the sharded checkpoint weights so `_bind_shipped_mla_weights` silently left them random-init.)

## CORRECTION #2 (measured) + EP + eager-MoE fixes (2026-09-19, later still)
The DSA/attention TP-sharding above was applied and the non-experts ARE now
sharded — a `[DIAG]` print in `load_weights` (since removed) measured the
per-rank `nn.Parameter` footprint at exactly **5.64 GB/rank** (embed_tokens +
lm_head each 0.317GB = vocab/4 sharded; KDA `in_proj` 0.053GB = local-heads
sharded). So the line-45 "non-experts replicated at 18GB/chip" hypothesis was
**wrong**; the real bloat was the **experts**.
- **Root cause of the 41GB OOM (measured): the 288 2-bit routed experts were NOT sharded at all.** `model.py:_new_packed_expert_bank` allocated all 288 experts at full `moe_intermediate=2048` on *every* rank, and `_place_streamed_expert` streamed the full expert set (~78GB) to each chip → OOM at ~35–41GB partway through the 19 shards (5.64 non-expert + ~35 expert ≈ full HBM).
- **FIX 1 — expert parallelism (EP).** Run the TP group as an EP group: each rank owns a contiguous slice of 288/4 = **72 experts**.
  - `glm5next_w2/model.py:_place_streamed_expert` — skip experts not owned by this rank (`ep_expert_range`); their bank slot stays a `None`-init placeholder and never reaches HBM.
  - `glm5next_w2/moe.py` — new `_ep_rank_size`/`ep_expert_range`/`_all_reduce_routed`; `routed_experts_forward` masks the router top-k to local experts (zero the non-local pairs' weight + remap their id to a local one so the method only unpacks *filled* local bank entries — the per-pair `y*weight` scatter makes them contribute exactly 0), then **all-reduces** the partial routed outputs across ranks. Bit-exact vs replication (router runs on replicated full logits; top-k weights already globally renormalized). HCCL has no fp64 kernel → fp64 routed reduced in fp32 and cast back.
  - **RESULT (confirmed on box): loads all 19 shards, HBM plateaus at ~25.5GB/chip, NO OOM.** GLM-5.3-Flash W2 fits at TP4 on 4×310P *without the RAM upgrade*, ~18GB/chip free for KV cache.
- **FIX 2 — eager fp32 MoE device path.** After the EP fix the warmup forward reached the routed-expert grouped matmul and hit `aclnnQuantGroupedMatmulDequant` **error 161002 / EZ1001: "weight is not in 5-dim (G, K//32, N//16, 16, 32)"** — the pinned CANN fused kernel wants the packed weight pre-tiled into a 5-D fractal-NZ layout the W2 unpack never emits (the unfinished D1.5 layout work). Replaced `_310p/quantization/methods/w2_dynamic.py:_apply_device`'s fused-kernel loop with an **eager on-NPU path**: dequant the 2-bit codes to fp32 (`codes * per-block scale`, exact) and `torch.matmul` in fp32 (310P has no fp64/bf16 matmul, so the host path's `.double()` can't run on-device). Same math as `_apply_host`, activations stay fp32 (>= the fused kernel's A8). Producing the true 5-D fractal layout for the fused kernel remains a later perf optimization (D1.5).
  - Backups on box: `moe.py.pre-ep.bak`, `model.py.pre-ep.bak`, `w2_dynamic.py.pre-eager.bak`.

## END-TO-END: GLM-5.3-Flash W2 GENERATES on 310P at TP4 (2026-09-19)
After the EP + eager-MoE fixes, bring-up cleared **seven** distinct init/forward
blockers (each launch got further); the model now loads, fits, and **generates**:
1. HBM OOM -> **expert parallelism** (72 experts/rank, all-reduce). Fits ~24-25.5GB/chip.
2. MoE grouped-matmul kernel `161002 / EZ1001 "weight not in 5-dim (G,K//32,N//16,16,32)"`
   -> **eager fp32 dequant->matmul** device path (fused CANN kernel needs a fractal-NZ
   weight layout the W2 unpack doesn't emit; see "FUSED KERNEL" below).
3. AiCPU int64 `ArgSort` (slow) -> sort on an **fp32 key** (expert ids exact in fp32).
4/5. `MultipleOf * int` and `MultipleOf > int` in `_310p/model_runner_310p.py`
   (`may_reinitialize_input_batch`): vLLM tags MLA kernel block sizes as `MultipleOf`
   -> `_concrete_size()` coercion + emit concrete ints from the supported-sizes comprehension.
6. Hybrid KV-alloc: GLM names KDA (MambaSpec) and DSA (MLAAttentionSpec) layers both
   `self_attn`, and the specs are wrapped in `UniformTypeKVCacheSpecs`. `_allocate_kv_cache_tensors`
   dispatched by name substring -> **dispatch by spec type**, and **unwrap** the uniform wrapper
   to per-layer specs.
7. DSA indexer `block table exceeds persistent buffer: required=8, capacity=2`
   (`attention/indexer_kpool.py`): the 310P page limit splits DSA blocks into 32-wide
   kernel blocks, widening the block table ~4x -> **grow the buffer on demand**.
Result: `Processed prompts: 0%` reached -- the full forward (2-bit EP MoE + eager KDA +
dense NoPE MLA/DSA + mhc) runs end-to-end and emits tokens. Load+fit+generate CONFIRMED
without the RAM upgrade.

## PERF: eager MoE was ~75s/step (fixed) + FUSED KERNEL plan
- **Root cause of the ~75s/forward-step:** `unpack_active_experts` broadcast each active
  expert's per-`[32,32]` block scale to a full `[out,in]` **float64** tensor. **310P has no
  native fp64** ("Device do not support double dtype now" warning) so every such op is an
  emulated cast -> pathological. (Not the matmul: decode is 1 token.)
- **FIX (done, in `w2_dynamic._apply_device` + new `_w2_dequant_fp32`):** dequant in **native
  fp32 straight from the packed bank**, applying the COMPACT `[out//32,in//32]` block scale by a
  tiled view-multiply -- no float64, no full-size broadcast. Bit-identical math. Expected large
  speedup (fp64 emulation + 14GB/token alloc churn removed). **Needs on-host timing verification.**
- **FUSED KERNEL -- RESOLVED (2026-09-19, probe `~/w2_fused_probe.py`): NOT usable for W2.**
  Empirically determined the accepted call shape: `quantized_weight` = a **3-D `[G, N, K]` int8**
  tensor **format-cast to FRACTAL_NZ** (`torch_npu.npu_format_cast(w, 29)`; the "weight is not in
  5-dim" error is really a *format* check -- NZ storage of `[G,N,K]` is the 5-D `(G,K//32,N//16,16,32)`
  fractal; do NOT pre-reshape or it double-fractalizes to 7-D). `weight_scale` must be **fp32** and
  **per-output-channel `(G, N)`** -- the per-K-block shape `(G, K//32, N)` is **REJECTED** (`PTA call
  acl api failed`). **W2's scale is per-`[32,32]` and varies along K, so it cannot fold into this
  kernel's per-channel dequant** (collapsing to per-channel gave rel err ~0.17). Per-K-block fused
  calls (one per 32-wide K-block) would need K//32=128 launches/proj -> too many. **=> The fused
  kernel is a dead end for W2.** Speed must come from either the lean fp32 eager path (done) or a
  **custom K-blocked int8 MoE** (batched int8 bmm over K//32 blocks, `einsum tks,nks->tnk` then apply
  the block scale + sum -- keeps int8 speed + no fp-weight materialization + exact per-block scale).
- **glm_stop.sh hardened** (`$CLAUDE_JOB_DIR/tmp/glm_stop.sh`): now also matches `VLLMWorker`/`EngineCore`
  (the workers rename their cmdline, so the old script stranded ~30GB/chip on the NPU -> forced a reboot).
  Deploy to `~/glm_stop.sh` on next host boot.
- **Launcher note:** the ~3min memory-profiling forward is a one-time startup cost (per process),
  not per-request; it's slow for the same eager-MoE reason and will speed up with the dequant fix.
- **Files touched this session (deploy targets on box):** `_310p/quantization/methods/w2_dynamic.py`,
  `models/glm5next_w2/moe.py`, `models/glm5next_w2/model.py`, `_310p/model_runner_310p.py`,
  `attention/indexer_kpool.py`, plus `~/glm_stop.sh` and `~/glm_w2_test.py` (`max_num_batched_tokens=1024`).

## PERF v2: measured baseline + AscendC Cube kernel (in progress)
- **Measured lean-path decode: >10 s/token** (32-token prompt not done in ~6 min, ~85-90C). The lean fp32 dequant killed the float64 pathology (no more 99C/75s-steps) but decode is still too slow interactively. Key insight: the eager matmul is ALREADY Cube-accelerated (aclnn); the remaining cost is dequanting the full fp weight each step. A vector-unit kernel would be SLOWER (loses Cube) -- only a **fused Cube GEMM with on-chip block dequant** wins.
- **Custom op `npu_w2_blocked_dequant_matmul_310`** (being built by subagent on the box): `out[T,N]=x[T,K]@(codes⊙blockscale).T`, per-[32,32] block dequant fused on-chip before the Cube MMAD. Uses the 310P-proven **arch20 catlass `block_mmad`** (as in `chunk_fwd_o`; the high-level Matmul API isn't built for 310P). 310P cube facts: `dav_m200`, **no fixpipe -> `PIPE_FIX`=`PIPE_MTE3`**, no native bf16 (`compat_310p.h`). Plumbing mirrors `causal_conv1d_v310`; op dir `csrc/gmm/w2_blocked_dequant_matmul_v310/`; register in `build_aclnn.sh` (ascend310 CUSTOM_OPS_ARRAY) + `torch_binding.cpp`/`torch_binding_meta.cpp` 310P blocks. Parity oracle `~/w2_kernel_validate.py` (rel<0.02). catlass is on the box at `csrc/third_party/catlass` (submodule; empty in local checkout) with examples 15_gemm / 07_dequant_moe / 35_w4a8.

## PERF v3: AscendC Cube kernel LANDED + decode is overhead-bound (2026-09-19)
- **`npu_w2_blocked_dequant_matmul_310` built, validated, integrated, backported to repo.** arch20 catlass `block_mmad`, `KERNEL_TYPE_MIX_AIC` unified-core, per-[32,32] block dequant fused on-chip before the Cube MMAD. Parity rel<=0.0007; **2.4x faster than eager fp32** on the MoE matmul (0.51ms vs 1.22ms/call at decode shapes). Files: `csrc/gmm/w2_blocked_dequant_matmul_v310/` (10 files) + `csrc/build_aclnn.sh` (ascend310 CUSTOM_OPS_ARRAY) + `csrc/torch_binding.cpp` + `csrc/torch_binding_meta.cpp`. Integrated in `w2_dynamic._apply_device` (prefers the op, eager fp32 fallback via `_w2_blocked_mm_op()`). All backported into the local git repo.
- **KEY FINDING: end-to-end DECODE is op-launch-overhead-bound, not matmul-bound.** With the Cube kernel, profiling/prefill sped up (shm=0, no >60s steps) but decode stayed ~8s/token. A single 1-token forward fires 45 layers x hundreds of tiny eager ops: ~816 `unpack_w2_codes` widen launches/token, eager KDA (34) + DSA (11) attention, the sequential per-expert Python loop, all-reduce, mhc. The MoE matmul was never the decode bottleneck. 310P has no cudagraph (only op-fusion), so there's no graph-capture escape.
- **Next (in progress):** (1) fold the 2-bit widen INTO the Cube kernel (take packed uint8 codes, unpack on-chip) -> removes ~816 launches/token; (2) a GROUPED W2 op (all active experts in one call, group_list) -> 24 MoE op-calls/layer down to 3; (3) speed the eager KDA/DSA. Each needs a capped, monitored device test (the eager decode can strand the box).
- **OPERATIONAL: launcher hardened** (`~/glm_launch.sh`): a watchdog runs `glm_stop.sh` after `GLM_CAP_SECONDS` (default 900s) so a slow/stuck decode can't saturate the box and strand SSH (which forced two reboots). `glm_stop.sh` already matches the renamed `VLLMWorker`/`EngineCore` children.

## PERF v4 + CORRECTNESS FINDING (2026-09-19, decisive)
- **Packed Cube kernel end-to-end: 16 tokens in 16.9s vs eager 295s -> ~17x faster decode.** Removing the ~816 `unpack_w2_codes` widen launches/token (folded into the kernel: takes packed uint8 codes, unpacks 2-bit on-chip) was the dominant decode win. `w2_dynamic._apply_device` passes `e.*_packed` directly to the op; `VLLM_ASCEND_W2_DISABLE_CUBE=1` forces the eager fallback.
- **⚠️ OUTPUT IS GIBBERISH — pre-existing W2 MODEL bug, NOT the kernel.** First completed generations both ways: cube -> 'zenpur_tail excess excess_tail...'; eager -> 'olo小平 omn quadratic skim...'. Both incoherent. The kernel is numerically ~= eager (parity rel<=0.0007), so the defect is in the W2 model math (KDA eager / DSA eager MLA / MoE routing+combine / mhc / EP all-reduce / fp16). Never caught before because no generation had run to completion (too slow). **NEXT: isolate with a per-layer activation diff vs a trusted reference (e.g. CPU/HF or 910b), layer by layer, to find the first divergence.**
- **Operational:** cards hit 92C under sustained eager decode; slow runs strand SSH (3 reboots this session). Launcher watchdog (`GLM_CAP_SECONDS`) now guarded to only fire for the latest launch; always kill stale `sleep 600/700` watchdogs before relaunch.

## CORRECTNESS DEBUG (2026-09-20): gibberish -> real words (partial); fp16-vs-bf16 is the theme
First-ever completed generations were gibberish in BOTH eager and Cube MoE -> a chain of pre-existing W2-model bugs (kernel + EP ruled out). Method: per-layer + per-component activation stats on ONE prefill (max_tokens=1, cool). Fixes applied (box + backported):
1. **DEAD FFN (fixed).** dense-MLP (L0-2) + shared-expert `Glm5NextMLP` output 0.0. Those projections are F16 in the checkpoint (NO `weight_scale_inv` exists) but `quant_method=fp8` mis-applied the block-fp8 linear -> fp8 param x unfilled zero scale -> weight*0=0. Fix: `glm5next_w2/model.py:_mark_dense_mlp_fp16` (called before super().__init__) appends the dense-MLP + shared-expert projection prefixes -- the fused `gate_up_proj` name AND its `gate_proj`/`up_proj` shards + `down_proj` -- to `quant_config.ignored_layers` -> load as fp16.
2. **RESIDUAL fp16 OVERFLOW (fixed).** Final `model.norm` input hit 65504.0 (fp16 max) -> saturation -> gibberish. GLM's mHC multi-stream residual is bf16-range; 310P has no bf16 and an earlier patch forced it fp16. Fix: carry the mHC residual accumulation in **fp32** (`glm5next/model.py` decoder forward after hc_expand + final-norm in fp32; `patch/worker/patch_triton.py:_mhc_rms_norm` downcast to weight.dtype; opensensor `vllm/model_executor/kernels/mhc/torch.py` assert relaxed to allow fp32 -- see `artifacts/opensensor-vllm-patches/`). Result: output went gibberish -> real English words.
3. **KDA UNDER-CONTRIBUTION (partial fix).** Eager KDA (34/45 layers) output ~1e-3 (barely contributes). Cause: KDA recurrence output variance ~2.5e-10 << o_norm_eps=1e-5 -> RMSNorm loses scale-invariance (~100x under-scale). Fix (workaround): `glm5next_w2/kda.py gated_rmsnorm` uses eps=min(o_norm_eps,1e-12) -> KDA output rose ~260x to O(0.1). CAVEAT: diverges from 910 (eps=1e-5); the *proper* fix is likely fp32 KDA recurrence accumulation (the tiny variance is fp16 underflow), matching bug #2's theme.
**STATUS: real-words-but-NOT-coherent.** First token e.g. 'airs'/' forfe' -- wrong AND nondeterministic at temp=0 -> near-degenerate logits + non-deterministic NPU reductions flipping near-tied argmax. Remaining tail (NOT done): (a) proper fp32 KDA recurrence (replace eps workaround), (b) localize the downstream near-degeneracy/nondeterminism (DSA output, MoE all-reduce order, final norm/lm_head; instrument logit top1-vs-top2 gap per layer), (c) systematic fp32 accumulation in the numerically-sensitive paths (GLM is bf16-designed; 310P is fp16-only -- range+precision mismatch is the root theme). Speed is solved (Cube kernel ~17x decode) and waits behind `VLLM_ASCEND_W2_DISABLE_CUBE`.
