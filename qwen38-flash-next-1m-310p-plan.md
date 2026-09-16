# Plan: Qwen3.8-Flash-Next (Qwen4Exp) 1M Context on Four Ascend 310P Chips

**Generated**: 2026-09-15
**PRD**: `docs/source/developer_guide/Design_Documents/qwen38_flash_next_1m_310p_prd.md`
**Scope**: Critical path to G6 (one 1,048,576-token request) — WP0–WP10. WP11/WP12 are stubbed follow-on gates.
**Constraint from user**: 310P target host NOT yet available. All code + host-side unit/parity tests execute first. All device work is a final serialized hardware wave (D1–D9). No task before D1 may require an NPU.

## Overview

Deliver an Ascend 310P execution path for Qwen4Exp (36 GDN + 12 QSA layers, PLE n-gram host embedding, 512-expert W8A8 MoE, PP=1) on 4×310P TP4 eager, reaching a validated 1M-token single request. Strategy: reuse in-tree vLLM `qwen4_exp` common code (hyperconnection, QSA cache/specs, PLE embedding ABC), implement Ascend components out-of-tree in `vllm_ascend/models/qwen4_exp/` (glm5next / kimi_k3 precedent), register via `ModelRegistry.register_model` overriding the in-tree CUDA/AMD dispatch. First bring up a correct eager-torch fallback for every layer type (slow but right), then replace each component with native 310P ops behind CPU parity unit tests against a shared eager-reference harness. The 1M cache layout decision (Candidate A = C8 QSA cache, Candidate B = QSA-aware DCP4) is prototyped on host and decided with measured allocations at 128K on hardware before the 262K gate.

## Key codebase facts (verified)

- vLLM fork: `/run/media/matteius/20TB-drive/vllm` (0.22.0-based). Model at `vllm/models/qwen4_exp/` with `__init__.py` lazy `__getattr__` dispatch: ROCm→`amd.*`, everything else (incl. NPU!)→`nvidia.*`; only XPU/TPU raise. Arch names: `Qwen4ExpForCausalLM`, `Qwen4ExpForConditionalGeneration`, `Qwen4ExpMTP` (`vllm/model_executor/models/registry.py:114,602,697`).
- Shared code: `common/hyperconnection.py` (GatedResidual), `common/ple.py` (`PLEVocabParallelEmbedding`), `common/qsa_cache.py` (Triton-guarded, has torch fallback `_build_qsa_metadata_torch`; `QSAKeyStateCache`→`CircularBufferSpec`, `QSACompressedKeyCache`→`MLAAttentionSpec(tokens_per_state=compress_ratio)`, backend name `QWEN4_EXP_EXP_QSA_STATE`, dtypes incl. `fp8_e4m3`).
- CUDA impl to port from: `nvidia/model.py` (layers; GDN = in-tree `vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn.QwenGatedDeltaNetAttention`; MoE = `Qwen3NextSparseMoeBlock` subclass), `nvidia/ngram_embedding.py` (ABC `Qwen4ExpPLEEmbedding`, `Qwen4ExpPLEPinnedHostEmbedding` uses UVA/`get_accelerator_view_from_cpu_tensor`, `Qwen4ExpNGramEmbedding`), `nvidia/ple_layer.py`, `nvidia/qsa.py`, `nvidia/indexer_qsa.py`, `nvidia/ops/*` (Triton), `nvidia/model_state.py` (`Qwen4ExpModelState(MambaHybridModelState)`, v2 worker; PP=1 enforced; `Qwen4ExpForCausalLM.get_model_state_cls` at `nvidia/model.py:687`).
- Config plumbing: `vllm/transformers_utils/configs/qwen4_exp.py`, `vllm/model_executor/models/config.py:858-916` (hybrid cache contract, PLE PP=1, dual-batch-overlap banned, MTP-only spec decode), `vllm/config/engram.py` (`EngramConfig`: `cpu_offload` via `VLLM_PLE_CPU_OFFLOAD` pinned-host UVA lookup, `dp_shared_memory`, `embedding_across_dp`).
- vllm-ascend precedent: OOT model registration in `vllm_ascend/models/__init__.py` (Glm5Next, KimiLinear, DeepseekV4). MRV2 runner `vllm_ascend/worker/v2/model_runner.py`; state dispatch `vllm_ascend/worker/v2/model_states/__init__.py` (`model.get_model_state_cls()` wins; hybrid→`AscendMambaHybridModelState`; 310P→`_310p/worker/v2/model_state.py` Triton-free states). 310P v2 runner `_310p/worker/v2/model_runner.py` selected in `_310p/worker_310p.py:34`.
- 310P assets: Triton-free GDN ops `_310p/ops/fla/` (`chunk_gated_delta_rule.py`, `fused_recurrent_gated_delta_rule.py`, `fused_gdn_gating.py`), `_310p/ops/gdn_attn_builder_310.py`; worker patches `patch_qwen3_5.py`, `patch_idex_310.py`; W8A8 `_310p/quantization/` (`modelslim_config.py`, `methods/w8a8_dynamic.py` `AscendW8A8DynamicFusedMoEMethod310:37`); `_310p/fused_moe/`; `_310p/attention/attention_v1.py` + metadata builder; sharded loader `_310p/sharded_state_loader_310p.py`; kv config `vllm_ascend/core/kv_cache_interface.py`, `vllm_ascend/utils.py`, patch `patch_kv_cache_utils.py`.

**Documentation policy**: local pinned checkouts are the authoritative docs for vLLM (no Context7 upgrade of pinned tree; PRD bans Transformers upgrades). CANN/torch-npu op availability must be verified against the pinned container, not marketing docs.

## Prerequisites

- vLLM fork checkout with write access: `/run/media/matteius/20TB-drive/vllm`
- vllm-ascend repo: `/run/media/matteius/20TB-drive/vllm-ascend` (this repo)
- ModelSlim W8A8 conversion completing externally (`progress.json` is the source of truth); source rev `de4b8e4d43b917e7706784d8bb445c9af86a3540`
- Pinned container per R1 (CANN / torch-npu / Transformers frozen); `ruff`, `pytest`; `bash format.sh ci` before commit
- Commits: Conventional Commits + `git commit -s` (AGENTS.md). One signed commit per merged task bundle per repo.
- 4-chip 310P target host (two Atlas 300I Duo) — available only for the final hardware wave

## Dependency Graph

```
LEGEND  [asc]=vllm-ascend  [fork]=vLLM  [x]=external  [hw]=device wave

Wave 1 (infra, all parallel):
  T0.1[x] checkpoint   T0.2 env freeze   T0.3 probe tool
  T0.4 corpus gen      T0.5 mem-accounting   T0.6 eager-ref harness
  T1.1[fork] dispatch guard    T7.1 rope config
Wave 2:
  T1.2[asc] pkg + registry (independent, starts immediately)
Wave 3:
  T1.3[asc] model state ← T1.2      T1.4[asc] kv specs ← T1.2
  T3.1 w8a8 map ← T0.1              T4.2 ngram hash UT ← T0.1
  T7.2 token budget ← T7.1
Wave 4 (components):
  T1.5 assembly ← T1.2,T1.3,T1.4,T0.6
  T3.2 shard load ← T0.1,T0.5       T3.3 moe QDQ parity ← T0.6,T3.1
  T4.1 PLE host method ← T1.2       T5.1 GDN wire ← T1.2,T0.6
  T6.1 indexer ← T1.2,T0.6
Wave 5:
  T4.3 PLE port ← T0.6,T1.3,T4.1    T5.2 GDN lifecycle ← T5.1,T1.3
  T6.2 QSA kernel ← T6.1,T1.4
Wave 6:
  T4.4 PLE prefetch ← T4.1,T4.3     T6.3 QSA chunk meta ← T1.4,T6.2
Wave 7:
  T6.4 QSA layer e2e ← T6.1,T6.2,T6.3,T1.5
  TOBS observability ← T0.5,T4.4
Wave 8:  T8.1 C8 proto ← T6.4       T8.2 DCP4 proto ← T6.4,T0.5
Wave 9:  T8.3 1M projection record ← T8.1,T8.2,T3.2

DEVICE WAVE (hardware only, serial after host work):
  D1 probe run ← T0.3,T0.2,hardware
  D1-MB PLE transport micro-bench ← D1,T4.1
  D1.5 on-device component parity (G0) ← D1,T3.3,T4.4,T5.1,T6.2
  D2 G1 8K real-weights ← T1.5,T3.1,T3.2,T3.3,T4.4,T5.2,T6.4,T7.2,D1,D1-MB,D1.5,TOBS
  D3 G2 quant correctness ← D2
  D4 candidate bench@128K + DECISION ← D3,T8.1,T8.2,T8.3
  T8.4 integrate chosen candidate (host re-run of parity suite) ← D4
  D5 G3 128K ← D4,T8.4
  D6 G4 262K + ≥8GiB/chip headroom ← D5
  D7 G5 512K ← D6,T7.1
  D8 G6 1M ← D7
  D9 prefill/chunk tuning ← D6  (feeds D7, D8 rerun if regressed)

Follow-on (out of scope, stubbed): S1 prefix-cache, S2 MTP, S3 multimodal,
S4 ACLGraph, S5 EP4/FlashComm1, S6 docs/tutorial/feature-matrix ← D8
```

## Tasks

### T0.1: Checkpoint completion, manifest and quality report
- **depends_on**: []
- **location**: external ModelSlim conversion run; record pointers in `artifacts/qwen38-1m/checkpoint-manifest.json`
- **description**: Monitor conversion run to completion (49 stages; `progress.json` canonical). Produce manifest: shard list + SHA256, tensor names for expert/scale/offset mapping, `ple_layer_ids`, `ngram_size`, EOS id, rope config, held-out perplexity report (finite/complete). Freeze the G2 accuracy threshold from before/after report **before** any Ascend output is inspected (PRD §8.1).
- **validation**: manifest JSON validates; 49/49 stages done; perplexity finite; threshold recorded with date.
- **status**: Not Completed
- **log**:
- **files edited/created**:

### T0.2: Environment freeze and revision recorder
- **depends_on**: []
- **location**: `tools/qwen38_1m/env_freeze.py` (asc), `docs/.../qwen38_flash_next_1m_env.md`
- **description**: Script that captures and pins revisions for vLLM, vllm-ascend, torch-npu, CANN, Transformers, tokenizers, ModelSlim, checkpoint hash; asserts server imports resolve to `/vllm-workspace/vllm` and `/vllm-workspace/vllm-ascend`. Emits JSON consumed by every run artifact (R1, R13).
- **validation**: UT: given a fake env dict, emits all required keys; `import_check` fails loudly on wrong paths.
- **status**: Completed
- **log**: 2026-09-15 (commit 2688288ac) — Added an injectable-collector environment freeze recorder that pins vLLM/vllm-ascend/torch-npu/CANN/Transformers/tokenizers/ModelSlim revisions + checkpoint hash into schema-versioned JSON, with a loud `import_check` asserting the pinned `/vllm-workspace` import paths (prefixes parameterized for testability). No torch-npu import at load; defaults read package metadata/container env, tests inject fakes. 11 host UTs green (`--noconftest`), ruff clean. Matches the T0.3 `hw_probe.py` conventions (dataclass report, SCHEMA_VERSION, to_dict/to_json/from_dict, human_summary). Gotcha: shared conftest import fails in this env, so tests must run with `--noconftest`.
- **files edited/created**: `tools/qwen38_1m/env_freeze.py` (new), `docs/source/developer_guide/Design_Documents/qwen38_flash_next_1m_env.md` (new), `tests/ut/qwen38_1m/test_env_freeze.py` (new)

### T0.3: Hardware/topology probe tool (WP1)
- **depends_on**: []
- **location**: `tools/qwen38_1m/hw_probe.py` (asc)
- **description**: Records chip inventory, free NPU bytes per chip, firmware/driver, HCCL device topology incl. within-card vs cross-card links, PCIe gen/width, NUMA map, DDR channels, sustained H2D/D2H bandwidth micro-bench. JSON schema + human summary. Author + CPU-test now; run on target in D1.
- **validation**: UT with mocked `npu-smi`/`torch.npu` outputs; schema round-trip.
- **status**: Completed
- **log**: 2026-09-15 — Implemented `probe()` assembling a versioned `ProbeReport` (schema v1) from injectable collector callables: chip inventory (id/card/name/firmware/driver/total+free bytes/NUMA), PCIe gen/width, DDR channel population, and a host<->device bandwidth micro-bench. `classify_links()` derives HCCL peer links and tags each within-card vs cross-card from `card_id` (matches the TOBS collective annotation, `CHIPS_PER_300I_DUO_CARD=2`). `min_free_bytes()` surfaces the binding placement constraint (feeds the capacity model / open decision #1). JSON round-trips via `from_dict`; `human_summary()` renders GiB free per chip, link counts, PCIe and bandwidth. The default hardware chip collector raises `NotImplementedError` until finalized against the pinned CANN container on target (D1), so callers must inject real output before then; the module imports no torch-npu at load. 8 UTs green via `pytest --noconftest` using mocked collectors (no NPU); ruff check + format clean. **Note**: loading a `tools/`-path module via importlib requires registering it in `sys.modules` before `exec_module` on Python 3.14 (dataclass annotation resolution) — reused for future `tools/qwen38_1m` UTs.
- **files edited/created**: `tools/qwen38_1m/hw_probe.py` (new), `tools/qwen38_1m/__init__.py` (new), `tests/ut/qwen38_1m/test_hw_probe.py` (new)

### T0.4: Long-context corpus and retrieval-probe generator
- **depends_on**: []
- **location**: `tools/qwen38_1m/corpus_gen.py` (asc), `tests/e2e/long_context/`
- **description**: Deterministic (seeded) prompt builders at 8K/128K/262144/524288/1,048,576-minus-516 total tokens. Embeds 8 distributed needle/retrieval records near beginning, quarter, middle, three-quarter, end + 1 adversarial no-answer case. Records exact token counts via the checkpoint tokenizer. Server-side config snippets to disable input truncation and report accepted token count (R10).
- **validation**: UT: token counts exact at all sizes; probes deterministic across runs; expected answer keys extractable.
- **status**: Completed
- **log**: 2026-09-15 (commit d29ac2b66) — Added deterministic seeded NIAH corpus generator with an injectable tokenizer (`TokenizerLike`: encode + eos_id); exact token counts at all 5 sizes (8192/131072/262144/524288/1048060) hit via a re-encode padding convergence loop, so it is tokenizer-agnostic. 8 answerable needles across beginning/quarter/middle/three-quarter/end + 1 adversarial no-answer (NO_ANSWER sentinel); `expected_answers()` yields the grading map. e2e snippets (server-config.yaml, request-template.json) disable truncation and require accepted-token reporting (R10). 14 host UTs green (`--noconftest`), ruff clean. **Deferred**: real-tokenizer exact-count validation → device wave (D1); the Qwen4Exp checkpoint tokenizer is not on this host.
- **files edited/created**: `tools/qwen38_1m/corpus_gen.py` (new), `tests/ut/qwen38_1m/test_corpus_gen.py` (new), `tests/e2e/long_context/{README.md,server-config.yaml,request-template.json}` (new)

### T0.5: Per-rank memory accounting harness (R2, R13)
- **depends_on**: []
- **location**: `vllm_ascend/observability/qwen38_mem_accounting.py` (asc)
- **description**: Hooks weight-load and cache-allocation to report per-rank bytes by component (embedding, non-expert FP16, expert W8A8, QSA main KV, indexer ring/compressed, GDN state, PLE host table, workspaces, free, peak). Fails startup when per-rank placement imbalance >5% w/o approved reason. Host-testable via fake device stats.
- **validation**: UT: component sums match a synthetic allocation trace; imbalance violation raises.
- **status**: Completed
- **log**: 2026-09-15 — Implemented `MemoryAccountant`/`RankMemoryReport` with a `MemComponent` enum covering all required components (embedding, non-expert FP16, expert W8A8, quant scales, QSA main KV, indexer ring/compressed, GDN state, workspaces, PLE host table). Pure stdlib (no torch/torch-npu) so it is fully host-testable via injected device stats; the harness never queries the accelerator. Device-vs-host split via `HOST_COMPONENTS` keeps the shared 95.43 GiB PLE table out of per-rank device totals and the imbalance check, and `host_table_bytes()` raises if ranks report divergent host bytes (guards against ×rank PLE copies). `validate_balance()` raises `PlacementImbalanceError` above the 5% `MAX_PLACEMENT_IMBALANCE` unless an approved reason is passed. Emits machine-readable JSON (`to_dict`/`to_json`, feeds TOBS) and a GiB human summary. 13 UTs green via `pytest --noconftest` (shared `tests/ut/conftest.py` fails to import in this env due to an unrelated installed-vLLM `is_weak_contiguous` mismatch — the module itself imports cleanly). ruff check + format clean. Test fixtures use the measured 31.88 GiB/chip / 95.43 GiB-PLE figures from the runtime-requirements doc.
- **files edited/created**: `vllm_ascend/observability/qwen38_mem_accounting.py` (new), `tests/ut/qwen38_1m/__init__.py` (new), `tests/ut/qwen38_1m/test_mem_accounting.py` (new)

### T0.6: Eager reference harness (CPU) for GDN, QSA, indexer, PLE, MoE QDQ
- **depends_on**: []
- **location**: `tests/ut/qwen38_1m/reference/` (asc)
- **description**: Pure-PyTorch (FP64/FP32) reference implementations of: Qwen4Exp QSA indexer scoring + block selection + 2,048-token budget semantics; QSA sparse attention with partial rotary, Q/K norm, output gate, causal, selection-count semantics; GDN conv+recurrent (chunk and unchunked); Qwen n-gram hashing with EOS boundaries; PLE projection/conv; W8A8 dynamic INT8 QDQ (per-token act scale, checkpoint weight scale/offset). Derived from `nvidia/ops/*` Triton kernels (port kernel formulas, do not import Triton). Tolerances declared per op BEFORE first comparison of final results, per PRD §8.1.
- **validation**: self-consistency UTs (chunked vs unchunked GDN, index hash vs brute force).
- **status**: Completed
- **log**: 2026-09-15 (commit fb3ee0749) — Eager CPU reference harness under `tests/ut/qwen38_1m/reference/`. All 6 pure-PyTorch (FP64/FP32, no-Triton) references complete + self-consistency tested (184 UTs green via `--noconftest`). GDN ships both chunked (self-derived UT/WY block) and unchunked (recurrent, matches fused sigmoid-gating kernel) — chunked==unchunked ~1e-16. QSA indexer covers compression/scoring/top-k/2048-budget at all required boundary lengths (1, ratio-1, ratio, ratio+1, partial, exactly-2048, 2049+); QSA attention (24q/2kv/dim256, partial RoPE, QK-norm, output gate) sparse==dense-masked. n-gram exact vs brute-force incl. EOS boundary + history advance. W8A8 per-token act QDQ + per-channel weight scale/offset with correct `(q-offset)*scale` order (wrong-order regression guard). Tolerances declared in `tolerances.py` BEFORE all asserts (per-op, each with rationale). Gotcha: W8A8 round-trip bounded absolutely by scale/2 per-element (relative bound meaningless near zero). This unblocks T3.3, T4.3, T5.1, T6.1, T6.2, T1.5.
- **files edited/created**: `tests/ut/qwen38_1m/reference/{__init__,tolerances,gdn_reference,ngram_hash_reference,ple_reference,qsa_indexer_reference,qsa_attention_reference,w8a8_reference}.py` + `test_{gdn,ngram_hash,ple,qsa_indexer,qsa_attention,w8a8}_selfconsistency.py` (all new)

### T1.1 [fork]: Backend-safe qwen4_exp dispatch
- **depends_on**: []
- **location**: `vllm/models/qwen4_exp/__init__.py` (fork)
- **description**: Replace the `else: import nvidia` fallthrough with explicit `is_cuda()`/`is_rocm()` branches; unknown platforms get a structured `NotImplementedError` naming the platform, unless an out-of-tree override is registered. Must not regress CUDA/ROCm. Keep XPU/TPU messages. Also harden the JIT-warmup path `vllm/model_executor/warmup/qwen4_exp_qsa_warmup.py` (dynamically imports Triton kernels at worker init, currently guarded only by `sys.modules.get` + JIT flag) so it is skipped before ever touching `qwen4_exp.nvidia` on non-CUDA platforms.
- **validation**: fork UT: simulating an NPU platform never imports `vllm.models.qwen4_exp.nvidia` — including a worker-init simulation with JIT warmup enabled that asserts `vllm.models.qwen4_exp.nvidia` never enters `sys.modules`; CUDA/ROCm dispatch unchanged (existing tests).
- **status**: Completed
- **log**: 2026-09-15 (fork commit 6a4da62) — Replaced qwen4_exp `__getattr__` implicit nvidia fallthrough with explicit `is_rocm()`→amd / `is_cuda()`→nvidia branches; unsupported platforms raise a platform-naming `NotImplementedError`. OOT override preserved by resolving `ModelRegistry.models[name]` first, skipping the self-referential in-tree lazy entry (module_name/`__module__` checks) to avoid recursion. Hardened QSA warmup with an early `is_cuda()` skip so worker init never imports the nvidia backend on NPU. New `tests/models/qwen4_exp/test_dispatch.py` (7 subprocess-isolated tests) proves NPU never imports nvidia (incl. warmup sim), OOT override wins, CUDA/ROCm unchanged; `test_config.py` still 9 passed. Orchestrator re-ran: 7 passed (42s) under `VLLM_TARGET_DEVICE=cpu --noconftest` (fork conftest needs tblib/compiled libs absent here). Fork repo was already detached-HEAD; not pushed.
- **files edited/created**: fork: `vllm/models/qwen4_exp/__init__.py`, `vllm/model_executor/warmup/qwen4_exp_qsa_warmup.py`, `tests/models/qwen4_exp/test_dispatch.py` (new)

### T1.2 [asc]: Ascend Qwen4Exp package and registration
- **depends_on**: []
- **location**: `vllm_ascend/models/qwen4_exp/{__init__,model,ple_layer,qsa,indexer_qsa,ngram_embedding,mtp}.py`, `vllm_ascend/models/__init__.py`
- **description**: Create `AscendQwen4ExpForCausalLM` (+ `ForConditionalGeneration` alias rejecting multimodal inputs at first gate; `Qwen4ExpMTP` registration left to S2). Reuse `common.hyperconnection`, `common.qsa_cache`, `common.ple`, upstream `Qwen3NextSparseMoeBlock`, `QwenGatedDeltaNetAttention` with Ascend wiring. No Triton/CUDA imports on 310P (guard via `_310p.ops` + `device_op`). Model classes expose `get_model_state_cls`, mamba state shape/dtype hooks (mirror `nvidia/model.py:746-830`), and per-component `load_weights` with fused-expert mapping hooks (T3.1). **Also own the dtype policy** (PRD §5.3/R4): pin `dtype` (`float16`), `mamba_ssm_cache_dtype`, QSA main/indexer/`kv_cache_dtype` and every required FP16/FP32 cast site (attention, router, shared expert, PLE projection, gated residual, LM head); record the table in the task log and propagate to T0.5/T8.3 byte math — 310P kernel dtype support must be checked against the pinned CANN here, not assumed. Include `ParallelLMHead` + embedding tie consistent with vocab parallelism and AscendSampler integration (R3 determinism). Register all three arch names in `vllm_ascend/models/__init__.py`; note: OOT `ModelRegistry.register_model` overwrites the single in-tree dict entry and this override is process-global via the `vllm.general_plugins` entry point (`vllm_ascend:register_model`).
- **validation**: CPU UT: import under faked NPU platform, build tiny random Qwen4Exp config, model constructs on meta device; server-side `--model` resolution picks Ascend class and the fork's `Qwen4Exp*Config` validators (`vllm/model_executor/models/config.py:858-916`) pass unmodified for Ascend launches (same arch names → same config contract); plugin-ordering UT incl. `VLLM_PLUGINS` exclusion behavior; grep-gate: no `triton` import reachable from package on 310P flag; dtype-policy table exists and every QSA/GDN/PLE module reads it (no local dtype literals).
- **status**: Completed
- **log**: 2026-09-15 (commit d64064d87) — Created importable/registered `vllm_ascend/models/qwen4_exp/` package + authoritative `Qwen4ExpDtypePolicy` (float16 main + fp32 accumulation; fp32 SSM cache; float16 KV/conv/QSA/indexer/PLE-projection/gated-residual/LM-head/embedding; fp32 router/logits/PLE-norm-accum/gated-residual-accum). Access via `ASCEND_QWEN4EXP_DTYPE_POLICY` or `.from_vllm_config()`; `REQUIRED_CAST_SITES` + `policy.cast_site(name)`; frozen dataclass, `dtype_policy.py` is the ONLY file with dtype literals (source-scan test forbids bare literals elsewhere; grep-gate confirms no triton import in package). `AscendQwen4ExpForCausalLM` constructs on meta (ParallelLMHead + VocabParallelEmbedding + tie; state-cls/mamba-dtype/load_weights/get_expert_mapping hooks); `AscendQwen4ExpForConditionalGeneration` rejects multimodal at first gate. Registered all 3 arch names in `models/__init__.py` (additive lazy-string, verified non-breaking). **Adaptation**: installed host vLLM is 0.22.0 and lacks the fork's `Qwen4ExpTextConfig`, so the skeleton duck-types on `hf_text_config` (downstream ports from fork `nvidia/*` still reference the fork tree). **CANN dtype support (float16 main + fp32 accum) is a documented assumption to verify on hardware** — override in `dtype_policy.py`, not call sites. Stubs: forwards→T4.x/T5.x/T6.1; load_weights/fused-expert→T3.1; state-cls & mamba shape hooks→later; ple_layer→T1.3, qsa→T1.4, indexer_qsa→T1.5, ngram_embedding→T1.3; MTP registered, wired in S2. 13 new UTs green (`--noconftest`), ruff clean.
- **files edited/created**: `vllm_ascend/models/qwen4_exp/{__init__,dtype_policy,model,ple_layer,qsa,indexer_qsa,ngram_embedding,mtp}.py` (new), `vllm_ascend/models/__init__.py` (registration), `tests/ut/qwen38_1m/{test_qwen4exp_registration,test_qwen4exp_dtype_policy}.py` (new)

### T1.3 [asc]: 310P Qwen4Exp model state (PLE n-gram context)
- **depends_on**: [T1.2]
- **location**: `vllm_ascend/_310p/worker/v2/model_state.py` (new `Ascend310PQwen4ExpModelState`), `vllm_ascend/worker/v2/model_states/__init__.py`
- **description**: Port `Qwen4ExpModelState(MambaHybridModelState)` onto `Ascend310PMambaHybridModelState`: n-gram context buffer maintenance, EOS padding, `ple_query_start_loc`, rollback safety; shapes fixed for graph-capture compatibility even in eager. PP=1 enforcement retained.
- **validation**: CPU UT on fake input batches: n-gram context correctness at chunk boundaries, EOS padding, after rejected-speculative rollback (S=0 degenerate first), dummy-inputs path.
- **status**: Completed
- **log**: 2026-09-15 (commit 4d895e714) — Ported fork `Qwen4ExpModelState` PLE n-gram context onto `Ascend310PQwen4ExpModelState(Ascend310PMambaHybridModelState)`: rollback-safe context buffer (rebuilt from `num_computed_tokens` each step, no stale carryover), EOS-padded leading tokens, fixed-shape `ple_query_start_loc`/`ngram_context` for graph capture, PP=1 enforced (PP>1 and ngram_size=1 rejected). Wired via worker dispatch `model_states/__init__.py`: the model `get_model_state_cls()` hook still wins, but since T1.2's hook currently raises NotImplementedError it falls through and a 310P hybrid advertising `ple_layer_ids` routes to the new state (model.py left untouched — in scope). Covers chunk boundary, EOS pad, rollback incl. S=0 degenerate-first, dummy-inputs, oversized 310P `max+2` start-loc buffer. Test installs minimal host stubs then removes `torch_npu` so sibling "no torch_npu" asserts still pass. 19 UTs; full qwen38_1m suite 373 passed (orchestrator-verified), ruff clean. Unblocks T5.2, T1.5, T4.3.
- **files edited/created**: `vllm_ascend/_310p/worker/v2/model_state.py` (+state class), `vllm_ascend/worker/v2/model_states/__init__.py` (+dispatch), `tests/ut/qwen38_1m/test_qwen4exp_model_state.py` (new)

### T1.4 [asc]: Materialize Qwen4Exp KV-cache specs on 310P
- **depends_on**: [T1.2]
- **location**: `vllm_ascend/core/kv_cache_interface.py`, `vllm_ascend/utils.py`, `vllm_ascend/patch/platform/patch_kv_cache_utils.py`, `_310p` kv-config path
- **description**: Support `CircularBufferSpec` (QSA raw ring, capacity must divide/align attention block size → scheduler LCM) and `MLAAttentionSpec(tokens_per_state=compress_ratio)` (compressed index) alongside GDN `MambaSpec` and full-attention groups; hybrid group packaging, block-size LCM, granularity/bytes calc for 310P. Ensure `QSAStateBackend` (name `QWEN4_EXP_EXP_QSA_STATE`) pages allocate on NPU without Triton.
- **validation**: CPU UT on `kv_cache_utils`: spec set from tiny Qwen4Exp config yields a `KVCacheConfig` with correct groups/block sizes; 1M-token block math exact (report bytes/chip table for BF16/C8 layout — feeds T8.x).
- **status**: Completed
- **log**: 2026-09-15 (commit c98860dc8) — QSA KV-cache specs on the 310P host lane: key-only `AscendQSARawRingSpec` (ring cap 4, page 1024 B) + `MLAAttentionSpec(compress_ratio=4)` compressed index, packaged in hybrid groups beside GDN MambaSpec/full-attn. **EXACT 1M bytes/QSA-layer** (element sizes from dtype policy: fp16=2 B, fp8_e4m3=1 B) — BF16: ring 1,024 B + compressed 67,108,864 B (64 MiB) = 67,109,888 B; C8: ring 1,024 B + compressed 33,554,432 B (32 MiB) = 33,555,456 B (C8 compressed exactly ½ BF16); scales linearly by QSA-layer count → feeds T8.x. Ring capacity divides scheduler LCM (128); non-dividing ratios raise it (ratio-3→384). `AscendQSAStateBackend` (`QWEN4_EXP_EXP_QSA_STATE`, `uses_triton()==False`) via pure-torch slot-mapping fallback. **DEVIATION (documented)**: installed vLLM 0.22.0 lacks `CircularBufferSpec`/`num_states` API and its generic `AttentionSpec` doubles for a V tensor → introduced a key-only Ascend ring spec + LCM handling in `patch_kv_cache_utils` (guarded to only fire when a QSA ring group is present — verified non-invasive for other models; runner tensor-alloc wiring stays NPU-side). Shared-file edits purely additive (+13/+26/+13). 15 CPU UTs green, ruff + py_compile clean.
- **files edited/created**: `vllm_ascend/models/qwen4_exp/kv_cache.py` (new), `tests/ut/qwen38_1m/test_kv_cache_specs.py` (new), `vllm_ascend/core/kv_cache_interface.py` (+register), `vllm_ascend/patch/platform/patch_kv_cache_utils.py` (+LCM), `vllm_ascend/utils.py` (+helper)

### T3.1: W8A8 tensor mapping and load-time rejection
- **depends_on**: [T0.1]
- **location**: `vllm_ascend/models/qwen4_exp/weight_mapping.py`, `vllm_ascend/_310p/quantization/modelslim_config.py` extension; UT in `tests/ut/qwen38_1m/test_weight_mapping.py`
- **description**: Explicit map: source fused expert tensors + ModelSlim per-expert scale/offset tensors → `AscendW8A8DynamicFusedMoEMethod310` weight layout; router/shared/attention/LM-head as FP16. Reject missing, extra, duplicate, incompatible-shape/dtype tensors with actionable errors. Drive from T0.1 manifest. **Run under the T1.2 dtype policy.**
- **validation**: CPU UT with manifest + synthetic safetensors index: happy path covers all 73,728 expert projections; each rejection class fires once; every tensor checked against the frozen dtype policy before mapping.
- **status**: Not Completed
- **log**:
- **files edited/created**:

### T3.2: Streamed sharded loading and EP/TP placement accounting
- **depends_on**: [T0.1, T0.5]
- **location**: `vllm_ascend/_310p/sharded_state_loader_310p.py`, T3.1 mapping; UT `tests/ut/qwen38_1m/test_shard_placement.py`
- **description**: Ensure no rank materializes the full 512-expert bank: stream + shard by TP4 (EP4 path ready but TP4-default per R3). Verify expert sharding under `Qwen4ExpSparseMoeBlock`; integrate T0.5 per-rank accounting; simulate full-checkpoint placement on CPU (manifest-only) and emit predicted per-chip bytes.
- **validation**: CPU simulation UT: per-rank bytes ≤ target from PRD §6 / `qwen38_flash_next_1m_runtime_requirements.md` (**measured 31.88 GiB/chip non-PLE**, was ≈30.8 GiB estimate) ±manifest drift; peak CPU RSS during simulated load bounded; no tensor instantiated twice.
- **status**: Not Completed
- **log**:
- **files edited/created**:

### T3.3: W8A8 fused MoE parity vs eager QDQ reference
- **depends_on**: [T0.6, T3.1]
- **location**: `tests/ut/qwen38_1m/test_moe_w8a8_parity.py`; fixes in `_310p/quantization/methods/w8a8_dynamic.py`, `_310p/fused_moe/`
- **description**: On CPU (using `AscendRoutedExperts` math path / numpy int8 simulate or device-op stub), compare 512-expert top-10 + shared expert forward against T0.6 QDQ reference at declared tolerances. Cover expert-distribution skew, router renormalization, offset application order. Avoid `tensor.item()` in hot paths (AGENTS.md).
- **validation**: parity UT green at pre-declared tolerances; skew test (all tokens→one expert) passes.
- **status**: Not Completed
- **log**:
- **files edited/created**:

### T4.1: Host PLE table ownership and lookup method (R5)
- **depends_on**: [T1.2]
- **location**: `vllm_ascend/models/qwen4_exp/ngram_embedding.py` (`AscendPLEEmbeddingMethod` subclassing in-tree ABC `Qwen4ExpPLEEmbedding`), `EngramConfig` usage (asc ascend_config surface)
- **description**: One logical FP16 host table shared across 4 worker processes. **Ship BOTH transports behind one interface** with a config switch, because the winner can only be settled on hardware in D1: (a) `is_uva_available`/`get_accelerator_view_from_cpu_tensor` pinned-UVA path (mirrors `Qwen4ExpPLEPinnedHostEmbedding`), (b) file-backed shared mmap + registered transfer windows + batched row gather. The switch defaults to (a) with auto-fallback and is exercised by a D1-runnable micro-benchmark whose result is a D2 entry criterion. **Never** allow ×rank copies; fail-fast host accounting vs 48 GiB OS/transfer reserve (PRD §6). Pin only measured regions.
- **validation**: UT with /dev/shm tables: 4 processes observe same physical pages (rss proves sharing), row reads correct, host byte accounting exact; no 95 GiB pin attempted by default; transport-switch micro-bench script ships with the task for D1.
- **status**: Completed
- **log**: 2026-09-15 (commit 3e0ef5791) — Dual-transport host PLE table behind one interface `AscendPLEEmbeddingMethod`: (a) pinned-UVA (device calls guarded/injectable, D1-verified) mirroring fork `Qwen4ExpPLEPinnedHostEmbedding`, and (b) /dev/shm MAP_SHARED mmap with `register_transfer_window()` (pins only measured regions) + batched `gather_rows()` (fully host-tested). `create_ple_embedding_method()` defaults to (a), auto-falls back to (b) when UVA absent, forced to (b) on `EngramConfig.dp_shared_memory` (no new env var). Single shared copy enforced via T0.5 `MemoryAccountant` (`PLE_HOST_TABLE`, `host_table_bytes()` rejects ×rank); 48 GiB reserve fail-fast (`HostByteBudgetError`, no file created); whole-table pin raises. 4-process fork test proves one physical copy (all workers see all 4 sentinels; mmap_length==table_bytes). Dtypes only from policy `cast_site("ngram_embedding")` — T1.2 dtype-literal test still passes. 26 UTs green (T4.1+dtype+registration, no regression), ruff clean. Fork ABC not importable host-only → contract mirrored, documented D1-composed. Async prefetch left as hook for T4.4. Unblocks T4.3 (needs T1.3) and T4.4.
- **files edited/created**: `vllm_ascend/models/qwen4_exp/ngram_embedding.py` (filled T1.2 stub), `tools/qwen38_1m/ple_transport_bench.py` (new), `tests/ut/qwen38_1m/test_ple_host_method.py` (new)

### T4.2: Exact n-gram hashing and EOS boundary tests
- **depends_on**: [T0.1]
- **location**: `tests/ut/qwen38_1m/test_ngram_hash.py`
- **description**: Port Qwen n-gram hashing (hash fn, vocab parallel sharding columns, EOS padding of first tokens, history advance). Verify against the checkpoint's embedding table directly: sample token windows → hashed row identities equal a brute-force re-derivation from the safetensors index/table.
- **validation**: UT exact-match on sampled + boundary windows (sequence start, EOS, across TP shard boundaries; shard col mapping per manifest).
- **status**: Not Completed
- **log**:
- **files edited/created**:

### T4.3: PLE layer port (projection/gather/combine) + parity
- **depends_on**: [T0.6, T1.3, T4.1]
- **location**: `vllm_ascend/models/qwen4_exp/ple_layer.py`, `ops/`
- **description**: Port `Qwen4ExpPLELayer` from `nvidia/ple_layer.py`: row gather via the T4.1 host PLE method interface, projection (FP16/FP32 per T1.2 dtype policy), integration point in decoder layer. Because T4.3 both consumes T4.1 and is a prerequisite of D1's transport pick, order inside the task is: land against transport (b) first (CPU-testable), then re-run parity against (a); D1 re-measures both. CPU eager path. No device hot-path `.item()`; batched lookups only.
- **validation**: CPU parity UT vs T0.6 reference at short context (declared tolerances); PLE row identity test feeds T4.2 rows.
- **status**: Not Completed
- **log**:
- **files edited/created**:

### T4.4: Async row prefetch, dedup and PLE metrics
- **depends_on**: [T4.1, T4.3]
- **location**: `vllm_ascend/models/qwen4_exp/ple_prefetch.py`
- **description**: Batch/dedup requested rows; async H2D on dedicated stream(s) overlapped with compute; decode-loop must not sync per row. Metrics: host bytes, page faults, transfer bytes/step, hit rate, lookup latency (→TOBS schema).
- **validation**: CPU-sim UT: dedup ratio on realistic access trace; decode step function completes without sync-point in simulated stream timeline; metrics emitted.
- **status**: Not Completed
- **log**:
- **files edited/created**:

### T5.1: GDN wiring on 310P
- **depends_on**: [T1.2, T0.6]
- **location**: `vllm_ascend/models/qwen4_exp/qwen4exp_gdn.py` (adapter over `QwenGatedDeltaNetAttention` or subclass), `_310p/ops/fla/*`, `patch_idex_310.py` precedent
- **description**: Wire Qwen4Exp GDN layers (conv + recurrent state shapes/dtype for hidden 2560 config) to existing Triton-free 310P fla kernels; verify partial-rotary/short-conv params from HF config; eager path only (no ACLGraph).
- **validation**: CPU parity UT vs T0.6 chunked/unchunked GDN; state shape/dtype assertions == Qwen4Exp config-derived expectations.
- **status**: Completed
- **log**: 2026-09-15 (commit 1fcc9cc25) — Config/dtype-policy-driven adapter over the Triton-free 310P fla kernels: `Qwen4ExpGDNParams.from_hf_config` derives geometry (head_dim 256×0.25→rotary_dim 64, conv kernel 4); conv/recurrent state shapes mirror `MambaStateShapeCalculator` (ssm `(32,128,128)`, conv `(8192,3)`); dtypes from T1.2 policy (conv fp16, ssm fp32). Delta rule dispatches `fla_pytorch` (real fused_recurrent/chunk) / `ascend_npu` (chunk_gated_delta_rule_310) / `eager`. **Finding (documented, plan-anticipated)**: stock fla PyTorch fallbacks are float32-internal so they miss the T0.6 float64 tolerances (out ~6.6e-8 / chunk ~4.0e-7 vs 1e-8/1e-9) — added a dtype-honoring eager path reaching ~1e-16..4e-15 at fp64 and reproducing the stock kernels at fp32; a RED test pins the stock-kernel gap. chunked==unchunked incl. grouped-value Hv=2·Hk. State lifecycle left to T5.2 (clean hooks). 50 UTs; ruff clean.
- **files edited/created**: `vllm_ascend/models/qwen4_exp/qwen4exp_gdn.py` (new), `tests/ut/qwen38_1m/test_gdn_wiring.py` (new)

### T5.2: GDN state lifecycle across prefill/decode/preemption/reuse
- **depends_on**: [T5.1, T1.3]
- **location**: state ops in `_310p/worker/v2/model_state.py`, `vllm_ascend/models/qwen4_exp/` , UT `tests/ut/qwen38_1m/test_gdn_lifecycle.py`
- **description**: Copy/slot-remap semantics across 4 ranks (per-rank state replicated per TP rules — no aliasing between requests, block reuse, preemption→resume with correct state or explicit fail-closed, request completion). Reuse `MambaAttentionBackendEnum` copy-func pattern (`nvidia/model.py:802`).
- **validation**: CPU UT: chunked-vs-unchunked identical outputs; interleaved 2-request sequence + forced preemption keeps outputs equal to unpreempted run; aliasing test (two requests never point to same state block).
- **status**: Not Completed
- **log**:
- **files edited/created**:

### T6.1: QSA indexer on 310P
- **depends_on**: [T1.2, T0.6]
- **location**: `vllm_ascend/models/qwen4_exp/indexer_qsa.py` + `ops/`
- **description**: Port `nvidia/indexer_qsa.py` + `ops/qsa_indexer.py`/`qsa_pre_indexer.py` to torch/NPU ops: 4 q-heads, 1 k-head, dim 128, compression ratio 4, budget 2,048; raw ring write (per token) and compressed history (1 row/4 tokens) via `common.qsa_cache` slot mappings (use `_build_qsa_metadata_torch` fallback, no Triton). Deterministic top-k.
- **validation**: CPU parity UT vs T0.6 brute-force at boundary lengths (1, ratio-1, ratio, ratio+1, partial group, exactly-2048, 2049+), repeated-block selection, final token; selection sets identical to reference.
- **status**: Completed
- **log**: 2026-09-15 (commit 17fce870b) — Ported the weight-free QSA indexer to torch (Triton/CUDA/NPU-free). New `models/qwen4_exp/ops/` package: `qsa_indexer` (mean-pool compress, fp32 relu-summed block scores, deterministic top-k, causal-tail expand) and `qsa_cache` (ports of ring/compressed slot mappings + `_build_qsa_metadata_torch` fallback + scatter/gather — fork `common/qsa_cache.py` not importable host-only, so logic ported). Deterministic top-k = `argsort(descending, stable)`, ties by ascending block index (matches reference). Critical path uses cache slot-mapping: raw-ring write per token (retains open-group suffix), compressed history 1 row/completed group, compressed keys gathered through the cache before scoring. Dtypes from policy (fp16 storage / fp32 accum, no literals — T1.2 scan still passes). Selection sets IDENTICAL to T0.6 at every boundary length + over-budget/repeated/final-token; bitwise-deterministic across runs. Exposes packed selection buffer (indices + valid-count column) for T6.2. 19 UTs green (+7 dtype sibling), ruff clean.
- **files edited/created**: `vllm_ascend/models/qwen4_exp/indexer_qsa.py` (filled stub), `vllm_ascend/models/qwen4_exp/ops/{__init__,qsa_indexer,qsa_cache}.py` (new), `tests/ut/qwen38_1m/test_qsa_indexer.py` (new)

### T6.2: QSA sparse attention kernel (BF16) on 310P
- **depends_on**: [T6.1, T1.4]
- **location**: `vllm_ascend/models/qwen4_exp/qsa.py` + `ops/`
- **description**: Gather selected ≤2,048 rows + compressed rows from caches; attention with Q/K norm, partial rotary, output gate, causal, selection-count semantics; main-cache dtype follows the T1.2 dtype policy (planning baseline BF16 — matches PRD §6; C8 int8 variant is T8.1); write path reserves the quant hook for Candidate A. Deterministic. Torch-eager implementation sized for correctness; structure permits later kernel replacement.
- **validation**: CPU deterministic UTs vs T0.6 for same boundary lengths + full-block selection + zero-selection first token; run-to-run bitwise stability test.
- **status**: Completed
- **log**: 2026-09-15 (commit d33f52b22) — Torch-eager Triton-free QSA sparse GQA: per-head Q/K GemmaRMSNorm (`x*rsqrt(mean(x²)+eps)*(1+w)`) → partial neox RoPE (rotary_dim=64=256×0.25, tail passthrough) → gather indexer-selected rows via T6.1 `qsa_gather_rows` → softmax GQA (scale head_dim⁻⁰·⁵, 24q/2kv/256) over the selected set → `out*sigmoid(gate)` applied last. Selection-count semantics honored (only first `valid_count` packed entries; `-1` padding dropped; zero-selection first token → exactly 0). Dtypes from policy (fp16 storage/fp32 accum, no literals). Parity vs T0.6 at fp64: max err ~1.9e-15 (tol 1e-8/1e-9); full-block op-vs-dense-oracle 3.3e-16; bitwise deterministic on the fp16/fp32 path. **C8 quant hook reserved** in `qsa_write_kv_to_cache(quant_hook=)` / `kv_write_quant_hook` (unset — C8 is T8.1). Seams left for T6.3 (chunked prefill) / T6.4 (assembly). Did not touch `ops/__init__.py`. 19 UTs (+19 T6.1 sibling still green), ruff clean.
- **files edited/created**: `vllm_ascend/models/qwen4_exp/qsa.py` (filled stub), `vllm_ascend/models/qwen4_exp/ops/qsa_attention.py` (new), `tests/ut/qwen38_1m/test_qsa_attention.py` (new)

### T6.3: Chunked prefill for QSA (configurable chunk size, default 4096)
- **depends_on**: [T1.4, T6.2]
- **location**: QSA layer/metadata wiring; scheduler config validation (asc)
- **description**: Route `QSAMetadataBuilder` torch fallback through 310P MRV2; chunk-boundary correctness: ring/compressed state advances identically across chunk splits; make chunk size a config knob; preemption-aware recompute policy.
- **validation**: CPU UT: full-sequence vs chunked(4096, ragged last chunk, chunk==block_size edge) outputs equal within pre-declared tolerance.
- **status**: Not Completed
- **log**:
- **files edited/created**:

### T6.4: End-to-end QSA decoder layer assembly and short-context parity
- **depends_on**: [T6.1, T6.2, T6.3, T1.5]
- **location**: `vllm_ascend/models/qwen4_exp/qsa.py`, model assembly
- **description**: Plug full QSA path (project→indexer→select→sparse attn→gate) into the decoder layer replacing the T1.5 eager fallback; 12 QSA layers exercise identical code. Logit parity vs reference model at 8K for fixed seeds.
- **validation**: CPU parity UT full-forward tiny model vs T0.6 composite reference; deterministic greedy outputs on 8K synthetic equal across two runs.
- **status**: Not Completed
- **log**:
- **files edited/created**:

### T7.1: Explicit long-context RoPE/YaRN configuration
- **depends_on**: []
- **location**: fork `vllm/transformers_utils/configs/qwen4_exp.py` handling + `vllm/model_executor/models/config.py`, asc `_310p/worker/v2/rope.py`, `vllm/config` docs snippets
- **description**: Native 262,144 mode unchanged. Add explicit, validated extension config (rope_type/scaling + max positions) supplied via deployment metadata; refuse `max_model_len > 262,144` without it (no silent raise). Implement/verify yarn (or Qwen-specified) scaling up to 1,048,576 against Qwen reference formula on CPU; publish the authoritative parameter set (PRD open decision #5 → record in plan log when fixed).
- **validation**: CPU UT: cos/sin tables at positions {0, 262143, 262144, 524288, 1048575} equal reference formula within FP tolerance; startup guard UT rejects bare max_model_len bump.
- **status**: Completed
- **log**: 2026-09-15 (asc commit 83c8ea1ed, fork commit 3ab5dda) — Native 262,144 window unchanged; added an explicit, validated YaRN extension path that REFUSES `max_model_len > 262144` without a valid config (no silent auto-scale), enforced in both the fork config validation (`Qwen4ExpForConditionalGenerationConfig.verify_and_update_config`) and the 310P worker rope module. **Authoritative 1M RoPE set (resolves open decision #5)**: rope_type="yarn", rope_theta=10_000_000.0, factor=4.0, original_max_position_embeddings=262_144, beta_fast=32, beta_slow=1, mscale=yarn_get_mscale(4.0)≈1.13863, partial_rotary_factor=0.25 (rotary_dim=64); extends to 1_048_576. CPU cos/sin verified at {0,262143,262144,524288,1048575} against an independent reference and vLLM's canonical YaRN at atol=rtol=1e-4. 14 asc UTs green (`--noconftest`); ruff clean. Fork guard verified by loading fork `config.py` standalone (fork conftest can't collect here — missing compiled libs). Two disjoint fork commits (T1.1 6a4da62, T7.1 3ab5dda) landed cleanly in sequence.
- **files edited/created**: asc: `vllm_ascend/_310p/worker/v2/rope.py` (extended), `tests/ut/qwen38_1m/test_rope_yarn.py` (new); fork: `vllm/model_executor/models/config.py` (extended), `tests/models/qwen4_exp/test_config.py` (extended)

### T7.2: Context-window budgeting (input + output + drafts)
- **depends_on**: [T7.1]
- **location**: asc serving config, admission validation UT
- **description**: Enforce 1,048,060 input + 512 output + 4 draft ≤ 1,048,576; report accepted-prompt tokens (no truncation, R10); admission and execution concurrency = 1 for 1M path; explicit errors instead of truncation.
- **validation**: UT: boundary accept/reject cases; response metadata includes accepted token count.
- **status**: Not Completed
- **log**:
- **files edited/created**:

### T1.5: Full-model assembly with eager fallback (dummy-weight boot)
- **depends_on**: [T1.2, T1.3, T1.4, T0.6]
- **location**: `vllm_ascend/models/qwen4_exp/model.py`
- **description**: Assemble 48-layer model (PLE layer 1 position per `ple_layer_ids`, 36 GDN, 12 QSA, MoE incl. shared expert, gated residual/hyperconnection) using correct components where already wired and pure-eager reference math (from T0.6) as stubs for GDN/QSA/PLE internals so the whole graph constructs and runs on meta/CPU. Dummy-weights forward on CPU (tiny config) exercises control flow, `ParallelLMHead`/embedding tie, and kv-cache spec materialization end-to-end.
- **validation**: CPU: tiny random model loads dummy weights, forwards, samples a token; KV-group report matches T1.4 expectations; two fixed-seed greedy forwards produce identical tokens (determinism smoke, pre-stages device determinism); no CUDA/Triton import on 310P flag path.
- **status**: Not Completed
- **log**:
- **files edited/created**:

### TOBS: Run artifacts, memory reports and first-fatal-rank observability (R13)
- **depends_on**: [T0.5, T4.4]
- **location**: `vllm_ascend/observability/qwen38_runlog.py` + integration in `_310p/worker_310p.py`, logger
- **description**: Every server run writes machine-readable JSON + human log: revisions (T0.2), topology/rank identity, token limits, per-rank component memory, host PLE/pinned/RSS/swap/NUMA, chunk progress, tokens/s, PLE/QSA transfer+hit metrics, hybrid-state lifecycle events, first fatal error + rank (traceback preserved, not "worker died"). Long-prefill progress heartbeat for 1M runs. **Includes per-collective link-class tracing (R3): annotate HCCL collective traffic as within-card vs cross-card so D4 can compare Candidate A vs B collective cost by link.**
- **validation**: UT: inject synthetic failure in one rank → artifact names the rank and root exception; metrics sections present; collective trace entries carry a within/cross-card classification derived from D1 topology.
- **status**: Not Completed
- **log**:
- **files edited/created**:

### T8.1: Candidate A prototype — C8 QSA cache
- **depends_on**: [T6.4]
- **location**: `vllm_ascend/models/qwen4_exp/qsa_c8.py`, `tests/ut/qwen38_1m/test_qsa_c8.py`, spec hook in T1.4
- **description**: Signed-INT8 main K/V with documented per-? scale granularity; fused write/read quant paths in the eager impl; accuracy harness comparing C8-vs-BF16 selection sets and short-context logits; memory math for 1M (PRD table row "8-bit QSA K/V 6.00 GiB/chip").
- **validation**: host UTs: round-trip identity tolerances; selection-set agreement % vs BF16 on T0.4 short probes above a pre-declared bar; spec materializes in T1.4 UT.
- **status**: Not Completed
- **log**:
- **files edited/created**:

### T8.2: Candidate B prototype — QSA-aware DCP4
- **depends_on**: [T6.4, T0.5]
- **location**: `vllm_ascend/models/qsa_dcp/` (asc; may reuse `worker/dcp_utils.py`, `attention/context_parallel`), UTs
- **description**: Sequence-shard main QSA K/V across 4 ranks; indexer history stays replicated. Map selected global positions → (owner rank, local slot); exchange **only** selected rows (no all-gather of 1M cache); deterministic top-k and output reduction across ranks. Validate purely on host with 4-process `gloo` simulation using the T6.4 eager layers.
- **validation**: host multi-rank UT: 4-way DCP output equals single-rank BF16 reference within tolerance; transfer accounting proves only selected rows moved; determinism test across reruns.
- **status**: Not Completed
- **log**:
- **files edited/created**:

### T8.3: 1M allocation projection and candidate decision record
- **depends_on**: [T8.1, T8.2, T3.2]
- **location**: `docs/source/developer_guide/Design_Documents/qwen38_flash_next_1m_cache_decision.md` (asc)
- **description**: Combine manifest-derived model bytes (T3.2), cache layouts (T8.1/T8.2), workspace/fragmentation allowances into a per-chip projection table for both candidates at 1M; define exactly what D4 must measure and encode PRD §13 stop rules as acceptance formulas. **D4 (with hardware) is the sole decision authority — this task pre-decides nothing.**
- **validation**: doc review; arithmetic reproduces PRD §6 baselines; each candidate's measured-vs-projected acceptance formula explicit.
- **status**: Not Completed
- **log**:
- **files edited/created**:

## Device wave (run only when the 4-chip host is available; strictly serial D1→D8 except D9)

### D1: Probe + environment freeze on target
- **depends_on**: [T0.3, T0.2, hardware]
- **description**: Run hw_probe + env_freeze on target; verify ≈46 GiB actual free per chip (PRD open decision #1); record HCCL topology (within- vs cross-card). Update T8.3 projections with measured numbers.
- **validation**: probe JSON complete; per-chip free bytes recorded; go/no-go if any chip below plan by >X (decide in-log).
- **status**: Not Completed
- **log**:
- **files edited/created**:

### D1-MB: PLE transport micro-bench (device sub-task of D1)
- **depends_on**: [D1, T4.1]
- **location**: run from T4.1's micro-bench script on target
- **description**: On real 310P hardware, compare pinned-UVA vs mmap+registered-window row-gather for latency/throughput/page-faults at decode-realistic row rates; the winner is a D2 entry criterion logged into TOBS.
- **validation**: transport selected with numbers; if neither meets the decode floor (10 tok/s projection), escalate before D2.
- **status**: Not Completed
- **log**:
- **files edited/created**:

### D1.5: On-device component parity (G0 on 310P)
- **depends_on**: [D1, T3.3, T4.4, T5.1, T6.2]
- **location**: `tests/e2e/pull_request/one_card/` (asc), driven per rank on 4×310P
- **description**: Before any full-model real-weight boot, run tiny-config Qwen4Exp component slices on NPU against the T0.6 eager reference: W8A8 fused-MoE (real checkpoint scales for 2–4 experts extracted from shards), native GDN kernels, PLE gather+projection over the D1-MB-selected transport, QSA indexer + sparse attention. This is PRD gate G0 with real ops — isolates op-level mismatches from full-model failures.
- **validation**: per-component parity artifacts within pre-declared tolerances on device; any failure produces a bounded follow-up op task before D2, not a model-level debug session.
- **status**: Not Completed
- **log**:
- **files edited/created**:

### D2: G1 — real-weight startup, TP4, eager, 8K
- **depends_on**: [T1.5, T3.1, T3.2, T3.3, T4.4, T5.2, T6.4, T7.2, T0.4, TOBS, D1, D1-MB, D1.5]
- **description**: Serve real W8A8 checkpoint from `/workspace` (imports verified per R1). `/v1/models` 200 only after the readiness path has completed one real inference (R10 — startup without a successful request is not a pass); non-empty completion of 8K probe; per-rank weight report consumed (imbalance ≤5%); no loading interruption = resumable/retryable load per R13.
- **validation**: gate artifact: readiness probe returns 200 only after real inference path is ready; response text non-empty; accepted token count correct; weights-missing check clean; memory report attached.
- **status**: Not Completed
- **log**:
- **files edited/created**:

### D3: G2 — quantized correctness at 8K
- **depends_on**: [D2]
- **description**: Deterministic responses across restarts; quality delta vs frozen threshold from T0.1; TP4 (EP4 correctness variant only if enabled in production config — default off per R3).
- **validation**: threshold record + response artifacts; determinism across 2 restarts (temperature 0, fixed seed).
- **status**: Not Completed
- **log**:
- **files edited/created**:

### D4: Candidate benchmark at 128K + final decision
- **depends_on**: [D3, T8.1, T8.2, T8.3]
- **description**: With real weights, measure each prototype layout's per-chip post-allocation headroom and 128K prefill/decode; **D4 is the sole decision authority** (T8.3 supplies acceptance formulas only) selecting Candidate A vs B per PRD §8/§13; stop rules apply (A accuracy stop, B communication stop after one optimization pass).
- **validation**: decision record signed off with measured tables; chosen layout handed to T8.4 for integration before any further gate.
- **status**: Not Completed
- **log**:
- **files edited/created**:

### T8.4: Integrate chosen candidate (post-decision host task)
- **depends_on**: [D4]
- **location**: `vllm_ascend/models/qwen4_exp/model.py`/`qsa*.py`, T1.4 spec wiring; either `qsa_c8.py` (A) or `qsa_dcp/` (B) lands in the production path
- **description**: Merge the D4-selected candidate into the QSA layers and KV-cache specs (replacing the BF16 prototype path used at D2–D3), delete-or-flag the rejected prototype, and **re-run the entire host parity suite** (T3.x, T4.3, T5.2, T6.1–T6.4, T8.x UTs) on the merged configuration on host before D5. If B was selected, this is the first production import of DCP collectives — cover with the T8.2 gloo multi-rank UTs re-run post-merge.
- **validation**: full host suite green on merged config; allocation report re-issued through T0.5/T8.3 formulas.
- **status**: Not Completed
- **log**:
- **files edited/created**:

### D5: G3 — 128K single request (post-choice)
- **depends_on**: [D4, T8.4]
- **description**: 128K corpus input at concurrency 1: full prompt accepted (no truncation), completion, full memory report; chunked prefill telemetry per TOBS.
- **validation**: gate artifact + accepted-token equality.
- **status**: Not Completed
- **log**:
- **files edited/created**:

### D6: G4 — native 262,144 at concurrency 1
- **depends_on**: [D5]
- **description**: Retrieval suite at 262K: no truncation, no preemption; measured ≥8 GiB free per chip after model+persistent cache (graph capture supersedes if eager-only, document). Establish real 310P prefill floor (may revise PRD §8.2 via recorded decision).
- **validation**: retrieval records correct; headroom measurement from allocator, not estimate.
- **status**: Not Completed
- **log**:
- **files edited/created**:

### D7: G5 — 524,288
- **depends_on**: [D6, T7.1 verified on device]
- **description**: Position correctness against reference probe (T7.1 tables on device), retrieval pass, cache strategy within budget.
- **validation**: gate artifact; no positional drift symptoms (e.g., retrieval near boundary positions).
- **status**: Not Completed
- **log**:
- **files edited/created**:

### D8: G6 — 1,048,060 input + 512 output reserve, concurrency 1
- **depends_on**: [D7, T7.2, T0.4]
- **description**: The million-token request: 8/8 distributed retrieval records correct, adversarial no-answer rejects retrieval, TTFT within ceiling (900 s initial; revise via recorded decision only), decode ≥10 tok/s floor, no sustained swap, no per-token transfer of full QSA/PLE history.
- **validation**: full TOBS artifact bundle; explicit statement of effective retained context (PRD §13 anti-claims).
- **status**: Not Completed
- **log**:
- **files edited/created**:

### D9: Chunked-prefill / workspace tuning
- **depends_on**: [D6]
- **description**: Sweep chunk sizes (from 4096), bounded workspaces, PLE prefetch window, transfer batching; re-measure D6 floor; feed back before D7/D8 attempts (re-run any regressed gate).
- **validation**: tuning report with best config frozen in deployment metadata.
- **status**: Not Completed
- **log**:
- **files edited/created**:

## Follow-on stubs (explicitly out of scope; each needs its own gated plan)

- **S1** Hybrid prefix caching (WP11): cache-key binding, all-state save/restore at consistent boundary, measured pool sizing — after D8.
- **S2** MTP fixed-step speculative decoding — after D8.
- **S3** Multimodal real-image request — after D8.
- **S4** ACLGraph capture attempt for QSA/PLE/dynamic lookup or documented evidence to stay eager — after D8.
- **S5** EP4 + FlashComm1 measured gate — after D8.
- **S6** Docs: model tutorial with exact launch command, feature matrix update, runbook, final report (PRD §14) — after D6 (262K valuable-release fallback).

## Parallel Execution Groups (host wave; hardware wave is serial)

| Wave | Tasks | Can Start When |
|------|-------|----------------|
| 1 | T0.1, T0.2, T0.3, T0.4, T0.5, T0.6, T1.1, T7.1 | Immediately |
| 2 | T1.2 | Immediately (independent package) |
| 3 | T1.3, T1.4, T3.1, T4.2, T7.2 | T0.1/T1.2 deps met |
| 4 | T1.5, T3.2, T3.3, T4.1, T5.1, T6.1 | Wave 3 partial |
| 5 | T4.3, T5.2, T6.2 | Wave 4 partial |
| 6 | T4.4, T6.3 | T4.3 / T6.2 done |
| 7 | T6.4, TOBS | Wave 6 |
| 8 | T8.1, T8.2 | T6.4 |
| 9 | T8.3 | T8.1, T8.2 |
| DEV | D1 (D1-MB) → D1.5 → D2 → D3 → D4 → T8.4 → D5 → D6 (→D9) → D7 → D8 | All host waves complete + hardware |

## Testing Strategy

- Host-side only before D1: `pytest -sv tests/ut/qwen38_1m/...` per AGENTS.md; no test may require an NPU before the device wave (CI-safe). Parity tolerances committed in-source before final comparisons (PRD §8.1).
- Import-hygiene gate: a UT asserts `import vllm_ascend.models.qwen4_exp` under a faked non-CUDA platform never loads `triton`/`cuda` symbols (310P constraint R2).
- Determinism: bitwise-rerun tests for indexer top-k, selected-row sort order and DCP reduction order (T8.2).
- Device gates use real weights only for G1+ status; dummy weights labeled separately (never a support claim, PRD §13).
- Every gate attaches a machine-readable artifact (TOBS) + human log; failed 1M attempt must not invalidate lower-context results (isolate run dirs).
- Lint/format before commit: `ruff check`, `ruff format`, `bash format.sh ci` in both repos; commits `git commit -s`, Conventional Commits (`feat(qwen4exp)…`, `feat(310p)…`), PR titles `[Feat][Model] …`.

## Risks & Mitigations

- **ModelSlim checkpoint lands late/corrupt (T0.1)**: all real-weight tasks (T3.x, T4.2, D2+) block; host UTs run against manifest schema + synthetic tensors meanwhile. Keep manifest-format contract frozen early.
- **`is_uva_available`/UVA-like mapping absent on 310P pinned host path (T4.1)**: shared-mmap + registered-window gather fallback already sketched; measure in D1 as first micro-experiment after probe.
- **CircularBufferSpec/MLAAttentionSpec not materializable without scheduler changes on 310P (T1.4)**: earliest-risk task — schedule T1.4 first in Wave 3; if block-size LCM collides with 310P attention block granularity, adjust in `patch_kv_cache_utils.py` and document deviation.
- **GDN numeric parity failures on 310P fla ops**: T0.6 tolerances pre-declared; if fla kernels can't meet Qwen4Exp shapes, write dedicated 310P kernel task before D2 (flag `files edited` includes new `_310p/ops/fla/gdn_310.py` entry).
- **Host memory 256 GB insufficient during prototype runs**: T4.1/T0.5 fail-fast host accounting; never pin full table; cap prototypes to sliced tables for host UTs.
- **Hardware arrives with <8 GiB headroom after BF16 (PRD risk)**: candidate decision D4 already forced before 262K; fallback C requires its own design task spawned only on both-candidate failure (PRD §13).
- **Triton transitive imports from upstream layers (`QwenGatedDeltaNetAttention`, quant utils)**: import-hygiene UT in T1.2/T1.5 catches at CI time, not device time; patches mirror `patch_triton.py`/`patch_idex_310.py` precedent.

## Open decisions to resolve in task logs

1. Actual free per-chip bytes (D1).
2. C8 accuracy viability with BF16 index keys (T8.1/D4).
3. DCP row-exchange vs C8 dequant cost on real topology (D4).
4. PLE physical sharing vs row-partitioned owners (T4.1/D1).
5. Authoritative 1M RoPE parameter set (T7.1). **RESOLVED (T7.1, 2026-09-15)**: yarn, theta=10_000_000.0, factor=4.0, orig_max_pos=262_144, beta_fast=32, beta_slow=1, mscale≈1.13863, partial_rotary_factor=0.25 → 1_048_576.
6. Frozen G2 quality threshold (T0.1).
7. Supported CANN/vllm-ascend deployment baseline (T0.2).
