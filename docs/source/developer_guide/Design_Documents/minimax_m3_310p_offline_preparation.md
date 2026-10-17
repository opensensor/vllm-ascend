# MiniMax-M3 on six 310P3 devices: offline preparation

Prepared October 16, 2026. The user selected MiniMax-M3 and explicitly requested
preserving the running Qwen service while preparing offline. This work imports
no NPU runtime, changes no serving code, and starts no hardware processes.
It is a preparation milestone, not a working model adaptation.

## Source checkpoint and integrity scope

Source: `/run/media/matteius/20TB-drive/models/MiniMax-M3`.
The local checkpoint has 59 indexed safetensors shards and 23,416 tensors.
Every indexed tensor exists in the specified shard. Header shapes, dtype byte
counts, contiguous offsets and complete file lengths pass the audit. All 57
routed layers contain 128 experts with all three expected projections.

Actual tensor payload is **854,172,958,720 bytes (795.51 GiB)**. The index claims
869,157,697,024 bytes. The audit records this discrepancy and uses header-derived
sizes; it does not rewrite the source index. This is not proof of payload
integrity or identity with an official Hub revision. Full-file checksums and
upstream provenance remain to be established before a production export.
The local config contains `unsloth_fixed: true`; do not infer provenance solely
from the directory name or its README.

The [audit receipt](../../../../artifacts/minimax-m3-offline-20261016/checkpoint-audit.json)
pins the local config/index hashes and contains complete shard accounting.

## Architecture and implementation already present

- Registered architecture: `MiniMaxM3SparseForConditionalGeneration`.
- 60 decoder layers, hidden width 6144; the first three layers have dense MLPs.
- 57 routed layers: 128 experts, top-4, expert intermediate width 3072,
  one shared expert, sigmoid routing with correction bias and output scale 2.
- 64 query heads, four KV heads, head dimension 128; per-head Q/K norms,
  partial rotary dimension 64, Gemma RMSNorm.
- MSA on 57 layers: 128-token blocks, top-16 blocks, four index query heads,
  index dimension 128, max scoring, one forced local block, no initial blocks.
- Clipped SwiGLU-OAI: alpha 1.702, limit 7, beta defaults to 1. Gate clipping
  has only an upper bound; up is clipped to both bounds. Ordinary SiLU times
  up is not equivalent.
- Vision tower: 32 layers, width 1280, 16 heads, temporal/spatial patch merging.
- Config context ceiling: 1,048,576 tokens. No practical context is validated.

The fork already registers both M3 language and multimodal classes in
`vllm_ascend/models/__init__.py`. Reuse the existing implementations in
`vllm_ascend/models/minimax_m3/`; no new standalone model class is needed.
The existing four-card E2E test targets A3 W8A8 with five dummy layers. It is
not 310P or complete-real-checkpoint evidence.

The local upstream checkout is `/run/media/matteius/20TB-drive/vllm`.
The model-adapter skill's Docker roots `/vllm-workspace/*` and `/workspace`
are absent on this offline host. Deployment must resolve actual runtime roots
and installed imports later; no dependency or transformers upgrade was made.

## Candidate memory and topology

The first storage candidate is **signed W4 routed experts**, with one FP16 scale
per 32x32 block. BF16 non-expert weights would become FP16; F32 tensors remain
F32. This reuses the existing packed integer format, not NVFP4 encoding.

| Component | Parameters | Candidate unique storage |
| --- | ---: | ---: |
| Routed experts | 413.122B | 193.13 GiB |
| Other weights, including shared experts and language heads | 13.287B | 24.83 GiB |
| Vision tower | 0.631B | 1.18 GiB |
| Total | 427.040B | **219.14 GiB** |

These are logical unique bytes, excluding TP replication, NZ padding, cache,
activations, graphs, communication buffers and allocator overhead. They are
not a measured device allocation or a guarantee of fitting. The six 310P3
memory domains must be budgeted individually using their actual free bytes.

**TP6 fails existing head divisibility checks.** TP2 x PP3 is the initial
dimension-compatible topology; EP can remain within each TP2 stage (64 experts
per rank for that stage's layers). A global EP6 implementation would need
uneven expert ownership or explicit padding because 128 is not divisible by 6,
as well as independent attention/EP groups and correct dispatch semantics.

A scan balancing unique weight bytes suggests layer boundaries `[0,21,41,60]`:

| Stage | Decoder layers | Unique stage bytes | Ideal half per TP worker |
| --- | --- | ---: | ---: |
| 0 | 0 through 20, embeddings and vision | 72.42 GiB | 36.21 GiB |
| 1 | 21 through 40 | 74.06 GiB | 37.03 GiB |
| 2 | 41 through 59, final norm and head | 72.65 GiB | 36.32 GiB |

The final column is a lower bound, not a rank allocation: routers, norms,
index-K projection and possibly vision are replicated. Balance actual latency
and memory after measuring. Three pipeline stages execute sequentially for
a single token; this topology is a capacity starting point, not a speed claim.

MSA sparsity does **not** eliminate full KV storage in the current backend.
With FP16 K/V and index-K, TP2 x PP3 needs approximately **148.5 KiB per token
across all stages**: 120 KiB main KV plus 28.5 KiB replicated index-K.
That is 18.56 GiB for one 128K sequence, before block rounding and metadata.
Sixteen full 128K sequences need 297 GiB of cache alone, so the skill's default
128K/batch-16 validation target is not an appropriate first capacity gate.
Start with 8K and one sequence, then profile 32K/128K independently.

## Confirmed implementation gaps

1. **MSA operators:** `msa_index_score_def.cpp` and
   `sparse_attention_score_def.cpp` register 910B/910_93/950, with no 310P
   registration. The current adapter calls these operators and a Triton top-k
   path. Generic custom-operator capability does not establish 310P support.
   Implement or adapt a 310P score/top-k/selected-block attention path, using
   the existing M3 semantics as the oracle. Preserve causal masking, forced
   local blocks, partial final blocks and per-index-head selection.
2. **Packed MoE execution:** reuse the signed W4 projection layout only after
   validating the actual `[3072,6144]` and `[6144,3072]` matrices. Existing
   DeepSeek/GLM packed wrappers apply ordinary SiLU, whereas M3 requires
   SwiGLU-OAI. Route correction bias affects selection; preserve the model's
   selected-score normalization and output scaling instead of substituting
   Qwen routing.
3. **Precision:** BF16-to-FP16 conversion and Gemma norm behavior need layerwise
   reference checks. Preserve F32 router inputs/weights and explicit rounding
   boundaries; weight reconstruction alone cannot establish stability.
4. **Loader and PP:** add explicit packed-weight mapping to the existing model
   seams. Map `w1` to gate, `w3` to up and `w2` to down. Stream by tensor/chunk,
   preserve private CPU upload staging, and qualify stage-local loading and
   the existing hidden/residual transport on this model.
5. **Multimodal:** reuse the upstream common processor and vision model, then
   validate FP16 vision execution, preprocessing and stage placement.
6. **MTP:** config advertises one next-token layer and seven MTP modules, but
   the checkpoint has only main decoder layers 0 through 59 and no tensors
   named for MTP/draft modules. Treat MTP as unestablished, not enabled by a
   launcher flag. Eagle3 support in the class is not a supplied draft model.

## Offline checks and next gates

The tools in `tools/minimax_m3/` provide a header audit and a bounded CPU W4
experiment. The experiment reads expert 0 from layers 3, 31 and 59, all three
projections, in 256-row chunks. It hashes each source matrix and its candidate
packed codes/scales and measures reconstruction against the original weights.
Packing is checked exactly; quantization is deliberately lossy.

The minmax experiment produced weight cosine 0.98889 through 0.99168 and
NRMSE 0.12927 through 0.15007 across nine matrices. These are weight statistics,
not task accuracy or hardware qualification. The MSE-scale experiment improved
NRMSE on all nine matrices, with cosine 0.99251 through 0.99383 and NRMSE
0.11096 through 0.12232. Both runs hashed identical source matrix bytes.
The receipts are [minmax](../../../../artifacts/minimax-m3-offline-20261016/w4-minmax-probe.json)
and [MSE](../../../../artifacts/minimax-m3-offline-20261016/w4-mse-probe.json).
MSE is the initial candidate for subsequent activation-based evaluation;
neither experiment performs activation calibration.

1. Establish source provenance and select/calibrate the W4 recipe using real
   activations, including shared/dense precision checks.
2. Qualify one sparse M3 layer and one routed layer against a CPU/GPU reference,
   including routing, SwiGLU-OAI, and cache updates. Prepare 310P adapters through
   existing model/operator seams with defaults unchanged for other devices.
3. Export stage-local W4 banks with source/format hashes and complete coverage
   checks. A packed format probe is not a runnable exported checkpoint.
4. After the user authorizes a hardware window, run dummy operator/API gates,
   followed by complete real-weight 8K text and image requests. Preserve the
   Qwen restore path and verify device import roots before launch.
5. Qualify PP/EP, then graphs, then larger context and concurrency. Evaluate
   task accuracy against the source before claiming useful adaptation.

| Feature | Status for this preparation |
| --- | --- |
| Checkpoint structure / W4 CPU probe | Offline checks only |
| Dummy serve / real-weight serve | Not attempted; Qwen preserved |
| 310P MSA | Bundled operator registrations do not cover 310P |
| TP2 x PP3 and stage-local EP | Dimension candidate; not hardware validated |
| ACLGraph / flashcomm1 | Not attempted; dependent on working operator path |
| MTP | Config flags present; draft checkpoint not established |
| Text / image / video inference | Not validated |
| 128K x 16 | Cache-only budget exceeds available aggregate device memory |

## Reproduce offline

Run from the repository root. These commands neither import `torch_npu` nor
connect to the serving host.

```bash
python -m tools.minimax_m3.audit_checkpoint \
  /run/media/matteius/20TB-drive/models/MiniMax-M3 \
  --output artifacts/minimax-m3-offline-20261016/checkpoint-audit.json

python -m tools.minimax_m3.probe_w4 \
  /run/media/matteius/20TB-drive/models/MiniMax-M3 \
  --method minmax \
  --output artifacts/minimax-m3-offline-20261016/w4-minmax-probe.json

python -m tools.minimax_m3.probe_w4 \
  /run/media/matteius/20TB-drive/models/MiniMax-M3 \
  --method mse \
  --output artifacts/minimax-m3-offline-20261016/w4-mse-probe.json

python -m pytest --noconftest tests/ut/minimax_m3/test_offline_preparation.py -q
```

`--noconftest` isolates these CPU-only tools from the suite's global NPU/vLLM
mock setup. This report deliberately provides no claimed working serve command
or invented accuracy threshold. Full export, runtime implementation and model
validation remain outstanding.

Validation: all 15 focused CPU tests passed, and all applicable pre-commit
manual-stage checks passed on the changed files, including markdown lint.
Receipts: [tests](../../../../artifacts/minimax-m3-offline-20261016/unit-tests.txt)
and [lint](../../../../artifacts/minimax-m3-offline-20261016/lint.txt).
The full runtime/NPU suite was not run under the offline-only constraint.
