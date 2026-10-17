# MiniMax-M3 310P offline runtime candidate

Prepared October 16, 2026. This implements the offline runtime work following
the [checkpoint assessment](minimax_m3_310p_offline_preparation.md). Qwen was
preserved: no NPU processes, service changes, runtime installations, or device
tests were performed. This is an **offline-tested implementation candidate**,
not a hardware-qualified model deployment.

## Implementation

The existing MiniMax-M3 model and multimodal registration are reused. The new
310P branch replaces routed `FusedMoE` allocation with resident signed-W4
buffers. Other Ascend device families keep their existing M3 paths.

- **Experts:** 32x32 blocks, signed low-nibble-first W4 codes, stored FP16 scales.
  Gate/up/down remain separate, matching `w1`/`w3`/`w2`. Native execution uses
  the existing `npu_w2_grouped_blocked_dequant_matmul_310` operator. Scales are
  expanded to FP32 once per forward for its ABI; weights remain packed.
  The 6144/3072 dimensions satisfy the source tiler's W4 shape constraints.
  This checks the source contract, not an installed binary or OPP package.
- **Routing:** FP32 sigmoid, correction bias for selection only, normalized
  uncorrected selected scores, and routed output scale 2. Stage-local EP owns
  contiguous complete experts. Device sorting/counts need no Python readback
  of expert IDs. Calls are capped at 128 routed rows, with longer inputs split
  into chunks. Empty and peer-owned groups contribute zero.
- **Activation:** clipped SwiGLU-OAI, alpha 1.702, beta 1, limit 7. The gate has
  only an upper clip; up has both bounds. Shared/dense MLPs use the existing
  pure tensor activation branch on 310P. Routed and partial shared output are
  combined before one TP reduction, with the result returned in FP16.
- **Sparse attention:** eager tensor implementation with 16-query workspace
  tiles. Each index head independently selects blocks for its KV head. The
  path preserves causal max scoring, forced initial/local blocks, GQA,
  partially filled pages, cached prefill, and scheduler padding. Main and index
  block tables can have different physical page assignments. Invalid cache
  contents are masked before arithmetic, including uninitialized NaNs.
- **Cache plumbing:** M3 gets ordinary ND main K/V pages and one ND index cache.
  A small backend-name dispatch in the 310P allocator preserves its existing
  private NZ allocation for other attention backends. CPU query boundaries
  come from scheduler metadata; sequence lengths stay on device.
- **Loading:** `--load-format minimax_m3_w4` verifies format/config/index hashes,
  complete source-key and expert coverage, selected file hashes, tensor
  metadata, positive finite scales, and all named dense parameters. It reads
  only local routed experts and local language layers. The existing model
  handles dense QKV/indexer/shared-MLP mappings and PP hidden/residual transport.

The runtime rejects BF16, graph execution, additional quantization, MTP, LoRA,
EPLB, sequence-parallel MoE, DP/DCP/PCP greater than one, incompatible TP, and
missing EP for TP greater than one. It requires the explicit export marker,
FP16 cache, and block size 128. CPU expert projection is a test oracle; NPU
execution fails if the grouped native operator is unavailable rather than
unpacking the full expert bank.

The architectural review surface is the existing M3 model, backend metadata
selection, and the focused 310P cache allocator branch. No new model runner
forward behavior, monkey patch, environment variable, or model architecture was
introduced. A future native MSA kernel can replace the portable arithmetic
without changing the export or routing contract.

## Offline evidence

Run the focused suite from the repository root:

```bash
python -m pytest --noconftest -q tests/ut/minimax_m3
```

The suite includes independent scalar attention and expert arithmetic
references; causal pruning; per-head selection; unequal main/index page
assignments; cached prefixes; padding with poisoned cache contents; EP sum
parity; chunk boundaries; resumable conversion; corrupt/missing payloads;
source mutation; overflow; PP filtering; strict dense loading; and unsupported
configuration rejection. Integration tests compile the production adapter and
allocator method bodies with CPU runtime stand-ins. These tests do **not**
instantiate the complete vLLM engine or exercise hardware collectives.
All **47 focused tests passed**, as did all applicable changed-file manual
pre-commit checks, including Python formatting and Markdown lint.

The broader existing `tests/ut/_310p/test_kv_cache_sharing.py` could not collect
against the host's installed vLLM: `vllm.v1.kv_cache_layout` is absent. The
collection failure is retained in `broader-cache-test-blocked.txt`. No package
upgrade or environment replacement was made. Full engine testing needs the
matching runtime source snapshot as well as `torch_npu` and hardware.

The real checkpoint gate exported layer 3, expert 0, all three matrices, using
MSE W4 selection. It verified six packed buffers and executed the full routed
expert MLP on two seeded random FP16 vectors of width 6144. The output was
finite. Output cosine against the source expert was **0.9858523**, with
**0.1687741 NRMSE**. The CPU oracle rounds reconstructed weights and projection
outputs to FP16; it cannot establish bitwise Cube accumulation parity.
These are random-input expert statistics, not activation calibration, task
accuracy, or full-model output quality.

Reproduce the bounded real-weight gate:

```bash
python -m tools.minimax_m3.smoke_expert \
  /run/media/matteius/20TB-drive/models/MiniMax-M3 \
  /run/media/matteius/20TB-drive/models/MiniMax-M3-W4-offline-probe \
  --report artifacts/minimax-m3-runtime-offline-20261016/real-expert-roundtrip.json
```

The probe directory deliberately has no runnable checkpoint manifest. The
[receipt](../../../../artifacts/minimax-m3-runtime-offline-20261016/real-expert-roundtrip.json)
records source tensor hashes, exported file hashes, geometry, and results.
Test and lint logs are in the same artifact directory.

## Full export and runtime preparation

The full 427B checkpoint was **not converted** during runtime implementation.
The exporter and complete miniature export/load/inference path were tested;
the real-weight gate is limited to one expert. Full conversion is a substantial
CPU/disk job and writes about 219.14 GiB of unique weight payload plus metadata.
Run this separately when that artifact is needed:

```bash
python -m tools.minimax_m3.export_w4 \
  /run/media/matteius/20TB-drive/models/MiniMax-M3 \
  /run/media/matteius/20TB-drive/models/MiniMax-M3-W4 \
  --method mse --threads 4 --chunk-rows 256

python -m tools.minimax_m3.runtime_plan \
  /run/media/matteius/20TB-drive/models/MiniMax-M3-W4
```

Conversion streams rows, including embeddings, instead of holding a full
checkpoint or expert bank in RAM. Per-tensor receipts support restart with the
same command. Resume verifies completed output hashes and requires unchanged
source config/index, shard sizes, and modification times. It does not establish
official upstream checkpoint provenance. Nonfinite or overflowing conversions
fail. Config marking and the final manifest happen only after all source keys
are exported. Do not modify source shards during conversion.

`runtime_plan` validates the manifest and **prints only**; it never launches a
process. Its candidate command uses TP2 x PP3, stage-local EP2, partition
`21,20,19`, eager execution, one sequence, 8K context, 128-token prefill chunks,
85% memory utilization, and text-only admission. The existing multimodal
wrapper still constructs/loads vision on each stage, even with image admission
disabled. Budget this replication; do not treat the earlier ideal per-rank
weight estimates as measured device use.

Before a future authorized hardware window, install this complete source
snapshot into an isolated runtime, resolve the `/vllm-workspace/*` and
`/workspace` paths, and verify the grouped operator and matching 310P OPP.
The offline host has no `torch_npu`; no dependencies were upgraded. The command
is a candidate, not a claim of successful startup:

```bash
cd /workspace
VLLM_PP_LAYER_PARTITION=21,20,19 vllm serve /models/MiniMax-M3-W4 \
  --served-model-name minimax-m3-w4 --load-format minimax_m3_w4 \
  --dtype half --tensor-parallel-size 2 --pipeline-parallel-size 3 \
  --enable-expert-parallel --enforce-eager --block-size 128 \
  --max-model-len 8192 --max-num-seqs 1 --max-num-batched-tokens 128 \
  --gpu-memory-utilization 0.85 --limit-mm-per-prompt '{"image":0,"video":0}' \
  --port 8000
```

Use a separate authorized hardware window before executing this command or
replacing Qwen. Check dummy operator execution first if useful, then the full
real export. Readiness alone is insufficient: request `/v1/models`, issue a
short completion, then a prompt exceeding one prefill chunk. Verify finite,
nonempty output, PP/EP correctness, memory usage, and reference quality.

## Feature and validation status

| Area | Status |
| --- | --- |
| W4 exporter, local loader, routing, clipped activation | CPU tested, including one real expert |
| Eager paged MSA | CPU reference and adapter wiring tested |
| TP2 x PP3 / EP2 | CPU partition/reduction checks; hardware collectives untested |
| Native grouped W4 operator | Reused existing source contract; M3 device invocation untested |
| Dummy / full real-weight server | Neither run under offline constraint |
| Text, image, video API output | Not validated; initial command admits text only |
| Vision placement | Existing replicated wrapper retained; FP16 device path untested |
| ACLGraph / flashcomm1 | Excluded from this eager runtime contract |
| MTP | Rejected; no draft tensors established in the checkpoint |
| LoRA / EPLB / context parallelism | Rejected |
| Throughput, latency, full-model accuracy | No measurements |
| 128K x 16 | Not attempted; approximately 297 GiB of FP16 cache alone |

No accuracy-metric E2E YAML is supplied because no full-model accuracy was
measured. The model-adapter skill's hardware and real-server sign-off gates
remain pending; the user's offline-only requirement takes precedence for this
implementation milestone.
