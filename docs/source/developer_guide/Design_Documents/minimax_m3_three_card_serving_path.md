# MiniMax-M3 on three Ascend 310P Duo cards

October 17, 2026. This is an offline serving pathway, not a deployed or
hardware-qualified service. Qwen and all six NPUs were left untouched.
The complete local W4 export contains 23,416 tensors and 219.14 GiB of weight
files; its final manifest is complete. No second full export is needed for
the first serving attempt.

## Recommended first service

Use **TP2 x PP3, stage-local EP2**, FP16 activations, existing signed block-W4
experts, eager execution, one request, 8K context and 128-token prefill chunks.
Map ranks `[0,1]`, `[2,3]`, `[4,5]` to the three physical cards, after confirming
the device mapping in the deployment window. These are six separate memory
domains, not a single pool. TP6 fails M3 head divisibility; global EP6 also
requires uneven ownership of 128 experts and a different dispatch design.

| Card / stage | Decoder layers | Routed experts per device per sparse layer | Placement |
| --- | --- | ---: | --- |
| 0 | 0–20 | 64 | Embedding, three dense layers, 18 MoE layers |
| 1 | 21–40 | 64 | 20 MoE layers |
| 2 | 41–59 | 64 | 19 MoE layers, final norm and output head |

Both devices in a stage receive the same hidden states. Each evaluates its
local routed experts and a TP shard of the shared expert. Their outputs are
summed once. This path needs no global expert all-to-all. Attention projections
are TP-sharded; index-K and routers are replicated. Hidden states and residuals
cross the two pipeline boundaries. The existing 310P transport uses adjacent
pairwise AllGather, avoiding unsupported direct HCCL Send/Recv assumptions.
The matching upstream coordinator must retain its `supports_tensor_dict`
integration; a generic upstream installation is insufficient.

For one decoding request, the three stages execute sequentially. This is a
capacity-first topology, with no measured tokens/s claim. TP1 x PP6 is a
possible fallback, but adds pipeline boundaries and gives each stage only one
chip; it is not the preferred starting point.

## Per-device memory gate

The new `tools.minimax_m3.serving_budget` reads the completed manifest, accounts
for replicated router/norm/index-K tensors, and exhaustively compares all
contiguous three-stage partitions. It conservatively charges the entire
vision tower and projectors to every rank, even though TP vision can shard
some weights. It does not instantiate the model or hash 219 GiB of payloads.
Payload hashes remain checked by the loader. Unknown tensor placements fail.

The best memory partition remains **21,20,19**:

| Stage | Weights per device, GiB | 8K KV per device, GiB | With provisional 3 GiB runtime allowance |
| --- | ---: | ---: | ---: |
| 0 | 37.06 | 0.199 | 40.26 |
| 1 | 38.69 | 0.195 | 41.89 |
| 2 | 37.98 | 0.186 | 41.17 |

These are estimates, not measured allocations. The 3 GiB allowance covers
workspaces, temporary tensors, communication, alignment and allocator costs;
profiling must establish whether it is sufficient. Peak loader memory also
needs a separate check. CPU reads are bounded by a tensor, but each worker
must avoid materializing its full checkpoint on the host.

At an **assumed 45 GiB reported device capacity**, a 95% budget is 42.75 GiB:
the busiest rank has about 0.86 GiB beyond the allowance at 8K. At 44 GiB and
95%, it misses the allowance by about 88 MiB. This is why actual per-device
capacity matters. The old 85% candidate is too restrictive for this budget;
`runtime_plan` now prints 95% with an explicit capacity warning. This is not
permission to take memory from Qwen: the hardware gate requires a separate
service window.

With the same allowance, one 32K request needs at most 42.47 GiB per device,
and one 128K request needs 44.82 GiB. Neither is a startup default. Sixteen
128K requests require about 297 GiB of KV alone. Sparse selection reduces
attention computation, not the allocated full-context KV/index cache.

Reproduce the ledger, substituting a measured capacity when available:

```bash
python -m tools.minimax_m3.serving_budget /models/MiniMax-M3-W4 \
  --device-gib 45 --utilization 0.95 --context 8192 --sequences 1 \
  --reserve-gib 3 --output /tmp/m3-budget.json
```

If the budget does not fit, first remove unnecessary vision construction for
a separately tested text-only variant, or qualify a smaller runtime reserve
through measurement. Reducing context buys little at 8K. Repartitioning was
already searched. Quantizing dense/shared weights is a further option requiring
new kernels/loading and quality gates. Two-bit expert weights are not a
prerequisite for this pathway.

## What Huawei ModelSlim contributes

The local ModelSlim source at revision
`01d1bf92e088b70acc807c3b98391106d460cbfa` already contains:

- `msmodelslim/model/minimax_m3/model_adapter.py`: layer-wise loading,
  unstacking MoE weights, smoothing interfaces and save-name remapping.
- `lab_practice/minimax_m3/minimax_m3_w8a8.yaml`: dynamic W8A8 with iterative
  smoothing and exclusions for router, indexer, norms, heads and vision.
- CPU INT4 quantization IR, usable without `torch_npu` or a full model load.

The provided M3 W8A8 recipe cannot solve capacity here: routed experts alone
would occupy over 384 GiB at one byte per parameter. It is useful as an adapter
and calibration starting point, not the proposed serving format.

A new isolated CPU probe exercised ModelSlim's symmetric INT4 IR on **nine
real 64-row slices**, covering w1/w2/w3 of expert 0 in layers 3, 31 and 59.
It reshapes each 32x32 weight block into a synthetic row, selects a symmetric
scale, rounds scales to FP16, then packs signed nibbles for our runtime.
All packing roundtrips were exact. Weight NRMSE was **0.146–0.168**, versus
**0.111–0.123** for the existing MSE export on those same slices. ModelSlim's
plain min/max was worse in every sample. This is weight reconstruction, not
activation calibration or language quality; it does not evaluate ModelSlim's
calibrated methods. The current export remains the first-service baseline.

```bash
python -m tools.minimax_m3.modelslim_probe \
  /models/MiniMax-M3 /models/MiniMax-M3-W4 \
  --modelslim-root /path/to/msmodelslim --output /tmp/m3-modelslim.json
```

For a subsequent quality pass, use the original BF16 source and a separate
candidate output. Collect representative text activations layer by layer,
including routed expert coverage and shared-expert outputs. Compare baseline
MSE with a calibrated block32 method or an explicitly supported alternative
format. Keep router and sparse indexer in their existing precision initially.
Clipped SwiGLU-OAI and Gemma `1 + weight` norms must survive any smoothing or
rescaling transformation; verify full-layer outputs before accepting it.

Standard asymmetric per-group ModelSlim W4 has scales/offsets and a grouping
layout different from our symmetric 32x32 format. It cannot be converted by
renaming metadata or merely repacking bytes. Either constrain the quantizer
to the existing block contract, as the probe demonstrates for min/max, or add
and qualify a separate loader/operator path. ModelSlim calibration is not a
replacement for the M3 sparse attention runtime or six-device communication.

The local host lacks both installed `msmodelslim` and Transformers' native
`minimax_m3_vl` model module. CPU IR was imported directly from the checkout
in an isolated process. The full adapter is not ready in this environment.
Use a separate matching reference environment or vendored compatible model
code; do not upgrade the Qwen environment. No dependencies were changed.

## Offline validation and remaining gates

The six-process CPU probe uses real Gloo groups with the production
`PackedM3Experts` arithmetic and `PipelineAllGatherTransport` packet methods.
It checks TP2 expert sums and PP3 hidden/residual transfers against a serial
three-stage reference for 1, 33 and 129 tokens. All six ranks passed; final
maximum absolute FP16 difference was 0.0009765625. Small synthetic weights
exercise routing/chunk boundaries; this is not full-model inference.

The CPU harness precreates pair groups in a world-consistent order. Direct
concurrent Gloo creation using the production local-group bootstrap timed out
in the first attempt. Consequently, **HCCL bootstrap/order remains a separate
hardware gate**; the CPU result validates packet exchange and math only.

```bash
python -m tools.minimax_m3.distributed_cpu_probe --output /tmp/m3-six-rank
python -m pytest --noconftest -q tests/ut/minimax_m3
python -m tools.minimax_m3.runtime_plan /models/MiniMax-M3-W4
```

The focused suite passed 78 tests, with four optional ModelSlim tests skipped
in the default environment. Those four passed in the isolated checkout-based
process. Changed-file manual pre-commit checks passed, including Markdown lint.

The implementation and printed command require a coherent runtime snapshot:
this Ascend plugin, the local upstream vLLM revision
`3ab5dda29acabea01f6a63d0806bdbbb4a27bde5` including M3 common
vision/config helpers and pipeline integration, and the matching 310P CANN,
Torch NPU, extension and OPP package. The local host lacks Torch NPU, and its
installed vLLM previously failed broader cache tests due to missing
`vllm.v1.kv_cache_layout`. Do not substitute the host installation for that
runtime. The upstream M3 config registry handles the exported model type;
its source-specific `auto_map` is not a reason to enable arbitrary remote code.

Before deployment, verify the exact grouped operator
`npu_w2_grouped_blocked_dequant_matmul_310` against `[64,3072,3072]` packed
w1/w3 banks and `[64,6144,1536]` packed w2 banks, FP16 inputs, FP32 block
scales and INT64 group ends. Current source tiling accepts these dimensions;
installed binary and numerical behavior remain untested. Check zero-local
routes and 128-route calls, and preserve the single FP32 reduction followed
by FP16 narrowing. The existing sparse-attention CPU reference is the oracle
for cached-prefix, per-head selection and partially filled 128-token pages.

Ordered deployment gates in a later hardware window:

1. Confirm physical rank pairing and per-device free/total bytes. Transfer the
   completed export and validate destination hashes. Prepare an isolated
   runtime and coherent OPP without modifying the Qwen installation.
2. Run bounded real-matrix W4 operator and sparse-attention checks. Then verify
   TP2 reductions and PP3 group initialization/packet transfer on all six ranks.
3. Start the full real-weight service at 8K/one request, measure every rank's
   load peak and steady state, and replace the provisional memory allowance
   with measurements. Dummy startup alone cannot pass this gate.
4. Require `/v1/models`, a short generation, and a prompt longer than 128 tokens
   to pass. Compare deterministic outputs/logits with the reference, then run
   held-out quality evaluation. Random expert cosine is insufficient.
5. Measure TTFT and decode separately. Improve eager sparse attention,
   32-token expert chunks and per-call scale conversion before attempting
   graphs or concurrency. Admit 32K, images and larger batches independently.

Candidate command, printed only during offline preparation:

```bash
cd /workspace
VLLM_PP_LAYER_PARTITION=21,20,19 vllm serve /models/MiniMax-M3-W4 \
  --served-model-name minimax-m3-w4 --load-format minimax_m3_w4 \
  --dtype half --tensor-parallel-size 2 --pipeline-parallel-size 3 \
  --enable-expert-parallel --enforce-eager --block-size 128 \
  --max-model-len 8192 --max-num-seqs 1 --max-num-batched-tokens 128 \
  --gpu-memory-utilization 0.95 --limit-mm-per-prompt '{"image":0,"video":0}' \
  --port 8000
```

ACLGraph, FlashComm1, MTP, LoRA and context parallelism remain excluded from
this candidate. Full text/image API output, full-model accuracy, thermal
behavior and speed are unmeasured. No accuracy YAML or hardware sign-off is
claimed. The offline constraint supersedes the model-adapter skill's hardware
delivery gates for this milestone.

Evidence: [budget](../../../../artifacts/minimax-m3-serving-path-20261017/budget-candidate.json),
[capacity sweep](../../../../artifacts/minimax-m3-serving-path-20261017/capacity-sweep.json),
[six-rank probe](../../../../artifacts/minimax-m3-serving-path-20261017/distributed/report.json),
[ModelSlim samples](../../../../artifacts/minimax-m3-serving-path-20261017/modelslim-probe.json),
and test/lint receipts in the same artifact directory.
