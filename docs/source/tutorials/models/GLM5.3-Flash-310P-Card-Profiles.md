# GLM-5.3 Flash on Two or Three Ascend 310P Cards

## Introduction

These experimental text-serving profiles target dual-chip Ascend 310P cards.
Two cards supply four chips; three cards supply six. The three-card implementation
has passed offline pipeline and checkpoint tests, but has not passed six-chip
inference, accuracy, prefix-cache or performance qualification. Exporting a
checkpoint does not establish its supported context length.

| Profile | TP | PP | Routed weights | MTP | Initial execution |
| --- | --- | --- | --- | --- | --- |
| `two-card-selective-w2` | 4 | 1 | Existing W2/W3/W4 allocation | 1 draft token | Full decode graphs, sizes 2 and 8 |
| `three-card-w3` | 2 | 3 | W3/W4; no W2 banks | Disabled | Eager for qualification |

GLM's 64 KDA heads and 64 attention heads do not divide by TP6. The six-chip
profile instead uses two-chip tensor groups within three pipeline stages.
For the 45-layer model, the default stages own layers 0–14, 15–29 and 30–44.
The checkpoint's `world_size` is the **expert partition size inside a stage**:
4 for the two-card model and 2 for the three-card model. It is not the total
number of workers when pipeline parallelism is enabled.

The pipeline transfers the FFN output, four-stream FP32 residual, post mix and
combination mix together. All four fields are needed to preserve the deferred
mHC operation across stage boundaries. Each stage loads only its local layers
and expert partition; embeddings belong to the first stage and the output head
to the last. Sequence-parallel MoE with PP is rejected. Packed MTP remains
restricted to PP1, so it must stay disabled for the three-card profile.

## Supported Features and Limits

The two-card precision policy preserves 22 W4 banks, 8 W3 banks and 13 W2 banks,
including the draft bank. The three-card artifact preserves the 22 W4 banks and
uses W3 for the other 21 banks. Its draft weights remain on disk for future work,
but are excluded from target-model loading when MTP is disabled.

Both profiles use native packed expert codes, A4 activations and FP16 scale
storage. Codes are prepared on CPU once and stored in the permanent Cube layout.
The runtime copies those bytes without decoding or repacking the weights during
loading. New W3 values must come from the higher-precision source; repacking W2
into a wider container cannot restore discarded information.

These artifacts require the opensensor vLLM Ascend fork and compatible Ascend
310P native kernels. They are not generic CUDA quantized checkpoints. This
release path is text-only; multimodal PP and PP decode graph replay require
separate implementation and qualification.

## Environment and Memory Admission

Use the same pinned vLLM, vLLM Ascend, PyTorch, torch-npu and CANN versions used
to qualify the kernel bundles. Preserve the existing qualified custom OPP stack
for KDA, MLA, mHC and indexer operations: the checkpoint includes the routed
expert bundles, not an entire CANN installation or every custom operator.
Run the commands below from the vLLM Ascend checkout. CPU preparation does not
require an NPU or torch-npu.

Generate a parameter ledger before scheduling a hardware run:

```bash
python -m tools.glm_perf.card_memory_plan \
  --source /models/GLM-5.3-Flash-FP8 \
  --cards 3 --output three-card-parameter-ledger.json
```

The ledger counts exact routed code and FP16 scale bytes under the profile and
uses a conservative FP32 replicated upper bound for dense parameters. It excludes
KV and recurrent state, scratch buffers, HCCL, graph pools, padding and temporary
load buffers. It cannot admit a context window by itself. The two-card dense
upper bound intentionally overstates memory because it does not credit actual
TP sharding. Confirm stage boundaries, runtime reservations and per-chip memory
before raising the context limit. Keep the prefill limit at 640; the native
loader binds its reusable prefill workspace capacity to that scheduler limit.

## Prepare Separate Checkpoints

Generate a reproducible conversion and launch recipe for each profile:

```bash
python -m tools.glm_perf.deployment_profiles --cards 2 \
  --source /models/GLM-5.3-Flash-FP8 \
  --model /models/glm-two-card-native > two-card-recipe.json
python -m tools.glm_perf.deployment_profiles --cards 3 \
  --source /models/GLM-5.3-Flash-FP8 \
  --model /models/glm-three-card-native > three-card-recipe.json
```

Each recipe includes `conversion_args` for a new canonical checkpoint and
`launch_args` with the matching topology. The initial context limit is 8192 for
qualification. Generating these files does not launch a service. Review memory
admission first. A larger window requires its own hardware test.

Run the appropriate `conversion_args` from the recipe. Always use separate
output directories. Full conversion supports default W2, W3 or W4, with explicit
layer overrides. Resume metadata binds the source index hash, precision policy
and shard settings; a different recipe is rejected rather than reusing cached
W2 outputs. Legacy progress files without this contract require a separate output.

When preserving an existing selective-W3 canonical base, only the W2 banks need
new quantization for the three-card candidate:

```bash
python -m tools.glm_w2.convert_w3_overlay \
  --source-dir /models/GLM-5.3-Flash-FP8 \
  --base-dir /models/GLM-5.3-Flash-selective-W3-310p \
  --out-dir /models/glm-three-card-native-canonical \
  --layers 33,34,35,36,37,38,39,40,41,42,43,44,45
```

This canonical overlay may reference unchanged base shards. The following native
export creates a self-contained directory containing only indexed tensor payloads,
independent of the original base. It verifies each written tensor's identity,
shape, dtype and bytes and keeps the manifest incomplete until the export finishes.
CPU preparation uses byte transcoding with bounded parallel work. If interrupted,
repeat the same command with `--resume`: only an incomplete matching candidate is
accepted, and each existing tensor is compared against the source-derived bytes
before reuse. Truncated or different candidate shards are replaced atomically.
Completed exports are never overwritten by resume.

```bash
python -m tools.glm_perf.export_native_cpu \
  --source /models/glm-two-card-native-canonical \
  --output /models/glm-two-card-native \
  --bundle /qualified-bundles/prefill-v1140 \
  --decode-bundle /qualified-bundles/decode-v984 \
  --indexer-bundle /qualified-bundles/indexer-v6 \
  --attribution-source /models/GLM-5.3-Flash-FP8 \
  --world-size 4 --cards 2
python -m tools.glm_perf.export_native_cpu \
  --source /models/glm-three-card-native-canonical \
  --output /models/glm-three-card-native \
  --bundle /qualified-bundles/prefill-v1140 \
  --decode-bundle /qualified-bundles/decode-v984 \
  --indexer-bundle /qualified-bundles/indexer-v6 \
  --attribution-source /models/GLM-5.3-Flash-FP8 \
  --world-size 2 --cards 3
```

The separate decode bundle preserves its scheduling for batches up to 16 rows;
larger batches use the prefill bundle. Both must match the permanent scale
contract and have distinct operator namespaces and helper packages. The existing
v984/v1140 assets are candidates for six-chip use, not proof of six-chip validity.
The prepared indexer bundle is verified and rewritten with relative asset paths,
so its AI-Core casts remain usable after a Hub download. Original license files
and the source model card are copied from the attribution source without changing
weights. The decode and indexer bundles are optional. Attach them to a completed export
with `tools.glm_perf.export_native_cpu.attach_decode_bundle` and
`tools.glm_perf.export_native_cpu.attach_indexer_bundle` without touching weights.

Verify every indexed shard and bundled kernel before distributing the directory:

```bash
python -m tools.glm_perf.verify_card_checkpoint \
  --model /models/glm-three-card-native --output three-card-storage-check.json
```

The verifier checks SHA-256 payloads by default. `--headers-only` skips weight
payload hashing and is suitable only for inspection, not publication verification.

## Functional and Accuracy Qualification

After memory admission, use the reviewed recipe's `launch_args` on a dedicated
server. The loader rejects mismatched profile markers, expert partition sizes,
scale contracts and runtime TP/PP/MTP settings. A Hub model ID is resolved with
its revision and download cache, including both native kernel directories.

Run the opt-in serving gate against that dedicated endpoint:

```bash
pytest tests/e2e/nightly/310p/single_node/card_profiles \
  --glm-profile-url http://127.0.0.1:8001 --glm-profile-cards 3
```

This gate checks the actual server topology, 640-token scheduler budget,
precision profile's served name, deterministic answer and positive prefix reuse.
It is not a full accuracy evaluation. Also qualify stage-owned KDA/DSA caches,
long-context request isolation, reset/reuse, HCCL transport and cancellation.
Compare real model outputs against the higher-precision reference on coding,
reasoning, repeated prompts and long-context tasks. TP2 changes reduction ordering;
CPU code-value preservation alone does not guarantee inference parity.

Record cold and warm time to first token, c1/c4 decode tokens/s, acceptance when
MTP is enabled, context length, cache reuse and peak memory per chip. The three-card
profile disables MTP and adds PP transport; it must not be advertised as faster
without measurements. No six-chip accuracy or performance numbers are available
for these candidates yet.

## Hugging Face Release Preparation

Keep two separate model repositories and retain W2 in the two-card release.
Preserve the original license files and model attribution. Before upload, pin the
runtime revision and compatibility versions, complete the hardware/quality gates,
and update each model card with measured limits and results. Include all indexed
safetensors, tokenizer/config files, `native-layout.json`, expert and indexer
kernel directories when attached, their provenance and the original license and
`SOURCE_README.md` attribution. Do not upload a canonical overlay that relies
on local base paths.

The eventual CLI operation is `hf upload YOUR_ORG/YOUR_MODEL /models/YOUR_NATIVE_MODEL`.
The repository names are placeholders, and uploading is a separate publication
step. Nothing in this workflow uploads or changes a running service automatically.
