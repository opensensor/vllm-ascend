# GLM Two-Card and Three-Card Candidates

This implementation preserves the two-card W2 work and prepares a distinct W3/W4
candidate for three dual-chip Ascend 310P cards. All work in this handoff is
offline. The Qwen agent retained the NPUs; no serving request, NPU test or service
mutation was performed. Neither model has been uploaded to Hugging Face.

| Candidate | Physical chips | Topology | Bank allocation including draft | Drafting |
| --- | --- | --- | --- | --- |
| Two cards | 4 | TP4, PP1 | 22 W4, 8 W3, 13 W2 | MTP1 |
| Three cards | 6 | TP2, PP3 | 22 W4, 21 W3, no W2 | Disabled initially |

The two-card canonical source remains intact. The three-card conversion rebuilt
layers 33–45 from the original FP8 source, rather than widening existing W2 codes.
The three-card artifact retains the draft bank on disk but excludes it from target
loading while MTP is disabled. Three-card pipeline inference is an unqualified
candidate until the six-chip gates pass; it is not a performance promotion.

## Implementation

The GLM text model now transfers all deferred mHC inputs between pipeline stages:
FP16 FFN output, FP32 four-stream residual, FP32 post mix and FP32 combination mix.
Stage-owned adaptation hooks and authoritative weight selection skip nonlocal
layers and placeholder modules. The first stage owns embeddings and the last
stage owns the final norm and output head. SP with PP is rejected; packed MTP
keeps its PP1 restriction.

```mermaid
flowchart LR
    input[Token IDs] --> first[Card 1: TP2, layers 0-14]
    first -->|hidden, residual, post, comb| middle[Card 2: TP2, layers 15-29]
    middle -->|hidden, residual, post, comb| last[Card 3: TP2, layers 30-44]
    last --> output[Final norm and logits]
```

The native CPU exporter stores the permanent Cube layout and FP16 scales without
re-quantizing codes. Shards contain only indexed tensors, and tensor bytes are
read back before the manifest becomes complete. Export resume requires matching
source, topology, scale and kernel contracts; existing tensors are compared with
source-derived values before reuse. Incomplete differing shards are replaced
atomically. W2, W3 and W4 byte transcoding is checked against the independent
Torch Cube packer.

Both candidates carry the v1140 prefill and v984 decode bundles. Dispatch uses
v984 for batches up to 16 rows and v1140 for larger batches, matching the existing
separation. The prepared v6 AI-Core indexer bundle is retained with relative asset
paths and verified hashes. Original license and attribution bytes come from the
FP8 source. Kernel compatibility on six chips remains untested.

The loader resolves complete Hub snapshots, including expert and indexer assets,
and rejects mismatched precision/topology markers. The native prefill workspace
capacity is bound to the scheduler's 640-token limit. GLM helper packages are now
included in built distributions; an isolated package-build/import test verifies
that runtime imports work outside the checkout.

## Offline Evidence

- The supported offline suite passed **443 tests**. Three existing files were
  excluded because this environment lacks torch-npu and required upstream vLLM
  modules. Earlier broad attempts failed at those dependencies; the exclusions
  are explicit, not hardware-equivalence claims.
- **2 packaging tests** passed, including imports from copied build output outside
  the source checkout.
- The final export/attribution suite passed **24 tests** after the last preflight
  additions. This overlaps the 443-test suite and is not an additional independent
  set of 24 cases.
- The final loader/pipeline regression run passed **28 tests**, also overlapping
  the supported suite.
- **2 opt-in serving gates skipped** because no GLM endpoint was supplied. They
  check real topology, deterministic output, prefix reuse and coding-client tool
  arguments when run on a dedicated service.
- An actual decoder-forward CPU test preserves mHC arithmetic bit-for-bit across
  two pipeline boundaries. Attention and FFN are stand-ins; this validates the
  handoff and control flow, not real GLM quality or HCCL transport.
- The W3 overlay completed **13 banks × 288 experts × 3 matrices = 11,232 matrix
  conversions** from FP8. Existing W3/W4 banks are retained unchanged.
- The narrow layer-33/expert-0 proxy in [weight-proxy.json](weight-proxy.json)
  reports relative squared reconstruction error of about **0.393–0.403 for W2**
  versus **0.0753–0.0779 for W3**, with FP16-rounded scales. This samples only three
  matrices and does not establish coding, reasoning or long-context accuracy.

The parameter ledger is header-only source accounting, not a hardware admission
result. Three-card stage routed-code/scale bytes are about 17.80/24.57/20.35 GiB per
TP rank. Including the deliberately conservative replicated-FP32 dense upper bound
gives about 30.62/33.59/31.73 GiB. Runtime pools, HCCL, KV/recurrent state, scratch,
padding and load temporaries are excluded. The two-card dense upper bound ignores
actual TP sharding and therefore overstates its footprint.

## Artifacts and Remaining Gates

The local model directories are:

- `/run/media/matteius/20TB-drive/models/GLM-5.3-Flash-two-card-native-candidate`
- `/run/media/matteius/20TB-drive/models/GLM-5.3-Flash-three-card-native-candidate`

Both native exports are complete and self-contained. Full payload SHA receipts
and final manifest snapshots are recorded alongside this handoff. These receipts
establish storage integrity only. Both full SHA-256 scans **passed**. Each model
contains 75,575 indexed tensors, including 37,152 code tensors and 37,152 FP16
scale tensors.

| Export | Indexed shards | Shard bytes (GiB) | Full checksum receipt |
| --- | --- | --- | --- |
| Two cards | 190 | 133.97 | [Passed](two-card-storage-check.json) |
| Three cards | 104 | 144.94 | [Passed](three-card-storage-check.json) |

The [offline validation record](offline-validation.json) links the captured test
results and identifies the remaining hardware gates. Generated [two-card](2-card-recipe.json) and
[three-card](3-card-recipe.json) launch recipes start at **8192 context tokens**
for qualification and use **640-token prefill chunks**. They have not been launched.

Required next gates are six-chip startup and per-stage memory, real model output
and reference quality, HCCL transport, stage-owned KDA/DSA cache reuse and reset,
request isolation/cancellation, prefix-cache accounting, tool calls, cold/warm
prefill latency and c1/c4 decode throughput. PP3 disables MTP and adds transport;
it must not be presented as faster without measurements. Larger context windows,
PP graph replay, multimodal PP and pipeline MTP remain unqualified or unsupported.

The deployment and publication workflow is in the
[card-profile tutorial](../../../docs/source/tutorials/models/GLM5.3-Flash-310P-Card-Profiles.md).
Before publishing, pin the runtime and compatibility versions and replace candidate
model-card limitations with actual hardware and accuracy results. Preserve the W2
release as a separate model.
