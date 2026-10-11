# GLM CUDA/Ascend sidecard

Start with GLM-5.3-Flash on one RTX PRO 6000 Blackwell and two Atlas 300I Duo
cards, exposing four Ascend 310P3 chips. The product goal is useful inference
for models larger than NVIDIA VRAM, improving on RTX plus CPU/weight offload.
Keep expert weights resident where they execute and exchange activations.

**Status: initial joined GLM service running.** The OpenAI-compatible endpoint is
`http://192.168.53.187:8001/v1`, model `glm-sidecard`. CUDA owns experts 0–71
in all 42 routed layers; the qualified Ascend backbone and other expert ranks
use both Ascend cards. Chat, streaming, concurrency and chunked prefill passed
operational checks. Greedy repeat variation remains unresolved, so full
numerical/quality qualification has not passed. See the [service runbook](SERVICE.md).
The current prefix-enabled profile advertises 645,120 context tokens and
measured approximately 56 uncached prompt tokens/s and 2.6–3.0 decode tokens/s.
The 8,192-token repeat reused 7,680 cached tokens, reaching first text in 9.16
seconds. See [measured performance](../../artifacts/sidecard-glm-20261011/fast/README.md)
for the cold-request timings and validation limits.

Five persistent CPU workers exercise
binary IPC, resident tiny SwiGLU experts and coordinator combination. The
canonical checkpoint path plans, exports and verifies owner tensors. The
packed-weight smoke loads exported W2/W3/W4 bytes in each child process.
The independent prototype server uses the resident loader described in the
runbook. A production accelerator expert backend, sidecard native checkpoint
exporter and measured sidecard speedup remain follow-up work. Hardware gates establish host-staged copy
integrity, real mixed-device expert execution and CUDA final reduction.
The previous Ascend-only server was replaced after CUDA preparation completed.

## Target ownership

| Component | Intended owner | Reason and gate |
| --- | --- | --- |
| Scheduler, tokenizer, sampling, embeddings, head | CUDA coordinator | One request lifecycle and one final output |
| KDA, DSA, KV/recurrent state, mHC, dense/shared MLP | CUDA | Keep state and sensitive residual arithmetic local; establish real-weight CUDA parity |
| Selected routed experts | CUDA | Use available GPU capacity and measured expert service time |
| Overflow routed experts | Four Ascend workers | Execute against resident packed banks using existing native kernels |
| Route selection, weighting, final combination | CUDA coordinator | Preserve correction bias, normalization, scale and shared addition |
| Cross-vendor communication | Separate processes, initially host staged | Qualify CUDA-to-host-to-NPU and return before considering direct device access |

```text
CUDA attention/state -> router -> compact dispatch ----> Ascend expert workers
                             \-> CUDA experts             |
                                  |                       |
                                  +<-- raw expert outputs-+
                                  |
                           ordered route combination
                           * routed scale + shared
                                  |
                             next CUDA layer
```

`cuda:0` and `npu:0` through `npu:3` are logical worker identities. Inventory
must establish the actual visible-device mapping before allocating weights.
HCCL remains an Ascend-only facility; NCCL remains a CUDA-only facility. This
does not configure a five-rank mixed tensor-parallel group.

## Commands

Run from the repository root. Planning tools use Python's standard library
and the project's `regex` dependency. They do not import vLLM, torch,
torch-npu, or initialize accelerator contexts.

```bash
python3 -m tools.sidecard inventory --ssh matteius@192.168.53.187
python3 -m tools.sidecard plan \
  --model-config /run/media/matteius/20TB-drive/models/GLM-5.3-Flash-FP8/config.json \
  --cuda-experts 72 --expert-bits 4
python3 -m tools.sidecard smoke
python3 -m tools.sidecard partition-smoke
python3 -m tools.sidecard checkpoint-plan \
  --source /run/media/matteius/20TB-drive/models/GLM-5.3-Flash-W2-310p \
  --placement artifacts/sidecard-glm-20261011/placement.json
python3 -m pytest -q --noconftest tests/ut/sidecard
```

Inventory executes only read-only host commands with bounded timeouts. Omit
`--ssh` to inspect the current machine. Plans go to stdout; redirect them to
an experiment artifact directory. The model-config SHA256 pins geometry,
not checkpoint identity or numerical equivalence.

The seed assigns 72 of 288 experts per layer to CUDA and 54 to each NPU for
all 42 main-model routed layers. It is a reproducible contiguous placement,
not a hot-expert policy. Each layer can subsequently carry different ownership
through `ExpertPlacement`. MTP is excluded from this first split.

At uniform W4 the logical expert-code lower bounds are 35.4375 GiB on CUDA
and 26.578125 GiB on each NPU. These exclude scales, NZ padding, native
packing, non-expert tensors, graph/workspace allocation, cache/state and driver
reserves. **They do not establish that a checkpoint fits.** The existing
qualified mixed W2/W3/W4 export cannot be treated as an all-W4 export or loaded
under this new partition without a matching exporter and loader.

## Reference boundary

`reference.dispatch` requires hidden rows, selected expert IDs, and nonnegative
route weights. Router weights must already use GLM's real scoring and
normalization; the reference does not recompute or renormalize them. A packet
contains an execution identity, layer, owner, original token/route-slot/expert
triples, and unique token rows for that owner. Idle owners receive no packet.
Zero-weight routes remain explicit in this initial correctness contract.

`ExpertReply` returns **unweighted full-width outputs for individual routes**.
`reference.combine` rejects stale, duplicate, missing, foreign and misowned
results, wrong shapes and nonfinite values. Replies may arrive in any order.
The CPU oracle rounds multiply and add separately to FP32 in original top-k
slot order, applies the routed scale once, then adds the shared result once.
It does not establish bitwise agreement with the existing Ascend reducer's
ordering or rounding; real-input comparisons must choose and qualify the
production arithmetic before integration.

For a token routed entirely to Ascend, the worst logical payload is four
FP16 hidden copies plus eight FP32 expert outputs: 160 KiB per layer, or
6.5625 MiB across 42 layers, excluding metadata and intermediate host copies.
At 640 prefill rows this becomes 4.1015625 GiB per model pass. Fewer active
owners/routes reduce traffic. CPU tuples represent the oracle and worker API.
The service's tensor transport passes contiguous binary buffers without
per-element Python floats, with synchronous pageable host/device copies.
Pinned memory and overlapped execution remain pending. Owner-level partial
reduction would reduce return traffic but changes floating-point addition
order and needs its own parity gate.

## Offline process runtime

`wire.py` implements bounded frames with a versioned prefix, JSON control
metadata, binary route/token identities, FP16 request rows and FP32 reply rows.
Placement and checkpoint hashes must match the configured worker session;
payload SHA256 catches changed packet bytes. Dimensions and byte ranges are
checked before unpacking. See the [wire contract](PROTOCOL.md).

`ExpertWorker` validates ownership before passing the **complete local batch**
to `ExpertBackend.compute(request)`. A future grouped accelerator backend can
consume all selected routes together. Only the tiny CPU `DenseExpertBank`
uses per-route matrix arithmetic. Workers return raw expert outputs and own
neither routing weights nor attention state.

`LocalWorkerPool` starts processes with Python's `spawn` context and separate
Unix socket pairs. Every pass receives a new execution ID, inputs are
snapshotted before dispatch, and active owners run through independent peers.
Before any pass, each worker prepares its backend, checks complete resident
expert coverage and acknowledges the configured owner/checkpoint/placement.
Startup has its own deadline; preparation failure returns a bounded reason
and closes the pool. A process starting is not sufficient to establish readiness.
The pool admits one pass at a time. Deadlines cover send and receive; framing,
worker or combination failures close the pool and reap its children. Closing
the pool is terminal cancellation; there is no retry or partial-result fallback.
The `smoke` command uses five synthetic CPU banks with noncontiguous global
expert IDs and compares three passes against a serial FP32 control.

## Checkpoint ownership plan

`checkpoint-plan` reads `config.json`, the safetensors index, and bounded shard
headers. It checks dtype/shape/offset consistency, slice overlap, complete
main-model expert coverage and canonical W2/W3/W4 block32 geometry. Expert
tensors go to their explicit owner; text-model dense/shared tensors go to
CUDA. Vision and draft tensors are excluded. Each owner's per-layer list maps
local index to global expert ID. Add `--include-tensor-slices` to emit exact
source filenames, tensor metadata and absolute byte ranges for an exporter.

The planner is limited to canonical individual expert tensors;
already-fused native banks require a separate export adapter. Reported bytes
are source tensor bytes, not final native allocations. Header hashes identify
the reviewed metadata, **not the payload contents**. Full payload checksums,
numeric compatibility and runtime admission are separate gates; the exporter
below records and verifies the bytes actually copied.

## Canonical export and checked loading

```bash
python3 -m tools.sidecard export-canonical \
  --source /path/to/canonical-checkpoint \
  --placement artifacts/sidecard-glm-20261011/placement.json \
  --output /path/to/new-sidecard-bundle
python3 -m tools.sidecard verify-partition \
  --root /path/to/new-sidecard-bundle \
  --placement /path/to/new-sidecard-bundle/placement.json \
  --owner npu:0 \
  --expected-checkpoint-sha256 CHECKPOINT_SHA256_FROM_EXPORT
```

`export-canonical` makes a fresh full GLM tensor plan and streams unchanged
source slices into owner safetensors shards with at most 8 MiB per copy chunk.
It preserves tensor names, dtypes, shapes, signed codes and scale values;
there is no requantization or native repacking. The default shard target is
1 GiB of tensor payload; an individual larger tensor occupies its own shard.
The complete source config is retained as provenance in `source-config.json`,
not as a vLLM launch configuration. This is a data export, not a serving recipe.

The exporter rejects existing destinations, incomplete/misowned plans and
changed source metadata. It checks shard snapshots before publication, hashes
every tensor while copying, and independently rehashes each exported shard
against the bytes written. Files and metadata directories are synced before
the final `sidecard-bundle.json` is published. Interrupted work leaves an
incomplete directory without that manifest; restarting requires a fresh path.

The bundle's manifest SHA256 becomes the shared checkpoint identity for all
workers. `PartitionStore` checks that identity, placement, source-config hash,
owner index, complete local expert coverage, canonical geometry and shard
checksums. It exposes immutable tensor descriptors and private byte reads.
`read_tensor` has a 64 MiB default allocation budget; `iter_chunks` stages
larger tensors with a final per-tensor checksum check. A staging consumer must
finish the iterator and complete device copies before reporting readiness.

Output checksum agreement establishes that the export retained the bytes read.
It does not authenticate the original checkpoint against an independent trusted
release checksum. Both claims are explicit in the manifest. Source fidelity,
native allocation sizes and accelerator behavior remain unqualified.

`partition-smoke` creates temporary synthetic canonical weights, exports them,
and starts five `CanonicalCPUBackend` specifications. Each child verifies and
loads only its owner partition before acknowledgement. W2/W3/W4 sign extension
and F16/F32 block32 scale decoding feed the tiny FP32 SwiGLU oracle. Three
passes match an independent control built directly from the source integer
values. Source-file hashes remain unchanged. This backend is restricted to
32/64-wide fixtures and an explicit expanded-element budget; it rejects full
GLM dimensions and does not emulate native activation quantization.

## Hardware gates

Run these in an isolated copy of the tools on the joined host, after sourcing
CANN and ATB. Each probe owns its allocations; it does not reset devices or
stop the existing service. Choose a fresh log directory for each copy run.

```bash
. /usr/local/Ascend/ascend-toolkit/set_env.sh
. /usr/local/Ascend/nnal/atb/set_env.sh
python3 -m tools.sidecard.hardware_probe \
  --npu-python /srv/ai/venvs/qwen38-w4-test-ce1862/bin/python \
  --log-dir /srv/ai/storage/artifacts/sidecard-copy-new-logs
```

The CUDA coordinator uses `libcudart` without installing CUDA PyTorch. An
Ascend child starts in the specified Python environment, initializes one chip,
and casts synthetic FP16 rows to FP32, optionally repeating each row eight
times to model return volume. Workers run sequentially across chips 0–3.
Requests have bounded shapes, sequence IDs and payload checksums. The probe
checks CUDA upload/download bytes, the NPU return and the final CUDA copy,
including signed zero. Every device copy/operation has a completion fence;
workers acknowledge readiness and shutdown and exit normally.

This uses ordinary pageable host buffers and a separate diagnostic byte
protocol, not the production expert wire protocol. Timings include allocation,
serialization, checksums and extra CUDA copies for verification. Large input
and return volumes reach 5 MiB and 80 MiB respectively. Neither these timings
nor sequential chip execution measure overlapping expert service or serving
throughput. The October 11 run passed all 24 cases; see the
[hardware evidence](../../artifacts/sidecard-glm-20261011/HARDWARE.md).

The native arithmetic gate requires the existing qualified `tools.glm_perf`
package. Preserve CANN's sourced Python path when adding its directory:

```bash
PYTHONPATH="/srv/ai/storage/artifacts/glm-next-math-hardware-20261011/qualified-source${PYTHONPATH:+:$PYTHONPATH}" \
  /srv/ai/venvs/qwen38-w4-test-ce1862/bin/python \
  -m tools.sidecard.native_expert_probe \
  --checkpoint /srv/ai/storage/models/GLM-5.3-Flash-two-card-native-candidate \
  --layers 3 8 33 --expert 72 --device 0 --rows 1 17
```

`--plan-only` admits headers without importing accelerator libraries. Execution
verifies the selected native shards and frozen kernel provenance, clones
mmap-backed tensors before upload, and keeps one expert resident per layer.
It compares changed FP16 activations against the existing independent CPU
activation-quantization reference and checks repeatability. Top-k=1 and unit
route weights make each returned row an unweighted single-expert output.
All 12 W4/W3/W2 cases matched exactly on chip 0. This does not establish CUDA
expert parity, complete owner-bank loading, captured GLM activation parity or
model quality. A future raw-output adapter could gather one row per route and
use this top-k=1 seam; validate arbitrary global-to-local maps, route-volume
limits and gather cost before adopting it.

## Selected-expert split gate

`ExternalWorkerPool` launches each expert worker in its specified Python
environment with trusted, bounded JSON configuration. Requests and replies
use the existing `SCARD001` protocol. Workers authenticate source shards,
prepare resident banks, complete upload fences and acknowledge coverage before
dispatch. Closing the pool disconnects every worker and waits for normal exits;
a stuck process is reported with its PID and is never hard-killed.

`NativeGateBackend` maps arbitrary global expert IDs to local bank indices.
The Ascend branch gathers compact token rows on device and invokes the qualified
native kernels with top-k=1/unit weights to return raw rows in original request
order. The CUDA branch uploads losslessly decoded INT8 weights and uses the
new correctness kernel for the same block32 activation and scale contract.
`CudaCombiner` validates complete reply coverage before restoring original
route slots and performing unfused FP32 multiplication, accumulation, routed
scaling and shared addition on CUDA. Inputs, outputs and copies are explicitly
owned and fenced. Attention, tokenization and model scheduling are not part of
this gate.

On the joined host, in an isolated source directory, compile the CUDA library:

```bash
/usr/local/cuda/bin/nvcc -std=c++17 -O2 --fmad=false -arch=sm_120 \
  -shared -Xcompiler -fPIC tools/sidecard/cuda_oracle.cu \
  -o ../cuda_oracle.so
```

After sourcing CANN/ATB, prepare CPU expectations in the Ascend environment.
Preserve the sourced Python path and add the qualified helper directory:

```bash
PYTHONPATH="/srv/ai/storage/artifacts/glm-next-math-hardware-20261011/qualified-source${PYTHONPATH:+:$PYTHONPATH}" \
  /srv/ai/venvs/qwen38-w4-test-ce1862/bin/python \
  -m tools.sidecard.split_gate prepare \
  --checkpoint /srv/ai/storage/models/GLM-5.3-Flash-two-card-native-candidate \
  --layers 3 8 33 --output ../split-golden-new
```

Use the emitted `golden_sha256` value for the run. The CUDA worker uses the
separate CPU-Torch environment for checkpoint manipulation and the CUDA runtime
for device work. No CUDA PyTorch installation is required:

```bash
PYTHONPATH="/srv/ai/storage/artifacts/glm-next-math-hardware-20261011/qualified-source${PYTHONPATH:+:$PYTHONPATH}" \
  python3 -m tools.sidecard.split_gate run \
  --golden ../split-golden-new --golden-sha256 <golden_sha256> \
  --npu-python /srv/ai/venvs/qwen38-w4-test-ce1862/bin/python \
  --cuda-python /srv/ai/venvs/laya-host-only-20261004/bin/python \
  --library ../cuda_oracle.so --logs ../split-run-new
```

The fixture selects actual global experts 0–7. CUDA owns 0/5; chips 0–3 own
1/6, 2, 3, and 4/7 respectively. It exercises noncontiguous banks, all eight
route slots, retained zero weights, changed inputs, shared addition and repeat
requests. The 18 real W4/W3/W2 cases and six dedicated GPU reduction cases
matched the CPU reference bitwise. This fixture placement is independent of
the full 72-CUDA/54-per-chip seed.

Backends deliberately reject more than three layers, sixteen experts per
owner per layer, or 256 routes. The coordinator reducer is bounded to 32 tokens
and eight slots. CUDA stores unpacked weights and uses a scalar correctness
kernel; optimized packed CUDA MoE remains pending. A separate resident service
loader now admits 72 CUDA experts across all 42 main routed layers using the
original EP partitions. The current tuple codec, CPU validation and host staging are
also part of measured gate latency. These results do not measure joined GLM
serving performance or qualify model quality. Source hashes, worker logs,
timing comparison and exact scope are in the
[hardware report](../../artifacts/sidecard-glm-20261011/HARDWARE.md).

## Integration gates

1. Freeze the hardware/PCIe/NUMA inventory and separate CUDA/CANN environments.
   Measure both host-copy directions on every chip at c1/c4 decode and prefill
   sizes; include completion fences, IPC latency and retained-buffer checks.
   Start eager with explicit buffer ownership and timeouts.
   Initial synchronous diagnostic copies passed; pinned buffers, concurrent
   owners, actual expert protocol copies and overlap remain pending.
2. Freeze one source checkpoint and quantization policy. Export only owned
   expert tensors with source/output SHA256 and global-to-local expert maps.
   Reject missing experts, wrong partitions and mixed scale/layout markers.
   Admit actual resident bytes independently on each device after reserves.
3. Run the existing Ascend native gate/up, SwiGLU/requantization and down
   projections as expert-only workers. Adapt the arbitrary partition and raw
   output seam outside the model runner. Do not instantiate four full GLM
   attention/state models or inherit the current EP-equals-TP ownership.
4. Use a minimal wrapper/composed expert component for the CUDA GLM model.
   Establish L0/L3/L7 and logits parity, then one real split MoE layer using
   captured inputs. All workers must use compatible source weights; CUDA
   NVFP4 and Ascend integer W4 are distinct numerical candidates.
5. Wire all 42 layers; check actual answers, tool arguments, multiple sequences,
   prefix reuse, drain, worker failure and cancellation. Execution IDs must
   distinguish every scheduler pass and layer; buffers cannot be reused before
   both copies and consumers finish. Attention state stays on CUDA.
6. Compare matched fidelity against RTX plus host offload and Ascend-only GLM.
   Measure cold TTFT, inter-token latency, c1/c4 throughput, memory peaks,
   per-owner service time and route co-occurrence. Tune placement using actual
   traffic and the slowest branch. Add MTP and graphs after eager correctness.

The performance dependency is:

```text
MoE latency = dispatch + max(CUDA expert time, Ascend expert time) + return/combine
```

Capacity and route co-occurrence constrain placement. A rarely selected expert
can still stall a layer. Neither nominal bandwidth sums nor task-time totals
are a sidecard speed forecast.

See the [research review](../../docs/source/developer_guide/Design_Documents/glm_sidecard_research.md)
and the [initial evidence](../../artifacts/sidecard-glm-20261011/README.md).
