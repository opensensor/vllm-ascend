# Two-card Atlas 300I Duo workstation plan

Prepared 2026-09-11 against vLLM Ascend commit `7ab47e73d`.
Expected hardware arrival: approximately October 2026.
Status: source and documentation review only; no NPU validation performed.

Assume two 96 GB Atlas 300I Duo cards, exposing four 48 GB processors,
on the proposed Threadripper 3970X / ASUS Prime TRX40-Pro Linux system.
Confirm the delivered part numbers and memory configuration before using these
capacity assumptions. The first milestone is a useful, reproducible local model
service with measured quality and latency. Larger models follow measured results.

## Evidence that changes the starting plan

| Target | Evidence available today | Planning decision |
| --- | --- | --- |
| Qwen3-0.6B FP16 | The current [quick start](source/getting_started/quick_start/online/qwen3-0.6b-310p.inc.md) has a 310P example. | Use this tiny model to check installation before quantization. |
| Qwen3-8B W8A8SC | The [dense-model tutorial](source/tutorials/models/Qwen3-Dense.md) documents TP1 with a pre-sharded checkpoint. | First quantized baseline; test each processor separately. |
| Qwen3.6-27B W8A8 | A [Duo test configuration](../tests/e2e/nightly/single_node/models/configs/Qwen3.6-27B-W8A8-310P-300I-DUO.yaml) specifies TP2. | Useful alternative baseline for hybrid attention. |
| Qwen3.8-27B W8A8 | The [official tutorial](https://docs.vllm.ai/projects/ascend/en/latest/tutorials/models/Qwen3.8-27B.html) now includes Duo deployment at TP2 or TP4. | A documented starting point, subject to reproducing the exact software and checkpoint combination. |
| Qwen3.8-27B W8A8SC | The [checkpoint author](https://huggingface.co/adeepv/Qwen3.8-27B-w8a8sc-Ascend310P) reports TP1 and TP2 operation with separate layouts, a pinned image, and patches. | Treat as a distinct reproduction experiment. Its published shards are text-only; quality loss has not been formally benchmarked. |
| Qwen3.6-35B-A3B W8A8 | The [official tutorial](https://docs.vllm.ai/projects/ascend/en/main/tutorials/models/Qwen3.6-35B-A3B.html) includes Duo, and this repo has TP4 [baseline](../tests/e2e/nightly/single_node/models/configs/Qwen3.6-35B-W8A8-310P-300I-DUO.yaml) and [MTP](../tests/e2e/nightly/single_node/models/configs/Qwen3.6-35B-A3B-w8a8-310p-MTP.yaml) configurations. | First substantial MoE reproduction target. These configurations are stronger evidence than the older experimental-support description, but are not results from our workstation. |
| Qwen3-Next-80B-A3B | The [tutorial](source/tutorials/models/Qwen3-Next.md) still describes A2/A3 and Triton. The [310P GDN implementation](../vllm_ascend/_310p/ops/fla/gdn_310.py) already shares the upstream GDN base and discusses Qwen3-Next semantics. | Audit dispatch, tensor shapes, state precision, and checkpoint compatibility before deciding new kernels are needed. |
| GLM-4.5-Air / gpt-oss-120b | Their generic tutorials do not establish a working 310P checkpoint and complete serving path. | Keep as later feasibility investigations; capacity estimates alone do not rank porting effort. |
| Qwen3-235B-A22B W4A8 | The [310P quantization registry](../vllm_ascend/_310p/quantization/methods/__init__.py) imports W8 methods. W8A8_DYNAMIC has a MoE implementation; W4A8 MoE is absent from this registry. | Require a small packed-weight kernel and loader demonstration before attempting the full checkpoint. |

There is a documentation inconsistency to resolve: the Qwen3.8 weight list names
`Eco-Tech/Qwen3.8-27B-w8a8-310p`, while its Duo launch example names
`Eco-Tech/Qwen3.8-27B-w8a8`. Inspect the chosen repository's quantization metadata
and revision before downloading or launching it. Do not substitute W8A8SC shards
into a W8A8 recipe.

The community W8A8SC author reports failures with dynamic W8A8 on their stack and
roughly 5 tokens/s decode for their sparse build. Those are useful reproduction
data, not a forecast for our machine or evidence that every current W8A8 build
fails. Their eager-mode requirement is specific to that recipe; official Duo
examples also exercise graph mode.

The [310P nightly configuration](../.github/workflows/configs/nightly_config.yaml)
uses `linux-aarch64-310p-*` runners for the listed models. The image workflow also
builds amd64, but an image build does not establish model correctness or HCCL
performance on Threadripper.

## Work during the month before arrival

These are planned tasks, not completed validation results.

| When | Work | Concrete completion criterion |
| --- | --- | --- |
| Week 1 | Confirm card SKU, auxiliary-power connector and pinout, cooling arrangement, slot clearance, PSU capacity, and supported driver/host-kernel combination. Check access to the required Huawei downloads. | A parts and installation manifest with remaining unknowns recorded. Use the actual board documentation for cabling and cooling. |
| Week 2 | Select a matched 310P software stack; inspect the container manifest for `linux/amd64`. Record image digest, driver/firmware requirements, CANN, torch, torch-npu, vLLM, and Ascend versions. | A reproducible software manifest and accessible installers. Host drivers and container user-space versions must be compatible. |
| Week 3 | Audit checkpoint configs, quantization descriptions, file lists, and TP-specific layouts. Stage the tiny and 8B baselines first. Prepare a fixed prompt set and benchmark result schema. | Exact model IDs/revisions, loader requirements, storage estimates, and reproducible evaluation inputs. |
| Week 4 | Trace the selected models through the 310P backend. Identify existing unit/operator tests and prepare launch profiles for TP1, TP2, and TP4 where supported. | A short list of known code paths and unresolved questions; commands checked against the pinned release. |

Avoid mixing versions merely because their numbers are close. At the reviewed
commit, [Dockerfile.310p](../Dockerfile.310p) uses CANN 9.1.0 and upstream vLLM
v0.28.0, while the Qwen3.8 tutorial describes a v0.23.0 baseline. The
[v0.26.0rc1 release notes](source/user_guide/release_notes.md) explicitly restrict
that release's fully validated model set. Choose a coherent recipe first, then
make upgrades separate experiments. Recheck available releases near arrival.

Keep independent records for the official W8A8 recipe and the community W8A8SC
recipe. Review any community patches against their original commit before
considering them for a newer checkout. Do not replace current backend files
wholesale with older versions.

## Host memory and checkpoint accounting

The proposed board supports 256 GB of unbuffered DDR4 according to the
[ASUS manual](https://dlcdnets.asus.com/pub/ASUS/mb/SocketTRX4/PRIME_TRX40-PRO/E16115_PRIME_TRX40-PRO_UM_V2_WEB.pdf).
If the available Corsair memory consists of four physical 16 GB modules, it totals
64 GB. Use it for initial setup and the small baselines. A later 8 x 32 GB
configuration is a sensible capacity target for large checkpoint work, but is
neither a universal loading requirement nor a guarantee every conversion fits.

Estimate each conversion separately. A 120-billion-parameter model expanded to
FP16 is about 240 GB of tensor payload before process overhead or intermediate
copies. Streaming or layerwise conversion may therefore matter even with a
256 GB host. Prefer ready-to-load compatible checkpoints during bring-up.

Budget local storage for source weights, converted weights, TP exports, container
layers, temporary files, and logs. Record actual file sizes before allocating a
large download; retained variants can exceed the final serving checkpoint size
several times over.

At the assumed capacity, `4 x 48 x 0.9 = 172.8` in the same units as the original
48 GB figure. This is an aggregate planning allowance, not a shared allocation.
Measure usable bytes on each processor and distinguish GB from GiB.

For each rank, account for:

```text
resident weights and scales
+ padding, repacking, and replicated tensors
+ peak temporary allocations during loading
+ activations and operator workspaces
+ HCCL buffers and graph allocations
+ KV cache and recurrent state
+ operating margin
<= measured memory available to that rank
```

Sparse MoE routing reduces computation per token; it does not remove the need to
store inactive experts in a conventional resident deployment. File size can also
understate runtime size if a loader expands packed tensors. Verify that a proposed
W4 path preserves compact storage before accepting a 235B fit estimate.

## First hardware session

1. Install and test one card, then add the second. Record exact model, firmware,
   device memory, health, PCIe mapping, and cooling behavior. Confirm four usable
   processors with both cards installed; do not infer device numbering from slots.
2. Test allocation, copies, and a small FP16 matmul independently on each processor.
   Check correctness and collect driver errors, temperatures, and memory use.
3. Run the tiny real-weight model on each processor. Require an actual completion,
   then test Qwen3-8B W8A8SC with its matching TP1 export and `sharded_state` loader.
4. Test HCCL correctness and latency for a pair within one card, a pair spanning
   cards, and all four processors. Include small messages and representative
   activation-sized messages. PCIe enumeration alone does not validate collectives.
5. Reproduce a TP2 hybrid model, then compare with TP4. Use ordinary tensor
   parallelism first; establish expert-parallel support separately for the selected
   model, backend, and release.
6. Enable graph mode, prefix caching, and MTP individually after a correct baseline.
   Recheck outputs after each change. Follow the release's capture-size limits and
   keep `enable_npugraph_ex` disabled for the documented Duo path.
7. Run a sustained mixed workload and retain the first stable software/model
   combination as the reference for further experiments.

Useful host inventory commands to capture on the new workstation:

```bash
uname -a
lspci -Dnn
lspci -tv
numactl --hardware
free -h
df -h
npu-smi info
```

A conservative tiny-model launch profile, adapted from the repository quick start,
is below. This is a planned command, not a tested result. Run it in the selected
310P amd64 container with one confirmed processor exposed, from `/workspace`.
The model must be available locally or downloadable in that environment.

```bash
vllm serve Qwen/Qwen3-0.6B \
    --host 127.0.0.1 \
    --port 8000 \
    --dtype float16 \
    --tensor-parallel-size 1 \
    --max-model-len 4096 \
    --max-num-seqs 1 \
    --gpu-memory-utilization 0.80 \
    --enforce-eager \
    --additional-config '{"ascend_compilation_config":{"enable_npugraph_ex":false}}'
```

From another shell in that container:

```bash
curl --fail http://127.0.0.1:8000/v1/models
curl --fail http://127.0.0.1:8000/v1/completions \
    -H 'Content-Type: application/json' \
    -d '{"model":"Qwen/Qwen3-0.6B","prompt":"The capital of France is","max_tokens":32,"temperature":0}'
```

Require nonempty, plausible generated text as well as HTTP success. For larger
models, also require repeatable task accuracy; a successful request alone does not
validate quantization, state handling, or long-context correctness. Dummy weights
can isolate startup/operator issues but cannot validate a real checkpoint.

## Measurements that decide the next experiment

Record the model revision, image digest, source commits, exact command, visible
device mapping, TP/EP settings, quantization layout, and host/kernel versions for
every result. Keep model output and errors alongside the metrics.

| Question | Controlled comparison |
| --- | --- |
| Does TP4 help this workload? | Same checkpoint semantics and prompts at TP2 and TP4, using matching exports if pre-sharded. Also compare two independent TP2 services for aggregate throughput. |
| Is a model useful interactively? | Time to first token, decode tokens/s per request, and p50/p95 latency at concurrency 1; then repeat at 4 and 8 if memory permits. |
| Can it handle useful context? | Start at 4K, then 8K, 16K, and 32K with fixed output allowance. Measure prefill and decode separately; verify retrieval and task accuracy at each length. |
| Does an optimization help? | Toggle one feature at a time, with identical inputs, cache conditions, and generation settings; include warmup and repeated trials. |
| Is it stable? | Track per-rank peak memory, host RSS, temperatures, clocks, and errors under repeated requests and mixed prefill/decode. |
| Does a larger model justify its cost? | Run the same coding, reasoning, and tool-call tasks with stated scoring rules. Total or active parameter counts do not establish task quality. |

## Porting milestones after the baselines

1. **Qwen3.6-35B-A3B W8A8:** reproduce the documented TP4 path and establish its
   quality/performance baseline before attempting alternate expert placement.
2. **Qwen3-Next-80B-A3B:** inspect the real checkpoint's quantization and dimensions;
   trace GDN, convolution, attention, routing, and grouped matmul. Reuse the
   [existing GDN tests](../tests/ut/_310p/ops/test_gdn_310.py) and
   [dynamic MoE tests](../tests/ut/_310p/quantization/test_w8a8_dynamic_moe_310.py).
   Convert a reproduced failure into a small regression case before patching.
3. **GLM-4.5-Air:** establish a compatible quantization recipe and test GLM routing,
   shared experts, attention, and loading at small scale before a full TP4 run.
   Choose between this and Qwen3-Next based on observed blockers and useful output.
4. **gpt-oss / packed-weight investigation:** use a small representative expert
   block to check format decoding, activation semantics, numerical error, and
   matmul throughput. Measure tile-decompression workspace and cost before claiming
   MXFP4 storage translates into a practical serving path.
5. **Qwen3-235B-A22B W4A8:** proceed only after compact resident storage, correct
   expert matmul, loading, and per-rank memory accounting are demonstrated. Start
   with short context and one request. Treat this as a research target with no
   delivery commitment or throughput estimate yet.

Revisit additional cards after measuring this system. Six processors do not imply
valid TP6, and eight processors matching model dimensions would still leave PCIe,
driver, collective, cooling, power, and physical-slot constraints to solve.
