# MiniMax M3.1 Flash on six Ascend 310P chips

## Introduction

This experimental port targets three dual-chip 310P3 cards with separate memory
banks of about 45.6–46.2 GiB. It uses a custom signed INT4 expert export with FP16
dense layers. It is not the published W8A8, 16-rank Ascend deployment.

The pinned source is
[MiniMaxAI/MiniMax-M3.1-Flash](https://huggingface.co/MiniMaxAI/MiniMax-M3.1-Flash/tree/54e4e69dfc452b1e94a426c24a4fdeeaa8e3ad5e).
Its complete export contains 23,428 tensors and occupies approximately 219.16 GiB.
Original NVIDIA FP4 bytes are decoded and requantized; they are not interpreted
as signed INT4. The conversion changes numerical values and requires quality
validation.

Real-weight serving passed nine small coherence checks: greeting, factual recall,
arithmetic, exact copying, JSON, multi-step arithmetic, multi-turn recall, a red
image, and retrieval from a 3443-token prompt. A separate 7943-token retrieval also
returned the correct key. These are bounded diagnostics,
not a broad quality benchmark. Full 60-layer dummy serving also passed an
8192-token cache-boundary request; dummy output is not evidence of coherence.

## Supported features

| Feature | Current scope |
| --- | --- |
| Tensor/pipeline parallelism | TP2, PP3; layer partition 21/20/19 |
| Expert parallelism | EP2 within each pipeline stage |
| Attention | Eager FP16 paged KV with FP32 sparse scoring/softmax |
| Images | One image, maximum long side 448 pixels in this profile |
| Video | Disabled in this profile |
| ACLGraph | Unsupported by the current eager adapter; rejected explicitly |
| FlashComm1 | Sequence-parallel MoE unsupported by this adapter |
| MTP/DSpark | Separate draft checkpoint not converted; speculation rejected |
| Context/concurrency | Initial profile 8192 tokens, one sequence |

An offline 128K × 16 capacity calculation needs approximately 92.57 GiB on the
busiest rank, including the provisional runtime reserve. It does not fit these
chips. The 8K profile is the practical starting point, not a claim of validated
long-context language quality.

## Environment preparation

This deployment uses the existing native 310P host installation, rather than an
A2/A3 container recipe. Runtime paths on `threadripper` are:

- `/srv/ai/venvs/minimax-m31`: Python 3.12, Torch 2.13.0+cpu,
  torch_npu 2.13.0rc1, transformers 5.17.0 (unchanged).
- CANN 9.1.0 with the installed 310P driver.
- `/vllm-workspace/vllm`, source commit
  `3ab5dda29acabea01f6a63d0806bdbbb4a27bde5`.
- `/vllm-workspace/vllm-ascend`, installed editable in the same environment.
- `/workspace/minimax-m31-env.sh`, selecting the matching custom operator
  package and preserving CANN's Python paths.

Use the current 16×256 grouped W4 kernel and its matching NZ256 load-time
packing. The canonical checkpoint stays unchanged. The qualified combined OPP
package is `/vllm-workspace/minimax-m31-merged-opp/vendors/custom_transformer`;
its two W4 binaries come from the current source build, while the existing
RMSNorm and other operators retain their qualified implementations. See
`artifacts/minimax-m31-serving-20261024/merged-opp.json` for hashes.

Earlier 16×128 kernels failed the NZ256 layout and concentrated-routing checks.
Do not mix them with this launch configuration. Current canonical and NZ256
512-route probes passed independent CPU comparisons, including all routes
selecting one expert (maximum normalized RMS error about 1.5e-5).

## Deployment

With the complete converted checkpoint installed, run from `/workspace`:

```bash
source /workspace/minimax-m31-env.sh
export ASCEND_CUSTOM_OPP_PATH=/vllm-workspace/minimax-m31-merged-opp/vendors/custom_transformer
export LD_LIBRARY_PATH=$ASCEND_CUSTOM_OPP_PATH/op_api/lib:$LD_LIBRARY_PATH
cd /workspace
export VLLM_SERVER_DEV_MODE=1
export VLLM_PP_LAYER_PARTITION=21,20,19
vllm serve /srv/ai/storage/models/MiniMax-M3.1-Flash-W4 \
  --served-model-name minimax-m31-flash \
  --load-format minimax_m3_w4 --dtype half \
  --model-loader-extra-config '{"expert_layout":"nz256","max_grouped_routes":512}' \
  --tensor-parallel-size 2 --pipeline-parallel-size 3 \
  --enable-expert-parallel --enforce-eager --block-size 128 \
  --max-model-len 8192 --max-num-seqs 1 \
  --max-num-batched-tokens 128 --gpu-memory-utilization 0.95 \
  --limit-mm-per-prompt '{"image":1,"video":0}' \
  --mm-processor-kwargs '{"max_long_side_pixel":448}' \
  --no-async-scheduling --disable-custom-all-reduce \
  --worker-extension-cls tools.minimax_m3.resident_worker.MiniMaxResidentExtension \
  --middleware tools.glm_perf.resident_middleware.ResidentControlMiddleware \
  --reasoning-parser-plugin /vllm-workspace/vllm-ascend/tools/minimax_m3/reasoning_parser.py \
  --reasoning-parser minimax_m31 --port 8000
```

Keep the pinned `chat_template.jinja` alongside the converted tokenizer files.
The loader checks payload hashes and rejects missing parameters. Do not bypass
those checks to get startup to complete.

## Functional verification

```bash
curl http://127.0.0.1:8000/v1/models
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"minimax-m31-flash","messages":[{"role":"user","content":"Say hello in one short sentence."}],"temperature":0,"max_tokens":64}'
```

`python -m tools.minimax_m3.http_probe --model minimax-m31-flash --output /workspace/m31-http`
records streaming responses, first-output time and total time. Its nonempty
HTTP output gate is not a coherence score. Add `--image /workspace/red.png` for
an image request, or `--long-context-tokens 8190 --case context_boundary` for
the 8K cache boundary. The latter repeats a token and is not a comprehension test.

## Accuracy evaluation

Run the reproducible small coherence suite against the loaded service:

```bash
python -m tools.minimax_m3.coherence_probe \
  --image /workspace/red.png --extended --max-tokens 128 \
  --output /workspace/m31-coherence
```

The image must be the solid red fixture. Nine checks passed with the optimized
expert path. Real-layer component probes also compared routing, sparse block
selection and attention to CPU references. The full-model raw single-token
comparison retained the CPU top-10 order, with maximum log-probability difference
about 0.0085. The optional original-FP4 CPU expert comparison omits source
activation quantization and is not a native-release quality oracle.

Use templated chat. An untemplated raw completion was repetitive and is not
qualified. The `minimax_m31` reasoning plugin limits end-marker detection to
the current assistant turn; upstream detection saw older template end markers
and leaked `</mm:think>` into streamed answers. No lm-eval score is claimed.

## Performance

Real model memory per rank was approximately 37.79, 38.26 and 37.55 GiB in the
three pipeline stages, before runtime/cache allocations. Requests below used
one sequence, greedy decoding, 128-token prefill chunks and the same prompts.

| Request | Canonical/old kernel | NZ256/current kernel |
| --- | ---: | ---: |
| Greeting, 201 input / 4 output tokens | 72.31 s | 8.75 s |
| France capital, 20 output tokens | 74.88 s | 15.26 s |
| 17 + 25, 12 output tokens | 59.45 s | 11.01 s |

These are individual elapsed request measurements, including reasoning tokens,
not sustained throughput averages. A 3443-token sparse retrieval returned the
correct two-word key in 71.18 s; the 7943-token version passed in 168.99 s. A real-expert bank microbenchmark at 128 tokens
improved from 1.81 s to 0.134 s with the current kernel, 512-route batching and
NZ256 packing; this is not a full-model speedup measurement.

A short-context attention fast path passed isolated CPU/NPU comparisons but
failed full-model greeting coherence. It is rejected from the serving baseline.
Retain full-model checks when changing reductions, masks or block selection.

## Resident Python and CANN reload

Real-weight validation passed module replacement, direct native activation and
rollback on all six ranks without changing worker PIDs or weight storage. Python
and native switches on the idle server took about 0.15 s each, excluding kernel compilation and
qualification; rollback took 0.14 s. See `resident-native-gate/report.json` and
its per-rank receipts in the evidence directory.

The launch above installs resident controls once. Run control commands on the
serving host; public inference remains available, while administrative routes
are restricted to loopback. The upstream development API can execute Python;
keep the supplied middleware enabled and do not expose an unguarded dev server.

```bash
python -m tools.minimax_m3.resident status
python -m tools.minimax_m3.resident switch \
  --module-file /path/to/candidate_minimax_ops.py
# Restore startup Python dispatch without rereading any checkpoint shard.
python -m tools.minimax_m3.resident switch
```

The module file uses the same functions and `PackedM3Experts.forward` contract as
`vllm_ascend/_310p/ops/minimax_m3.py`. Source is compiled freshly in a private
namespace. The transaction patches both the definitions and model call-site
aliases; it does not reconstruct model instances or reload modules in place.

Build a new direct CANN kernel and uniquely named bridge while serving continues:

```bash
python -m tools.minimax_m3.build_resident_native /workspace/m31-native-v4 \
  --namespace minimax_w4_v4 --source-root /vllm-workspace/vllm-ascend/csrc
python -m tools.minimax_m3.probe_resident_native \
  /workspace/m31-native-v4/manifest.json --device npu:0 \
  --output /workspace/m31-native-v4/standalone.json
python -m tools.minimax_m3.resident load-native \
  --manifest /workspace/m31-native-v4/manifest.json
python -m tools.minimax_m3.resident switch --native-name minimax_w4_v4
```

Loading and activation are separate. Each rank verifies artifact hashes, loads
its own binary handle and validates a CPU numerical reference. Compatible
changes drain requests, clear scheduler/physical caches, and check all six
worker PIDs and weight-storage fingerprints. Versioned code remains loaded
until process exit; rollback changes dispatch and never unloads live code.
Native registration/device failures keep serving paused and require inspection;
a poisoned worker needs restart. Ordinary Python preparation failures leave
existing dispatch intact. Layout, topology, model-construction and allocation
changes still require startup. This path currently requires eager execution.

Every switch replaces the entire prior candidate. To keep both a Python module
and native kernel active, supply both `--module-file` and `--native-name` in the
same command. The CLI serializes controllers and blocks new public requests
during a transaction. After a failed mutation, restore a valid generation and
use `resident resume`; do not resume disagreeing ranks.

Evidence and launch scripts are in `artifacts/minimax-m31-serving-20261024/`.
