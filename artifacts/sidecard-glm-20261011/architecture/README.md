# GLM architecture measurements, October 11, 2026

The RTX wins the measured resident arithmetic cases. The placement problem
is memory residency and transfer cost, rather than a complementary Ascend
compute advantage demonstrated by this sweep. This is a measured direction
for further work, not a newly deployed service speedup.

## Hardware and method

One RTX PRO 6000 Blackwell, SM120, and physical Ascend 310P3 device 2 were
measured on `192.168.53.187`. Device 3 was released with the GLM server but
was not used by the probes. Physical devices 0 and 1 retained the Qwen
workers throughout; [before](npu-before.txt), [isolated](npu-isolated.txt),
and [after](npu-after-bench.txt) snapshots identify the processes. The RTX's
existing 144-expert banks remained resident throughout.

Five warmups and eleven samples per case. JSON receipts retain device-event
and elapsed-call samples, not just means. Inputs and weights are resident
outside the arithmetic timers. Event intervals include stream idle time
caused by host dispatch; they are not instruction-only GPU cycle counts.
Wall times include launch, timing events and synchronization.

- [CUDA primitives](cuda.json): 43 cases, cuBLAS and a CUDA mHC post kernel.
- [Ascend primitives](npu-2.json): 77 cases, eager Torch, ND and NZ GEMM weights.
- [CUDA experts](cuda-experts.json) and [Ascend experts](npu-experts.json):
  12 paired cases each, authenticated real weights in layers 3, 8 and 33.
- [CUDA cache-flushed experts](cuda-experts-sm-flushed.json): twelve cases,
  with an SM kernel sweeping 256 MiB before each timed call. RTX L2 is 128 MiB.
- [Cold streaming](cuda-streaming-idle.json): 24 pageable/pinned cases with
  CPU gather, upload, lossless Cube packing and expert arithmetic included.
- [Transfers](transfer.json): four bitwise-checked socket/copy cases, using
  one FP16 input row and one FP32 partial row per token.

Synthetic GEMMs validate sampled CPU FP32 results, with absolute/relative
tolerances of 0.01/0.03 and recorded errors. FP16 GEMM outputs are FP16.
CUDA FP32 GEMMs use pedantic FP32, with TF32 excluded. Ascend uses the
qualified runtime's default precision policy. This tests runnable libraries
and dispatch paths, not equal assembly implementations or peak ISA limits.

Attention measures the QK/softmax/AV core with 32 heads, head width 256,
and a shared 2,048-entry selected KV window. It excludes indexing, paged KV
gathers, causal masks and the serving native attention path. Its ratio must
not be used as a prediction of deployed DSA or KDA speed. Native KDA,
long-context state capacity, HCCL and end-to-end placement remain separate
measurements. Short repeats of resident weights also do not represent
cycling through the entire model's weights.

## Resident arithmetic

Median elapsed call time in milliseconds. Ascend GEMM rows show the faster
measured ND/NZ variant for that individual case. These are equal matrix
shapes; TP2 and full-model cases are labeled separately in the receipts.

| Operation | Tokens | RTX | Ascend | Ascend / RTX |
| --- | ---: | ---: | ---: | ---: |
| KDA QKV/BFG projection, TP2 | 1 | 0.0362 | 0.7026 | 19.4 |
| KDA QKV/BFG projection, TP2 | 512 | 0.1803 | 1.4946 | 8.3 |
| KDA output projection, TP2 | 1 | 0.0241 | 0.3586 | 14.9 |
| KDA output projection, TP2 | 512 | 0.0756 | 0.7528 | 10.0 |
| LM head, 77,440 output channels | 1 | 0.4007 | 3.3232 | 8.3 |
| mHC projection, FP32 inputs | 1 | 0.0187 | 0.2061 | 11.0 |
| mHC post-mixing | 1 | 0.0108 | 0.2426 | 22.5 |
| mHC post-mixing | 512 | 0.0335 | 4.6104 | 137.6 |
| Selected attention core | 1 | 0.0353 | 2.4667 | 70.0 |
| Selected attention core | 512 | 0.3113 | 10.8765 | 34.9 |
| Device copy, 256 MiB | — | 0.3739 | 2.8449 | 7.6 |

## Real routed experts

Same 72 real experts, same deterministic FP16 inputs, two local routes per
token, top-eight weights of 1/8, A4 activations and original FP16 scales.
Both include quantization, gate/up, SwiGLU, down and FP32 weighted reduction.
CUDA explicitly calls the deployed scalar packed C ABI. Upload, download,
host route sorting, RPC and HCCL are excluded. Ascend host dispatch inside
the native helper is included. No requantization or precision reduction.

All twelve paired outputs pass 0.002 absolute/relative tolerance. Eleven
are exactly equal; the W2 512-token case has maximum difference 0.00024414.

| Weights | Tokens | RTX ms | Ascend ms | Ascend / RTX |
| --- | ---: | ---: | ---: | ---: |
| W4, layer 3 | 1 | 0.0679 | 1.4400 | 21.2 |
| W3, layer 8 | 1 | 0.0671 | 2.2520 | 33.6 |
| W2, layer 33 | 1 | 0.0663 | 1.9324 | 29.1 |
| W4, layer 3 | 4 | 0.0710 | 3.8501 | 54.2 |
| W3, layer 8 | 4 | 0.0695 | 7.1545 | 102.9 |
| W2, layer 33 | 4 | 0.0697 | 6.0255 | 86.4 |
| W4, layer 3 | 512 | 2.7026 | 41.8267 | 15.5 |
| W3, layer 8 | 512 | 2.6738 | 69.4925 | 26.0 |
| W2, layer 33 | 512 | 2.5488 | 59.0361 | 23.2 |

These repeated-layer CUDA measurements benefit from warm caches. Sweeping
256 MiB through SM loads/stores before each call, outside the timer, gives
the more conservative one-token CUDA times below. This is a CUDA cold-cache
comparison against the same warm Ascend results, not a whole-model replay.
All twelve flushed outputs are bitwise equal to the original CUDA outputs.

| Weights | Cache-flushed RTX ms | Ascend ms | Ascend / RTX |
| --- | ---: | ---: | ---: |
| W4 | 0.1301 | 1.4400 | 11.1 |
| W3 | 0.1311 | 2.2520 | 17.2 |
| W2 | 0.1294 | 1.9324 | 14.9 |

The one-token Ascend expert cost is 1.4–2.3 ms, not the roughly 6 ms
four-token W2 result. Multiplying the latter by 42 layers overestimates its
contribution to single-token decode. The raw gap is substantial, but the
backbone and orchestration still need their own critical-path measurements.

## Work backward from the measurements

1. Prefer resident CUDA execution for the measured projections and experts.
   Measure lossless compact W2/W3 storage and a bounded expert cache before
   making another static ownership expansion. The current CUDA representation
   stores every weight as a four-bit word-plane value, which expands W2/W3.
2. Keep cold-expert fallback on Ascend for now. Cold streaming must beat the
   maximum of two simultaneously running Ascend ranks, plus coordination;
   beating one isolated NPU call does not establish an end-to-end gain.
   All measured four-token cold transfers regress. W2 prefill improves against
   one chip, but W3/W4 regress. Neither comparison includes two-rank overlap.
3. Optimize transfer and dispatch alongside placement. The conservative
   integrity-checked socket round trip is 0.448 ms at one token and 33.268 ms
   at 512 tokens. It includes extra copies, hashing and checks, so it is not
   the production protocol's intrinsic latency. PCIe directions are reported
   separately in the transfer receipt.
4. Ascend's useful role is currently additional resident memory and the
   existing attention/state implementation. Measure native recurrent and
   paged attention paths before treating state movement as an optimization.
   This sweep does not establish an Ascend compute win at those operators.

The first cold sweep, [recorded here](cuda-streaming.json), overlapped server
loading. The idle repeat linked above is the placement evidence; all cold
outputs remain bitwise equal to their resident CUDA outputs. Neither sweep
is a deployed cache or a claimed serving improvement.

Median complete pinned cold calls with the service idle, in milliseconds:

| Weights | Tokens | Cold RTX | Resident Ascend, one chip |
| --- | ---: | ---: | ---: |
| W4 | 1 | 1.951 | 1.440 |
| W3 | 1 | 1.369 | 2.252 |
| W2 | 1 | 0.988 | 1.932 |
| W4 | 4 | 11.856 | 3.850 |
| W3 | 4 | 8.762 | 7.154 |
| W2 | 4 | 6.204 | 6.026 |
| W4 | 512 | 100.410 | 41.827 |
| W3 | 512 | 75.076 | 69.493 |
| W2 | 512 | 54.055 | 59.036 |

## Restored service and checkpoint repair

Restart found one checksum mismatch in 190 checkpoint files. The local
original shard matched the expected SHA exactly. The bad host shard was
preserved and both linked checkpoint paths were repaired with those original
bytes. Checkpoint manifests, quantization and CUDA banks are unchanged.
See [audit](checkpoint-audit.json) and [repair receipt](checkpoint-repair.json).

The restored deployment is
`/srv/ai/storage/artifacts/sidecard-glm-service-20261011-one-card-v14`.
Its [process receipt](service-process.json) and
[operational validation](restored-service-validation.json) record seven
passed API cases and 168 zero-error serialized/overlap checks across all
42 routed layers. Full validation remains false because greedy repeat
variation is unresolved. The 260,352-token limit is advertised capacity;
full-context prefill completion has not been qualified.

The [completed 2K-prompt/64-output test](restored-decode-64.json) measured
3.57 decode tokens/s, excluding first-token/prefill time. First text took
32.756 seconds. This is a restored baseline, not an architecture-placement
speedup. The other physical Ascend card remains assigned to Qwen.

## Reproduce

Run only in an explicitly admitted hardware maintenance window. The NPU
commands require visibility `2,3` and use logical device 0, physical device 2.
Use the qualified runtime environment and preserve its existing PYTHONPATH.
GPU and NPU benchmarks run in separate processes; the CUDA process uses CPU
Torch plus libcudart/cuBLAS, with no CUDA Torch installation required.

```bash
nvcc -O3 --fmad=false -std=c++17 -arch=sm_120 -shared -Xcompiler -fPIC \
  tools/sidecard/architecture_bench.cu -lcublas -o /tmp/architecture-bench.so
python -m tools.sidecard.architecture_benchmark --backend cuda \
  --library /tmp/architecture-bench.so --output /tmp/cuda-new.json
python -m tools.sidecard.architecture_benchmark --backend npu \
  --output /tmp/npu-new.json
python -m tools.sidecard.architecture_experts --backend cuda \
  --checkpoint /path/to/authenticated-native-checkpoint \
  --checkpoint-sha256 EXPECTED_MANIFEST_SHA --library /path/to/cuda-packed.so \
  --output /tmp/cuda-experts-new.json --comparison-dir /tmp/expert-reference \
  --cold-streaming
python -m tools.sidecard.architecture_experts --backend npu \
  --checkpoint /path/to/authenticated-native-checkpoint \
  --checkpoint-sha256 EXPECTED_MANIFEST_SHA \
  --output /tmp/npu-experts-new.json --comparison-dir /tmp/expert-reference
```

For the CUDA cache-swept comparison, replace `--cold-streaming` with
`--flush-cuda-cache --cache-flush-library /tmp/architecture-bench.so` and
use fresh output/comparison paths. The SM sweep is excluded from timing.

The [source receipt](source-receipt.json) separates executed snapshots from
the final reproduction tools. Final CUDA guards also rejected five invalid
geometry cases, and a twelve-case one-token rerun passed. Focused offline
tests: 365 passed with `--confcutdir=tests/ut/sidecard`; the broad repository
conftest cannot load the installed upstream Flash Linear Attention module.
Changed-file formatting passes. The full repository formatting command has
seven pre-existing failing hooks and changed no files owned by this work;
see the [test receipt](test-receipt.json).
