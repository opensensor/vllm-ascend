# Lossless compact CUDA expert storage

The final candidate preserves native W2/W3 code widths instead of expanding
every CUDA bank to W4. W2 uses one aligned 32-bit word for sixteen codes; W3
uses three aligned 16-bit planes. W4 retains the qualified nibble layout.
FP16 scales, activation quantization, integer dot products, ordered FP32
block accumulation, SwiGLU, and FP32 slot reduction retain their contracts.

## Hardware qualification

`paired-v4.json` contains 48 paired cases against the deployed packed scalar
CUDA library. It covers authenticated real W4/W3/W2 layers, 72 experts per
layer, A4/A8 activations, 1/4/128/512 tokens, and warm/cache-swept execution.
Every raw FP16 expert row and weighted FP32 partial matched bitwise. Native
GPU packing also matched independent CPU unpacking/encoding for the first
and last selected expert in both projections of all three layers.

Cache sweeping uses a 256 MiB SM read/write kernel outside the timed interval,
twice this RTX PRO 6000's 128 MiB L2. Times include activation quantization,
gate/up, SwiGLU, down, and weighted reduction, and exclude route gathering,
upload, RPC, and download. The table shows paired A4 wall medians in milliseconds:

| Precision | Tokens | Packed baseline | Compact | Time reduction |
| --- | ---: | ---: | ---: | ---: |
| W3 | 1 | 0.1278 | 0.1251 | 2.1% |
| W3 | 128 | 0.7959 | 0.6975 | 12.4% |
| W3 | 512 | 2.6770 | 2.4076 | 10.1% |
| W2 | 1 | 0.1317 | 0.1112 | 15.6% |
| W2 | 128 | 0.7807 | 0.5386 | 31.0% |
| W2 | 512 | 2.5595 | 1.8636 | 27.2% |

`boundary-gate.json` additionally qualifies the serving packet boundary for
144 experts, 128/512/640 tokens, and 1/2/8 routes per token: all 27 cases matched
bitwise, including 5,120 routed rows and FP32 partial replies. Its reference
uses the older INT8 row-major path in 32-token packets; its timing ratios
must not be presented as improvements over the current serving baseline.

The compact library SHA256 is
`3dc9b9dabaf51c1f2b048460e2c19b8ad10a6a29d2706a0d9ca53a898c6181a7`.
The deployed packed reference SHA256 is
`ffc6fedbbecb67bb0a4453aef5b9d856b76380dbce51fd4eb69557e19f16c4fd`.
All libraries and checkpoint identities are recorded in the receipts.

## Capacity and preparation

The 144-expert, 42-layer partition has 22 W4, eight W3, and twelve W2 layers.
Exact code storage falls by **13.5 GiB**. Scales and scratch remain separately
accounted. Admission includes the largest reusable native gate/up staging
bank, bounded scratch, and the existing 4 GiB device reserve. Ownership stays
at 144 CUDA experts plus 72 on each chip of the single Ascend card.

GPU preparation translates authenticated native Cube bytes directly into the
compact layout. In the final 72-expert gate, preparation took about 1.01 seconds
for W3 and 0.36 seconds for W2, including weight and scale uploads. An earlier
CPU unpack/repack adapter required about 25 seconds for the W3 bank.

This capacity creates room for a future bounded expert cache. Cache routing
and wider CUDA ownership are separate changes and are not implemented here.
Microbenchmark gains are not end-to-end service gains.

The completed V15 2,048-token cold prompt and 64-token fixed decode measured
3.6687 decode tokens/s, 32.7847 seconds to first text, and 49.9570 seconds total.
The prior V14 record was 3.5733 decode tokens/s. This is a modest observed
difference, not a controlled attribution of the entire change to the new kernel.
The optional 8K probe was canceled when the user requested larger optimizations.

## Running service

Deployment V15 uses physical Ascend devices 2/3 plus the RTX at
`http://192.168.53.187:8001/v1`, model `glm-sidecard`. CUDA resident allocations
are **62,059,503,616 bytes (57.8 GiB)**, exactly 13.5 GiB below the previous
daemon. Device-wide usage fell from 73,706 to 59,882 MiB; free GPU memory is
37,458 MiB. Configured context capacity remains 260,352 tokens. Full-capacity
prefill has not been qualified; prior 8K qualification remains the baseline.

`api-validation.json` records seven successful operational API cases and real
CUDA routes in all 42 layers. All 168 serialized/overlapped decode and prefill
shadow checks have zero error. The full validation flag remains false because
greedy repeated completion varies, as it did on the prior baseline. Compact
storage does not resolve that existing model/runtime limitation.

The other physical Ascend card was not signaled or reset. Its existing worker
PIDs remained running. Both old GLM process commands were checked against
saved receipts before SIGTERM. A separate baseline rollback deployment was
staged before the replacement. Process/configuration receipts are retained
alongside the arithmetic gates.

## Earlier candidates and reproduction

`paired-v1.json` preserves the rejected dynamic byte-plane unpacker, which
regressed W3 cache-swept decode by about five times. V2 specializes precision
but still regresses decode. V3 uses aligned 16-bit planes and improves W3;
its W2 decode regression motivated V4's single 32-bit load. Every variant
passed all 48 arithmetic checks. Executed sources are saved under
`executed-source/`; earlier snapshots use different packing formats.

Build the final source without floating-point reassociation:

```bash
nvcc -O3 --fmad=false -std=c++17 -arch=sm_120 -shared -Xcompiler -fPIC \
  tools/sidecard/cuda_compact.cu -o cuda-compact.so
```

Run `python -m tools.sidecard.compact_gate --help` for the paired hardware gate.
It requires pinned hashes for the checkpoint, both expert libraries, and the
SM cache-sweep library. It allocates one bounded layer per backend at a time
and never selects an NPU. The serving layout is `k32_words_native_n_v1`, with
explicit one-card mode, tensor transport, FP32 partial reduction, no INT8 bank
cache, and a complete pinned 27-case boundary receipt.
