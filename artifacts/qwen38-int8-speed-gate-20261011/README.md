# Qwen TP6 INT8 speed gate, October 11, 2026

Controlled testing found real projection gains, but has **not** established a
whole-model INT8 speedup. The live experimental 1M YaRN server retains its
qualified W4A8 backend. No serving worker was patched, restarted, paused, or
asked to clear its prefix cache during these measurements.

## Live serving baseline

The existing real checkpoint runs on six 310P3 chips with TP6/EP6, MTP2,
FULL_DECODE_ONLY graphs, two active request slots, and a 1,048,576-token request
limit. Full million-token language quality remains unqualified. Kilo traffic
was stopped for this session. The API PID remained 3024361 throughout testing.

| Measurement | Result |
| --- | --- |
| Three coding prompts, 256 output tokens each | 30.45, 27.65, 27.67 tok/s; median 27.67 |
| Median first-token latency, those prompts | 0.450 s |
| Serial 256-token prompt, 1,024 output tokens | 29.16 tok/s |
| Same workload with a separate cold 8K prompt | 17.64 tok/s |
| Largest SSE chunk gap during mixed prefill | 0.659 s |
| Cold 8K request during that generation | 63.49 s to first token |
| Separate cold 8K request, no competing request | 18.43 s to first token |
| Identical 8K repeat | 18.67 s; **zero cached tokens** |
| Separate cold 32K / identical repeat | 75.33 / 75.56 s; both zero cached tokens |

Client decode rates use `(completion_tokens - 1) / decode_duration`. SSE chunks
can contain several speculative tokens. Server request-timing evidence also
records computed and cached prompt counts. These are serving diagnostics, not
language benchmark scores. The short coding median does not qualify the 50%
whole-model improvement target over the older two-card baseline.

The identical 8K request is deliberately reported as a repeat, not a cache-hit
measurement. Its receipt has `cached_tokens: 0`; no prefix benefit is claimed.
The `cold-cached` filename came from the requested test mode before that result
was known. Raw receipts preserve the original name and contents.

## Matched projection measurements

Real layer-0/expert-0 checkpoint weights were used for fused gate/up
`[1280, 2560]` and down `[2560, 640]`. A separate run used layer 47/expert 511
on chip 4. Random FP16 inputs use a fixed seed and the qualified G128 max/127
activation policy. Both candidates and the native operator include activation
quantization in the measurements below. Routing, SwiGLU, final route reduction,
other model layers, and communication are excluded.

Each timing has five trials of 30 repetitions, with ten operations in each
captured graph to amortize Python replay dispatch. Tables use NPU event medians;
receipts also include host wall medians and every trial. The traffic guard
requires zero live requests before and after the interval and an unchanged
completed-request counter. It never pauses the server.

| Projection / rows per expert | Native W4A8 ms | Raw INT8 M32 ms | Packed lookup INT8 M32 ms |
| --- | --- | --- | --- |
| Gate/up / 3 | 0.0465 | 0.0295 | 0.1698 |
| Gate/up / 32 | 0.1191 | 0.0657 | 0.2060 |
| Gate/up / 128 | 0.4567 | 0.2374 | 0.3747 |
| Gate/up / 512 | 1.5775 | 0.8794 | 1.0136 |
| Down / 3 | 0.0173 | 0.0179 | 0.0888 |
| Down / 32 | 0.0874 | 0.0332 | 0.1034 |
| Down / 128 | 0.3294 | 0.1143 | 0.1835 |
| Down / 512 | 0.6543 | 0.4294 | 0.4992 |

The packed lookup candidate's matched native references are in its own receipt;
small differences from the table's raw-run references are normal between runs.
At 512 rows, it reduces gate/up latency by 35.7% and down latency by 25.0%.
At 128 rows, the reductions are 18.0% and 44.3%, respectively. At three rows,
both packed projections are substantially slower. This is a **prefill candidate**,
not a replacement for the current decode path.

The original raw INT8 M16 prototype reproduced on two chips and widely
separated checkpoint layers: fused gate/up at three rows improved from about
0.0465 to 0.0256 ms, and down at 128 rows from 0.3291 to 0.1773 ms.
The larger M32 tile and vector scale broadcasting repair its large-batch
regression. Raw INT8 at 512 rows now improves gate/up about 44% and down about
34%, but still requires twice the weight bytes.

Rows in these tables mean **rows assigned to one expert**, not the scheduler's
whole prefill batch. With 512 experts and top-10 routing, a 2,560-token prefill
would average about 50 rows per expert under uniform routing. Actual route
histograms must determine dispatch; a global batch-size threshold is insufficient.

## Weight-reuse sensitivity

The final v9 build cycles sixteen equal-valued weight banks at distinct device
addresses, with one graph containing all sixteen operations. The final
comparison also uses distinct candidate output buffers, matching the native
operator's separate outputs. Three trials of
20 repetitions pass exact reference and replay checks. The packed M32 lookup
path uses layer 47/expert 511 on chip 4; its gains persist:

| Rows per expert | Gate/up latency reduction | Down latency reduction |
| --- | --- | --- |
| 3 | -230.7% | -436.8% |
| 128 | 17.7% | 43.1% |
| 512 | 33.1% | 23.2% |

Raw M16's three-row gate/up reduction falls to 27.2%, and three-row down is
28.5% slower than native W4A8. This supports retaining the compact native
backend for decode. Cycling addresses tests the reuse assumption, not cold
memory traffic or the full diversity of real routed experts. Use the two `rotate16-distinct-output-v9.json` receipts for the matched final
comparison. Earlier `rotate16-v9.json` receipts reused one candidate output
buffer and are retained as preliminary evidence, not the final comparison.

## Math, transfers, and storage

The native backend expresses A8 as two INT4 limbs. The prototype instead
computes one INT8-by-INT8 G128 dot with exact INT32 accumulation. It retains
checkpoint FP16 scales, applies weight then activation scaling in FP32,
accumulates groups in order, and rounds to FP16 at the same output boundary.
All measured cases match the independent ordered reference and native W4A8
operator exactly, including graph replay. Tail batches 1, 3, 17, and 33 passed.
The final packed M32 kernel passed rows 3, 17, 33, and 128 on all six chips,
for both fused gate/up and down projections (48 exact cases).

Raw INT8 expands `q - offset` losslessly before capture and doubles global
weight bytes. Replacing a full down bank adds about 3.35 GB per chip; keeping
both old and new banks adds about 6.7 GB. An all-projection shadow bank is
incompatible with the live 1M profile's headroom. No shadow bank was installed.

The first packed decoder used INT16 masks, FP32 plane interleaving, and several
casts. It was correct but slower at every tested size. Its replacement uses a
1 KiB immutable table of 256 signed FP16 nibble pairs. A 32-bit Gather copies
those pair bits, followed by exact FP16 offset subtraction and INT8 conversion.
It eliminates the two decode planes and their interleave operations.

The lookup decoder uses 70,304 bytes of bounded UB scratch; the M32 projection
adds 77,760 bytes, for 148,064 bytes total. Its maximum resident L1 tile is
413,696 bytes. Packed checkpoint bytes and metadata remain in global memory;
expanded weights remain on core and are reused across the expert's row tiles.
There is no hot-path device-to-host tensor inspection. No measured bus-byte or
thermal reduction claim follows from these operator timings.

The M32 input load explicitly visits both M16 row blocks with the correct K32
stride. Scale instructions iterate the shorter of the row and output-block
axes. Weight scaling still precedes activation scaling, and the complete K
axis is accumulated before output storage.

## Reproduction and provenance

Use the pinned serving environment and coherent OPP stack. The captured wrapper
is [run-int8-component.sh.txt](run-int8-component.sh.txt); its host copy is
`/srv/ai/src/qwen-yarn-1m-tp6-20261011/run-int8-component.sh`. All probe libraries
live in append-only build directories and were loaded only in isolated probe
processes, never in serving workers.

```bash
bash /srv/ai/src/qwen-yarn-1m-tp6-20261011/run-int8-component.sh \
  --build-dir /srv/ai/src/qwen-int8-projection-probe-20261011-v9 \
  --output /srv/ai/src/qwen-int8-speed-gate-20261011/new-projection-receipt.json \
  --device 0 --tokens 3 128 512 --projections gate_up_proj down_proj \
  --fused-quant --compare-w4 --packed-weights --lookup-decode \
  --row-tile 32 --graph-unroll 10 --time
```

Omit `--packed-weights --lookup-decode` to test raw INT8. Use `--row-tile 16`
for the small-row raw variant. `--weight-rotations 16 --graph-unroll 1` cycles
sixteen equal-valued banks at distinct addresses to test sensitivity to weight
reuse. It does not model diverse router choices or prove cold memory traffic.
Every output path must be new; the tool refuses to overwrite evidence.

Build v3 is raw M16, v5 is the mask decoder, v6 adds M32 and vector scaling,
v8 adds lookup decoding, and v9 contains the formatted final implementation.
The final build provenance records frozen source, binary, and bridge hashes.
Later receipts also embed that provenance and all command arguments.

## Validation and next integration gate

Focused CPU checks passed 99 tests, including signed-nibble decode/layout,
M16/M32 input loads, quantization parity, workspace, rotary, and serving probe
checks. Scoped manual pre-commit checks passed after formatting. The previous
repository-wide formatting run has unrelated baseline failures; this session
does not claim the entire repository is lint-clean.

The next serving change should use measured per-projection, per-expert row counts to retain
the native decode path and select packed INT8 only where it wins. It still
requires grouped/routed execution, exact route order, fused gate/up and down
integration, real text/image/MTP checks, mixed-request measurements, and a
full serving A/B before admission. Repeated single-expert projections cannot
establish the requested end-to-end improvement. The live server retains its
existing graphs, checkpoint, thermal controller, and maximum context.
