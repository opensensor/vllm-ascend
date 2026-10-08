# GLM FP16 expert scale storage

The candidate persists block32 expert scales as FP16 and promotes each loaded
tile into the existing FP32 cache. Scale multiplication, integer Cube products,
accumulation order, SwiGLU and route reduction retain their existing arithmetic.
The scale storage marker is `fp16_storage_fp32_compute_v1`.

The CPU exporter is `tools.glm_perf.prerounded_scale_checkpoint --fp16-storage`;
matching builders use `--fp16-weight-scales`. Config, native manifest, shard
metadata and resident bank markers agree. The loader validates dtype before
placement, and native dispatch rejects mismatched scale buffers before submission.
The FP16 and prerounded-FP32 builder options are mutually exclusive. Export can
start with raw or previously rounded FP32 scales. Unchanged codes and dense
weights are hard linked; no unpacking or live weight transformation is required.

Production K2048/K4096 scale rows use aligned DMA and one vector promotion per
tile load. Sub-DMA synthetic/ragged rows use bounded scalar loads to avoid reading
past the final expert. FP32 scale cache and half staging remain disjoint from
retained gather indices, including the cached gate/up schedule.

The real checkpoint was exported on the NPU host to:
`/srv/ai/models/GLM-5.3-Flash-native-int4-fp16-scales-20261008`.
Its 37,152 authoritative scale tensors contain 608,698,368 FP16 payload bytes,
half the previous FP32 payload. Resident scale capacity saves **145.125 MiB per
rank** across target and draft, or 580.5 MiB across TP4. Hard-linked source shards
can also contain superseded scales, so this is not a claim that total checkpoint
files shrink by that amount.

## Validation

- Final compatible CPU GLM suites: **1,643 tests passed**; focused final
  regressions: **102 passed**. Assembly, grouped gate/up and KPool ops suites
  require dependencies unavailable on the local host and remain excluded.
- Every finite FP16 bit pattern, signed zero, rounding ties, overflow rejection,
  raw/rounded source exports, unchanged-byte identity, actual loader iterator,
  resident FP16 bank allocation and mismatched ABI rejection are checked.
- M16 decode and M32 prefill builds v964/v965 passed 30 synthetic and 12 real
  expert arithmetic/replay cases each (W2/W3/W4, A4/A8, real K4096/K2048).
- Earlier v962/v963 paired checks matched the FP32-storage pipeline bit for bit
  at 2/8/17/640 tokens, including changed graph inputs. Final v964/v965 kernel
  binary hashes match those earlier builds; the only C++ cleanup removed a
  duplicate compile-time admission guard.
- Previously queued compact-W4 and rounded-FP32 storage comparisons were also
  executed. Route-column v913→v952 and v919→v953 pairs passed as well.
  Across four features and both schedules, **192 paired cases** passed bitwise
  equality and changed-input graph replay. These are operator checks, not
  full-model throughput qualification.

The first real one-expert stage profiles show small, mixed changes. W4/A4 gate/up
was approximately 0.322 to 0.304 ms in that run; W3/A8 gate/up was approximately
0.882 to 0.897 ms. This uses synthetic activations and one expert, and cannot
establish a full-model throughput gain. The full TP4/MTP1 server loaded the permanent FP16 checkpoint and captured the
paired v964 decode/v965 prefill schedules. All four rank receipts confirm
43 resident banks, FP16 scale dtype and 152,174,592 scale bytes per rank. Serving
requests completed with 11 piecewise 640-token prefill replays and no native fallback.
On synthetic token-id prompts, c1 generation measured **8.47–8.76 tok/s**,
640-token cold TTFT **6.94 s**, and 6,400-token cold TTFT **80.93 s**. C4 total
request throughput was **8.87 tok/s including prefill**; its per-request generation
rates were 2.70/4.39/4.54/4.54 tok/s. These are absolute measurements after a fresh
server load, not a paired speedup claim. The language sample exhausted its
48-token limit in the reasoning field; language quality remains unevaluated.

The queued v954 query converter failed its exhaustive device gate and was not
selected. Diagnosis found that the dav-m200 Compare API truncates partial
256-byte repeats, leaving short-vector masks uninitialized, and NE(x,x) does
not implement the required NaN classification. The follow-up pads UB comparison
work to whole repeats and classifies finite exponent/mantissa fields instead.
Only owned DMA output padding is written. The corrected v966 and final v967
builds passed all 13 device gates, including every FP16 bit pattern, NaNs, short
tails, input guards, output padding and changed-input graph replay. Rebuilding
Indexer's authoritative forward also preserves its permanent RoPE conversions;
a regression covers the separate query and legacy converter bindings. V966
serving measurements preceded this binding fix and are not the final candidate.
The final v967 is installed across all 12 target/draft indexers alongside v964/v965.
Final combined serving measurements were c1 **8.47–8.68 tok/s**, cold 640-token
TTFT **6.75 s**, and cold 6,400-token TTFT **78.64 s**. C4 total request throughput
was **8.68 tok/s including prefill**, with per-request generation rates
2.73/4.12/4.25/4.25 tok/s. Both MoE dispatch schedules were exercised, and all four
workers reported valid graphs, no native failure and native query calls. These
runs do not establish a paired full-model speedup or a language quality result.
The 96-token chat sample included readable reasoning and began its answer but
hit the token limit; its raw response is archived without a quality pass claim.
The qualified archives contain frozen helpers and checksummed native binaries.

## Reproduce the permanent format

```bash
python -m tools.glm_perf.prerounded_scale_checkpoint \
  --source /path/to/permanent-native-checkpoint \
  --output /path/to/NEW-fp16-scale-checkpoint \
  --kernel-bundle /path/to/matching-fp16-scale-bundle \
  --fp16-storage
```

An existing output is rejected. Incomplete exports retain an incomplete marker
and are rejected by the loader. `PrefillDecodeNative` preserves and validates the
same scale ABI across both schedules. Do not submit FP16 banks to an older FP32
scale kernel. Language quality and full serving performance remain separate from
arithmetic and replay qualification.

## Running service and replay

The service is on port **8001**, served name `glm53-flash-selective-w3`.
The logfile is
`/srv/ai/artifacts/glm-prefill-weight-reuse-20261007/fp16-scales-server-20261008.log`.
The saved process JSON records the full launch command. Configuration retains
TP4, MTP1, 640-token chunks, four request slots, prefix caching and configured
max model length 311,040. This run tested at most 6,400 prompt tokens; the
configured maximum is not a validated full-length capacity result.
Full decode graphs at sizes 2/8 and resident 640-token prefill graphs are active.
The prefill receipt records **46 graph segments and 45 eager boundaries** around
live attention/indexer work, with 11 replays in the final benchmark. This is
piecewise capture; the complete prefill pipeline is not one uninterrupted graph.
EP/flashcomm1 were not changed or separately qualified; image/video inputs remain
disabled. Real checkpoint requests and replay were tested; dummy weights were
not used. The requested four-slot configuration was retained. The adapter skill's
16-slot capacity baseline was not exercised in this run.

The `.py.txt` files archive the exact launch, hot-swap and benchmark scripts.
To reproduce the mixed dispatch after startup, execute the saved mixed-live script
first (loads the qualified v965 manifest), then the final v967-live script (loads
its qualified converter and recaptures). Scripts use fresh generation IDs and
validate all four rank receipts; kernel admission requires the matching gate
JSON files. Private live resources and source paths are recorded in those files.
Hot swaps require a pause/drain with prefix invalidation; the successful final
transition resumes serving. The FP16 checkpoint performs no online weight
repacking. Worker PIDs and resident storage digests remain fixed across swaps.

A separate 512-token-budget serving smoke completed normally (`finish_reason=stop`)
and returned a nonempty final answer. Its response and all four worker receipts
are archived in `fp16-scales-final-serving-smoke-20261008.json`; this checks
serving completion, not model quality. Final receipt confirms the server resumed.
