# QSA gather staging: next isolated experiment

Status: **hardware-qualified; eight-block K gathering is active**. This file
preserves the original experiment design. The implementation, exact byte and
whole-attention gates, launch-overhead diagnosis, and wrapper correction are
recorded in [the results](qsa-gather-staging-results.md). Both K/V orientations
passed private correctness gates; serving retains the installed V path because
its new wrapper still offsets the isolated kernel benefit.

The builder freezes the exact headers listed below, emits independent
four/eight-block entries, and uses an owned bridge with producer queue draining.
No coherent OPP replacement is involved. See
[combined serving measurements](qsa-prefill-decode-results.md).

## Exact runtime and trace

Remote artifact root:
`/srv/ai/storage/artifacts/npu-collective-qfn-20261014-024609`.
The qualified runtime is its `qfn-pinned-position-runtime` subdirectory.
Read-only source access used `ssh -F /dev/null -o BatchMode=yes
matteius@192.168.53.187`; no device or process inspection was needed.

Latest profiled rank-0 CSV:
`qsa-resident-v1/native-timeline-w4-queue-v3/cold-prefill/rank0/cold-prefill_rank0_579046_20261014082326632_ascend_pt/ASCEND_PROFILER_OUTPUT/kernel_details.csv`.
The local summary is
[`rank0-allocation-summary.json`](qsa-resident-v1/native-timeline-w4-queue-v3/rank0-allocation-summary.json).
This bounded trace contains 6,852 prompt tokens. Operator durations are summed
exposure, not guaranteed critical-path savings; K/V streams can overlap.

`QsaGatherValueNzV310` names **both** gather orientations. Its complete bucket
is 2,070 calls / 1,363.664 ms. Full 64-query tiles account for:

| Orientation | Calls | Summed duration | Logical output shape |
| --- | ---: | ---: | --- |
| Transposed key | 975 | 680.165 ms | `[64,1,256,2064]` |
| Value | 975 | 675.851 ms | `[64,1,2064,256]` |

Both read FP16 cache `[16800,16,64,16]`, INT32 group indices `[64,512]`,
three INT32 per-query vectors `[64]`, and INT32 block table `[1,4096]`.
The 16-token-aligned selected extent is `ceil((512*4+4)/16)*16 = 2064`.
Outputs use NPU format 29 (NZ), not a contiguous ND interpretation of their
logical dimensions. The remaining calls have query counts 1, 3 or 4.

## Pinned sources and existing behavior

Paths below are relative to the qualified runtime:

| Source | SHA256 |
| --- | --- |
| `csrc/attention/qsa_gather_value_nz_v310/op_kernel/qsa_gather_value_nz_v310.h` | `dbceeef186d29656390a644061f781ecb682ad4ae5a9d83f0c4969db169671a1` |
| `vllm_ascend/models/qwen4_exp/ops/qsa_batched_attention_310.py` | `eae684c0ed7efe1d9c4c8b388cc8160d0efd0b1616bb37eed42304ead63b9661` |
| `vllm_ascend/models/qwen4_exp/ops/qsa_gather_nz_310.py` | `c67521303b3fc3b39dced1ef21ffe811e3f82aa76c3a586e9ee6b896d2ef3b98` |

The current repository's native header has the **same hash** as the pinned
header. The Python files are newer locally; do not overlay them wholesale.

Pinned `qsa_batched_attention_310.py`:

- Lines 17, 109–111: query tile 64, selected-token extent and NZ padding.
- Lines 155–174: reusable selected K/V buffers and optional stream ownership.
- Lines 204–228: split request portions with their own page-table rows.
- Lines 237–277: separate K/V streams, producer events and buffer reuse
  protected behind the previous PV consumer.
- Lines 329–346: original QK, masking, softmax and PV; preserve these exactly.

Pinned `qsa_gather_nz_310.py` lines 74–84 convert metadata to contiguous INT32
and invoke the same operator with a transpose flag for K versus V.

Native header:

- Lines 18–23: `TOKEN_BLOCKS_PER_TILE = 4`, with 256 elements per NZ block.
- Lines 46–47: local page-table staging requires aligned capacity <=2048.
  The current 4096-entry table therefore uses scalar global page lookups.
- Lines 55–65: tile, transposed-tile, group-index and optional table buffers.
- Lines 91–98: exact cache offset is
  `((physical_block*cache_channels + kv_head*head_dim_blocks + dim_block)*64
    - token%64)*16` FP16 elements.
- Lines 120–149: strided group/tail DMA and fallback reads for masked lanes.
- Lines 128–184: staged loop, full-pipeline barriers, K transposes and NZ stores.
- Lines 163–170: K stores one NZ token block at a time.
- Lines 172–181: V stores staged token blocks with one strided DMA.

## Proposed bounded change

Freeze the exact qualified header and change **only**
`TOKEN_BLOCKS_PER_TILE` from 4 to 8 in an isolated native candidate with its own
namespace and retained resources. Preserve the baseline alongside it. Do not
change page tables, cache layout, query tiling, route selection, masking,
softmax, accumulation or collective behavior.

At the measured shape there are 129 output token blocks. Staging eight instead
of four changes 33 loop iterations to 17 and the two full-pipeline barriers per
iteration from 66 to 34 per query. V output DMA calls similarly fall from 33
to 17. Input group DMA count, K transposes and K output DMA count remain
unchanged, so this is not a claim of halving total gather time.

For head dimension 256, each staging buffer grows from 32 KiB to 64 KiB.
K uses two buffers, so its tile storage rises from 64 KiB to 128 KiB; V rises
from 32 KiB to 64 KiB. Add group-index storage and, for eligible tables, local
page-table storage. Confirm compiler-reported UB requirements and hardware
limits before launching. Sixteen-block staging would double this again and is
outside this first proposal.

The current final partial tile remains one NZ token block. Strided DMA
parameters derive from `TOKEN_BLOCKS_PER_TILE`; verify all source/destination
strides and UINT16 limits after freezing the candidate. Increasing staging can
reduce scheduling flexibility or lose performance through larger buffers;
measure rather than assume a win.

## Qualification before a serving trial

1. Compare every output byte for baseline and candidate, including masked
   fallback values and padding. Attention-output equality alone can hide a
   gather defect in masked lanes.
2. Cover both K and V layouts, query counts 1/3/64, and counts
   0/1/15/16/17/255/256/511/512. Include mixed counts within one query tile.
3. Cover tails 0–3, page-boundary tails, randomized physical page maps,
   repeated selected groups and noncontiguous logical page ordering. Reuse
   output buffers across changing selections to expose unwritten stale lanes.
4. Exercise both full 4096-entry tables and aligned local tables <=2048;
   include two KV heads as a layout guard even though the profiled rank has one.
5. Compare complete QSA attention outputs byte-for-byte with unchanged QK,
   softmax and PV. Include multi-request query portions and reused buffers
   protected by the original stream/event ordering.
6. Alternate paired timing order using queued device events and wall timings.
   Report K and V separately and include dense and sparse selections. Preserve
   baseline and candidate source/binary hashes, then assess matched cold
   serving requests only if the component and whole-attention gates pass.

Use the controller's paused resident private-RPC workflow. Do not replace the
coherent OPP package, restart the model, mutate model KV caches for a probe, or
infer graph qualification from an eager private-tensor gate.

## Prior experiments and alternatives

The October 5
[gather follow-up](../qwen38-prefill-batch256-20261005/qsa-zero-invalid-20261005/README.md)
already tested compact tables: they passed byte equality but were consistently
slower. Larger query tiles 128/256 had no stable gain; 512/1024 regressed.
Do not repeat these as unexplored optimizations.

Zero-filling invalid groups improved sparse selections but regressed full
512-group K/V gathers by approximately 4–5%. The separately named
[selective gather](../qwen38-prefill-batch256-20261005/qsa-selective-gather-20261005/README.md)
passed its synthetic gates and improved first-chunk attention by 1.94% with
parallel gathers. It was not qualified for real multi-request/prefix serving
admission, and later dense chunks saw no gain. It changes masked gather bytes,
so it is distinct from this exact staging experiment.

Newer local Python code provides `prepare_qsa_gather_metadata`, tested in
`tests/ut/qwen38_1m/test_qsa_gather_metadata_reuse.py`, to prepare shared INT32
metadata once before K/V consumer events. The pinned source lacks it. In the
latest trace, `[64]` INT64→INT32 casts take 23.088 ms across 5,850 calls and
`[64,512]` casts take 13.970 ms across 2,925 calls. This is a smaller separate
candidate with stream-lifetime considerations; it does not explain the
1.364-second native gather exposure.
