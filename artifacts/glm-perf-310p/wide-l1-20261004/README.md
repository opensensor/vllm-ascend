# GLM packed-expert prefill: on-chip and route-combine trials

Latest: The next prefill-only serving trial was **not promoted**. The
command-ownership experiment retained completed queue tensors; that bug was
fixed and passed 54 operator tests, but full-model trials still encountered
cache admission, real-prefill OOM, and a later decode stall. The ownership
adapter remains compile-gated and off by default. All four NPUs are released;
port 8001 is stopped at the user's request.

The stronger next candidate is mHC post-mixing: on FP16-rounded state inputs,
its isolated 1280-token operator changed **11.36 → 8.38 ms**, with allocated
peak delta **560.6 → 180.0 MiB**. It is not bitwise identical and has not been
integrated or qualified in serving. See
[the complete test record](expert-grouping.md#prefill-only-serving-admission-and-command-retention).

Status: The expert-grouping serving trial finished and all four workers were
stopped at the user's request to release the NPUs. The candidate used 1280-token
prefill, the enlarged grouped-route cap, and the W3 wide-L1 package. The
7269-token cold retrieval passed in 70.24 s TTFT versus 77.69 s for the
640-token overlap control (9.6% shorter). Preserving 192K configured context
required the full profiled cache budget; minimum sampled free device memory
was only 435 MiB. Full-length context remains untested. Decode measured
4.05 tok/s c1 and 7.96 tok/s aggregate c4; c4 regressed from the control's
10.85 tok/s, so the candidate was not promoted. The next offline batch stages
prefill-only W3 dispatch and a native FP32 route-combine kernel. Details,
failed admission attempts, and operator results are in
[expert-grouping.md](expert-grouping.md). Compile options remain opt-in;
no commit or default-build promotion was made.

## Wide L1 baseline

The compile-gated `GLM_W2_GROUPED_L1_WIDE` path decodes W2/W4 NZ-packed
128-channel by 1024-K weight tiles to L1, then accumulates 256-K Cube stages in
L0C. It eliminates the decoded FP16 GM workspace for those calls. Both the
original 128-K stage and the 256-K revision passed bitwise operator parity but
were slower on realistic grouped shapes. At 256-K, W4 gate/up changed from
45.59 to 51.87 ms and W2 down from 28.13 to 30.90 ms. The matched 7,269-token
retrieval answered exactly with 81.23 s TTFT versus 79.40 s on the ordinary
packed-W3 OPP. See `operator-128-compare.json`, `operator-256-compare.json`,
and `serving-8k-candidate.jsonl`.

## Grouped route combine

The opt-in `ascend_glm_fused_route_combine` path uses the 310P
`npu_moe_token_unpermute` primitive to combine routed FP16 outputs on device.
`ascend_glm_empty_peer_rows` additionally lets the grouped C++ adapter leave
peer-owned zero-weight output rows uninitialized; the fused combine ignores
them. Six direct W2/W3/W4 NPU cases passed with initialized and uninitialized
outputs. The focused CPU suite passed 25/25.

| 7,269-token cold retrieval | TTFT | Answer |
| --- | ---: | --- |
| Ordinary packed W3 | 79.40 s | Exact |
| Fused route combine | 78.35 s | Exact |
| Fused combine and empty peer rows | 78.05 s | Exact |

The fused-only strict quality run scored 16/20. The earlier ordinary W3 run
scored 17/20; its additional miss was `code_range` wrapped in backticks. The
small TTFT difference is within single-run noise, so these options remain off
by default. Records are `serving-8k-fused-combine.jsonl`,
`serving-8k-empty-peer.jsonl`, and `quality-fused-combine.jsonl`.

## Current FP16 prefill profile

A fresh 2,048-target retrieval with a distinct code forced a cold prompt and
captured two 640-token iterations. It returned the exact code with 1,815
served prompt tokens. The four-rank trace is at
`/home/matteius/experiments/glm-w3-20261004/profile-current-fp16-640-20261004`
on the NPU host; `profile-current-fp16-summary.json` is the parsed summary.
This run used the current FP16 mHC, KDA NZ projection, Cube512 QSA, and default
packed-W3 grouped OPP. It did not enable the route-combine options.

| Rank-0 task category, two chunks | Current FP16 | Earlier BF16 |
| --- | ---: | ---: |
| Grouped packed W2/W3/W4 | 5.349 s | 5.315 s |
| KDA and convolution | 3.449 s | 3.456 s |
| Transfer and layout | 0.721 s | 2.889 s |
| Cube512 QSA | 0.694 s | 0.693 s |
| Four-rank task envelope | 13.430 s | 15.688 s |

The older AI CPU BF16 casts were already addressed by the FP16 mHC mode; they
are not the current leading target. The grouped kernel still spends a median
50.6% of its measured core time in vector work and 38.5% in MTE2, with 2.4%
in MAC. Category task times are attribution and may overlap. The current
remaining large costs are grouped packed projection and KDA, not route output
initialization or host-side streaming.

## Pipelined wide-L1 candidate

`GLM_W2_GROUPED_L1_WIDE_PIPELINED` is a second opt-in flag layered on
`GLM_W2_GROUPED_L1_WIDE`. It uses two 128-K L1 activation and L0A/L0B stages
while retaining the 128 by 1024 decoded B tile in L1. The intent is to let
MTE2 prepare the next A stage while Cube consumes the previous stage. This is
still experimental. CANN compiled both grouped kernel variants successfully.
The build-only OPP overlay is
`/srv/ai/src/glm-l1-wide-build-20261004/opp-l1-wide-pipelined-20261004`;
it contains the new grouped kernel binaries and JSON metadata over the
previous wide-L1 package. The 21ed kernel binary has SHA256
`9299ebd990dd08e077b9e17193f79236235761f3bcb2c3970528b39a739e9c7a`.
The subsequent matched 5,120-route test passed bitwise W2/W4 parity, but W4
gate/up took 48.65 ms versus 45.70 ms baseline, and W2 down took 29.25 ms
versus 28.16 ms. This failed the operator performance gate, so no serving run
was warranted. See `operator-pipelined-compare.json`. This candidate applies
to NZ-packed W2/W4; W3 still uses the existing GM staging path.

## Byte lookup follow-up: rejected

A small signed-FP16 byte lookup table replaced the RINT field extraction in
the pipelined wide-L1 path, reusing existing NZ scratch space. Four hardware
tests passed, including exhaustive byte values at K-stage boundaries and the
packed-buffer reuse regression; the two realistic projections were bitwise
equal to baseline. However, W4 gate/up regressed from 45.63 to 80.25 ms and W2
down from 28.10 to 39.72 ms. See `operator-byte-lut-compare.json`.

The lookup implementation was removed from main. Its tested source is saved
on the NPU host as
`/home/matteius/experiments/glm-w3-20261004/wide-l1-20261004/opp-l1-byte-lut-20261004/tested-kernel.h`.
The exhaustive NZ byte regression remains in
`tests/e2e/nightly/310p/single_node/ops/test_w2_grouped_nz_bytes_310.py`.
The build audit confirmed both packages retained the promoted scale-pair and
RINT compile options from the grouped operator's CMake configuration.

## Overlap across K tiles

The revised pipelined schedule keeps stage events alive across K tiles,
allowing the next tile's dequantization to overlap the preceding Cube work.
Only L1 readers must finish before B is overwritten; the Cube consumes its
own L0 buffers. The wide path also uses the existing post-Cast packed-buffer
handoff instead of fencing all preceding vector work before each byte load.

Four exhaustive-byte/buffer-reuse hardware tests passed, as did 37 selected
grouped operator regressions. The unrelated 768-token route-admission case
was excluded because this experiment preserves the prior OPP host wrapper.
The two realistic 5,120-route projections passed bitwise comparison:

| Projection | Baseline | Overlap | Time reduction |
| --- | ---: | ---: | ---: |
| W4 gate/up | 45.59 ms | 38.64 ms | 15.2% |
| W2 down | 28.05 ms | 24.79 ms | 11.6% |

Records are in `operator-overlap-compare.json`. The isolated overlay package
is `/srv/ai/src/glm-l1-wide-build-20261004/opp-l1-overlap-20261004`, and its
NZ grouped object SHA256 is
`6eecc8af91a6031399234a77af66f4985f40793b7e2146be67d5520496693f63`.
`run-serving-variant.sh` selects this OPP or the baseline with identical
checkpoint, graph, FP16 mHC, Cube512 QSA, 640-token chunks, prefix caching,
192K context, and route-histogram settings.

The matched cold retrieval served 7,269 computed tokens with zero cache hits
on both launches and returned the exact code. Client TTFT was 79.91 s on the
fresh baseline and 77.61 s on the overlap candidate, a 2.9% reduction in one
pair. This remains a small prefill difference. Records are
`serving-8k-baseline-matched.jsonl` and `serving-8k-overlap.jsonl`.

The candidate-first/baseline-second comparison then ran the same strict
20-case suite and five 256-token short responses on each server:

| Serving metric | Baseline | Overlap | Change |
| --- | ---: | ---: | ---: |
| c1 decode | 3.694 tok/s | 3.989 tok/s | +8.0% |
| c4 aggregate decode | 8.926 tok/s | 10.952 tok/s | +22.7% |
| Strict quality | 17/20 | 17/20 | Same three failed case IDs |
| Cold 7,269-token TTFT | 79.91 s | 77.61 s | 2.9% shorter |

All five short responses in each run reached 256 tokens. The shared failures
were `instr_reverse`, `instr_first`, and `code_slice`; the reverse answer's
text differed, so full-model output identity is not claimed. All 748 runtime
Python/extension file hashes matched between launches. The tiling sources
matched, although their separately built libraries differed in binary hash.

This is one matched serving pair, not a repeated randomized benchmark. The
main measured benefit is decode; cold prefill remains slow. The configured
context is still 196,608 tokens, and a full-length prompt has not been tested.
Records are `serving-{baseline,overlap}-quality-short.jsonl` and their summary
files; `serving-matched-summary.json` collects the comparison.

During the short tests, sampled peak device temperature was 71 C for the
candidate and 74 C for baseline. Device medians were approximately 2.5–3.5 C
lower for the candidate. The runs differed in duration and initial thermal
state, so these observations do not establish steady-state cooling or power
efficiency. Raw samples are `temperature-{baseline,overlap}-short.jsonl`.

The stopped server used `run-serving-variant.sh overlap validated`. Its log
is `/home/matteius/experiments/glm-w3-20261004/server-l1-overlap-validated-20261004.log`
and its PID file has the same basename with `.pid` instead of `.log`.

## Next resident-weight candidates: hardware testing deferred

The subsection below records offline staging before NPU access was restored.
Subsequent hardware results and the enlarged-batch experiment are recorded
in [expert-grouping.md](expert-grouping.md).

The following independent compile options extend `GLM_W2_GROUPED_L1_WIDE`
and its pipelined schedule:

- `GLM_W2_GROUPED_L1_W3` admits NZ-packed W3 to the 128-N by 1024-K
  resident path for expert groups of at most 128 rows. Decoded weights travel
  from UB to L1 to L0B, removing their GM write/read round trip. Canonical
  packing and unsupported K shapes retain their existing path.
- `GLM_W2_GROUPED_L1_SCALE_CACHE` loads all scale rows for one wide N tile
  once, keeping them in UB across K slices. Buffer addresses depend on layout
  and geometry, not an expert's row count, so reusable decode tables stay at
  the same addresses when successive experts select different schedules.
  Canonical W3 retains its original one-row scale allocation: reserving four
  rows for both layouts exceeded the compiler's UB bound. NZ W3 has separate
  scratch accounting and fits its enlarged cache in the 192 KiB UB.
- `GLM_W2_GROUPED_L1_LARGE_GROUPS` uses the existing 32-N by full-K resident
  schedule for larger groups with K at most 4096. It decodes one weight tile,
  keeps it in L1, and applies it to every 128-row M tile before advancing N.
  Thus those larger groups also avoid decoded-weight GM staging. This trades
  the wider Cube N dimension for weight reuse across all rows; its speed is
  not established.

These changes do not alter model routing, quantization, or checkpoint bytes.
The host still reserves fallback workspace. Default builds and the saved
overlap OPP remain unchanged. `build-resident-candidate.sh` compiles W3-only,
scale-only, and combined packages from the staged main sources, retaining
source, flag, and object hashes. It neither launches a server nor executes
hardware tests.

CANN compilation passed for all three packages, stored under
`/srv/ai/src/glm-l1-wide-build-20261004/`:

| Package | Options added to pipelined wide L1 | NZ object SHA256 prefix |
| --- | --- | --- |
| `opp-l1-w3-v2-20261004` | W3 | `435f6c41dc38cee92` |
| `opp-l1-scales-v2-20261004` | Scale cache | `352665e1711d0ba7` |
| `opp-l1-all-resident-v2-20261004` | W3, scale cache, large groups | `a0ae0d5e8a92d0bc` |

Full source, flag, metadata, and binary hashes are in
`build-{w3,scales,all-resident}-v2-hashes.json`. All packages used the same
staged functional sources; the subsequent main-source comment clarification
does not change their behavior. The five CPU tests in
`test_w3_cube_layout.py` and `test_w2_grouped_tiling.py` passed with
`--noconftest`. Targeted Ruff checks, shell syntax, and `git diff --check`
passed. These checks establish buildability and host-side layout/routing
contracts, not device correctness or performance.

The extended hardware tests cover W3 cross-byte fields at K-tile boundaries,
all three bit widths, empty experts and peer rows, and 128/129/257-row groups
that exercise resident/fallback transitions and multiple M tiles. The W3
benchmark now includes 18-row-per-expert prefill, a mixed 129-row group, and a
640-row concentrated group. Its eight-expert cases isolate projection costs;
they do not represent complete 72-local-expert serving latency.

The attempted boundary test stopped in repository `conftest.py` because
`modelscope` is unavailable, before running any device case. When hardware is
authorized again, use `pytest --noconftest` for these self-contained operator
tests. No hardware parity or speed result is claimed for these new options.

### Group each expert's tokens once

The current implementation already computes top-8 routes for the available
token batch, sorts routes by expert, and passes cumulative expert boundaries
to a grouped operator. That operator processes all rows for an expert before
advancing to the next expert. The existing GM path also decodes a weight tile
once and reuses it across its M tiles; the new large-group path preserves
that reuse on chip.

The remaining repetition is across scheduler chunks and grouped-call
boundaries. At 640 prompt tokens, there are 5120 route assignments per MoE
layer. Current main supports at most 6144 assignments per call (768 top-8
tokens); the retained serving OPP used the earlier limit. Simply increasing
the scheduler batch beyond the grouped cap still splits expert work into
multiple calls. A larger-batch experiment must coordinate this limit, route
buffer memory, KDA temporary/state memory, and attention workspace. Expert
choices can be computed for all available tokens at one layer after the
preceding computation; they cannot be precomputed for every layer from token
IDs alone. No routing approximation or whole-prompt allocation was introduced.
