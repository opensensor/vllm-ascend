# GLM prefill: four-query tiled native scoring

## Status

Prepared while the user tests the public GLM server. **No server restart,
resident patch, NPU initialization, or inference request was performed.**
The CANN compiler and C++ linker ran on CPU at lowered priority. The new
operator has a separate name and isolated package; no serving package was
overwritten. This is an experimental candidate, not a measured prefill win.

## Work removed

Current prefill gathers pooled keys, widens them to FP32, constructs
`[queries,32,pools]` scores, applies ReLU and head weights, and reduces heads.
The recent in-place weighting improvement already removed a second allocation,
but the full head-sized score tensor is still written and read in device RAM.

`GlmKpoolPrefillScoreV310` instead:

1. Loads four query rows, each with 32 heads, into a 128-row Cube operand.
2. Gathers one 64-key tile directly from paged shared storage.
3. Performs one 128x64x128 Cube multiplication with FP32 accumulation.
4. Applies ReLU, FP32 head weights, and a separate reduction for each query
   in local storage. Only `[queries,pools]` scores reach device RAM.
5. Masks each query's unfinished pools, padded rows, and invalid pages to
   negative infinity before output.

Four queries share a key tile, unlike the existing one-query decode kernel.
This does **not** imply four times less key traffic than the current large
prefill GEMM, which has different reuse. Native tiling could lose enough GEMM
efficiency or incur enough barriers to outweigh its memory savings. Measure
the entire selector before any serving trial.

For 640 queries and 2048 pooled keys, a head-sized score tensor contains
160 MiB versus 5 MiB of reduced scores. These are tensor sizes, not measured
memory bandwidth or a predicted 32x speedup. Larger contexts retain the
baseline query subchunk boundary, bounding reduced-score workspace to about
8 MiB per selection chunk, plus native subcall outputs/concatenation.

## Integration contracts

- `tools/glm_perf/kpool_prefill.py` aliases shared cache backing and passes its
  offset and strides. It never makes the full cache contiguous.
- Native calls accept 4..128 rows in multiples of four and one request table.
  The wrapper pads partial rows with position `-1`, then removes row/column
  padding before selection. Unsupported cache geometry retains baseline code.
- Each request is handled separately; empty requests and mixed dense/sparse
  requests retain their output layout. Host scheduler metadata supplies lengths.
- Top-k receives the baseline row count and **original, unpadded column count**.
  Its existing rank/index checks remain. No device length read is introduced.
- Query rotation retains BF16 rounding before FP16 Cube conversion, as in the
  qualified decode scorer. FP16 range/subnormal behavior and changed FP32
  accumulation require numerical and serving gates; bitwise parity is not claimed.
- The candidate patches only `_select_tokens`, not `_select_tokens_fixed`.
  No decode graph selector, expert layout, chunk budget, or model limit changes.
- The public server has resident development endpoints disabled. The candidate
  is prepared for a future instrumented launch; it has not been applied live.

## Completed offline checks

- **21 CPU tests passed** in `tests/ut/glm_perf/test_kpool_prefill.py`:
  alias/stride/offset preservation, partial tiles, baseline top-k geometry,
  exact ties, multiple requests, empty requests, dense fallback, output padding,
  malformed metadata, and execution of the native host geometry predicate.
  The native math is mocked with a CPU reference; these are not kernel parity.
- Targeted Ruff and whitespace checks passed.
- CANN 9.1.0 host and Ascend 310P device object compiled and packaged successfully.
- Supplemental PyTorch binding compiled and linked without loading it.
  SHA256: `336a784e315221ae60ab9dfd94c25a37047c750ef498ed4973deead56abaf5c8`.
- Compiler logs and the exact binding `build.ninja` are alongside this document.

Package:
`/srv/ai/src/glm-l1-wide-build-20261004/opp-kpool-prefill-tiled-20261005`

Binding:
`/home/matteius/experiments/glm-kpool-prefill-tiled-20261005/glm_kpool_prefill_score_candidate.so`

## Next hardware window

`test-kernel.sh` is staged, **not executed**. It runs 13 prepared hardware cases,
then an alternating-order benchmark of the complete selector at 640 query
rows and 8K/32K/128K/311K live context. Both sides include rotation, scoring,
and selection; the baseline also includes its page gather. It records selected
set/order parity, per-sample elapsed time, and peak allocated-memory deltas.

Gate native parity at tile/page/16000-token boundaries, invalid pages, poison
padding, 128-row calls, partial rows, exact ties, and the full 640-query wrapper.
Then compare matched cold 8K and 20K TTFT in the full model, c1/c4 decode,
quality, peak memory, and worker/weight identity. Keep all other settings fixed.
If the selector fails to improve, do not promote it based on reduced allocations.

Larger actual expert batches and the resident-W3 pipeline remain separate
experiments. This scorer candidate does not eliminate expert weight expansion.
