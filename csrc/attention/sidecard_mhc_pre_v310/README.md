# Sidecard mHC candidate integration

These kernels are offline implementations for T6 of the sidecard execution
plan. They have not been compiled or timed on an Ascend device. Keep both
candidates unbound until T11 builds them and T12 validates each enabled row
count. Portable tests validate references, contracts and fallback policy.

## Build and binding

Build `SidecardMhcPreV310` and `SidecardMhcPostV310` with the existing 310P
custom OPP build, explicitly setting the CMake cache options
`-DSIDECARD_BUILD_MHC_PRE_V310=ON` and
`-DSIDECARD_BUILD_MHC_POST_V310=ON`. Both default to `OFF`, including for
`ASCEND_OP_NAME=all`, and can be enabled independently. These are build
options rather than environment variables. Each package has host definitions, shape inference, tiling,
kernel entry point and local CMake files. The post package lives at
`csrc/gmm/sidecard_mhc_post_v310/`. Shared build lists and torch bindings are
intentionally unchanged; T11 owns those integration points.
The post `op_host/CMakeLists.txt` also guards direct GMM discovery and uses
a global include guard to avoid registering the same enabled candidate twice.

Include `sidecard_mhc_pre_310_torch_adpt.h` and the post package's
`sidecard_mhc_post_310_torch_adpt.h` after the existing NPU guard and execution
macros. Register in `_C_ascend` with PrivateUse1 implementations:

```text
sidecard_mhc_pre_310(Tensor residual, Tensor fn, Tensor scale, Tensor base,
    Tensor norm_weight, float norm_eps, float hc_eps) -> (Tensor, Tensor, Tensor)
sidecard_mhc_post_310(Tensor x, Tensor residual, Tensor post, Tensor comb) -> Tensor
```

The pre outputs are BF16-rounded FP32 post `[R,4,1]`, unrounded FP32 Sinkhorn
logits `[R,4,4]`, and normalized FP16 hidden `[R,4096]`. Use the existing
`mhc_sinkhorn_310` on the logits, then round its result through BF16 back to
FP32. The post output is BF16-rounded FP32 residual `[R,4,4096]`. It indexes
`comb[input_stream, output_stream]` and retains the complete FP32 residual
range. Neither package changes the legacy FP16-saturating post operator.

Both tilers accept 1 through 32 rows with width 4096 and four streams. No GM
workspace is needed. Pre uses one row per core with 118,816 bytes of local
buffers; confirm that the selected toolkit/device provides that UB capacity
before building. Post divides each output stream into 512-element tiles and
uses 9,216 bytes per core. All large copies have aligned lengths.
Tilers prefer the platform's AIV count and fall back to its AIC count for
older 310P metadata, rejecting zero vector-capable cores or insufficient UB.

Pre dot products use FP32, ascending 1024-element tiles, fixed pairwise
halving inside a tile, and a 64-lane vector reduction. Stream combinations
use explicit stream order 0, 1, 2, 3. Post performs separate FP32 multiply
and add operations, then software round-to-nearest-even BF16. No FP16 cast
occurs between residual load and BF16 rounding. Only normalized hidden and
the attention/MLP input are FP16.

## Python policy and device gates

`tools.sidecard.npu_mhc.prepare_weights` prepares contiguous FP32 mixer
weights once per owner construction. Each FP32 mixer matrix occupies
1,572,864 bytes; there are two per layer, approximately 3 MiB total.
`TextLayer.prepare` already converts these weights to FP32, so existing
contiguous FP32 tensors are reused without allocating another copy. This
is required residency relative to the FP16 checkpoint, not necessarily
incremental memory over the current baseline. Count actual allocations once
in memory and 128K admission checks; avoid conversions in the token loop.

`NativeMHC` defaults to empty qualified row sets and falls back to the
caller's baseline. Unsupported row counts, devices or residual geometry
also use that fallback. Native device failures propagate without unsafe
automatic retries. Qualified pre and post row sets are independent.

The probe accepts the existing `checkpoint`, `checkpoint_sha256` and
`ops_library` config keys, plus optional `mhc_layers` (default `[18,19]`). It
requires the parent orchestrator's exclusive hardware lease and existing
physical device assignment `ASCEND_RT_VISIBLE_DEVICES=2,3`:

```bash
python -m tools.sidecard.mhc_probe --config SPEC.json --output REPORT.json --rank 0
python -m tools.sidecard.mhc_probe --config SPEC.json --output REPORT-rank1.json --rank 1
```

The fixed contract requires all rounded fields to match qualified baseline
bytes. Raw pre logits use `atol=2e-4`, `rtol=2e-5`; this tolerance does not
relax the post, combination or normalized hidden checks. Finite synthetic
inputs must produce finite outputs. Tests cover both mixers, positive and
negative residuals above 65,504, changed coefficients, arbitrary signed
combination matrices and 1/4/32 rows. Probe timings have five warmups and
25 synchronized repetitions, excluding diagnostic transfers.

Changed reduction trees may fail the byte gates even when their raw FP32
error is small. Such shapes remain disabled. A passing and faster isolated
probe only marks probe eligibility; full-model boundaries, tool quality,
graph replay, matched API timing and 128K memory gates still remain.
