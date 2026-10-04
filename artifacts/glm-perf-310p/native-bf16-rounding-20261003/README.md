# GLM 310P graph prefill: native BF16 mHC rounding

Status: experimental, not Gate-A qualified. All measurements below used four
Ascend 310P devices, the same W4through32-noclip checkpoint, TP4, 512-token
prefill chunks, device-resident MLA, and `FULL_DECODE_ONLY` c1 graph capture.
The only model-code difference in the matched prefill pair was replacing the
integer BF16 rounding emulation in `patch_mhc_norm.py` with FP32→BF16→FP32
conversion. The prior graph-safe kpool scatter was present in both runs.

## Why this target

The baseline 517-token profiler request contained a 512-token chunk and a
5-token tail. Rank 0 attributed 3,764.4 ms of device-task time to 1,436
`BitwiseAnd` AI-CPU calls and 338.2 ms to 718 `RightShift` AI-CPU calls.
Input shapes matched mHC state tensors such as `[512,4,4096]` and
`[512,4096]`; the source used integer shifts and masks to round FP32 state
to BF16 precision. These are task-time sums, not a critical-path estimate.

## Isolated 310P check

`tools/glm_perf/micro_mhc_bf16_round_310p.py` compared the old bitwise expression with
native BF16 cast on 1,000,000 randomized finite FP32 bit patterns, tie cases,
large magnitudes, and graph replay after changing the captured input. All
finite comparisons and replay outputs matched bitwise. Median synchronized
eager times over 12 repeats were:

| FP32 shape | Integer emulation | BF16 cast |
| --- | ---: | ---: |
| `[512,4,1]` | 0.3525 ms | 0.3303 ms |
| `[512,4,4]` | 0.4626 ms | 0.3988 ms |
| `[512,4096]` | 10.8368 ms | 1.9507 ms |
| `[512,4,4096]` | 38.2167 ms | 6.7656 ms |

The direct cast preserves BF16's exponent range; the existing FP16 opt-in
continues to be a separate, lower-precision experiment. No claim is made for
NaN payload bit identity; model activations must remain finite.

## Matched serving check

The full 512-token profiler iteration fell from **20,689.45 ms** to
**17,426.62 ms** (15.8% lower). The 517-token client call fell from 23.62 s
to 20.31 s, but includes request overhead and a short second chunk. In the
new rank-0 profiler export, `BitwiseAnd` and `RightShift` AI-CPU work are
absent from the leading operation totals. Grouped W2 task time was effectively
unchanged (13,729.3 ms before, 13,721.8 ms after across both chunks), so this
change does not solve the principal prefill bottleneck.

The c1 24-token streaming probe after the change measured 23 decode gaps in
22.696 s, or **1.01 tok/s**; no meaningful decode improvement is claimed.

The strict 20-case graph quality suite remained **17/20**, with the same
`instr_reverse`, `instr_first`, and `code_slice` failures as before. Those
cases were also reproduced on the eager path before this change. This is not
a Gate-A pass and should not be used to claim production quality.
An additional 2,085-token retrieval prompt returned the exact code
`BLUE-ORCHID-7319` in 11 completion tokens (90.91 s end to end).

Evidence on the Threadripper host:

- Baseline trace: `/home/matteius/experiments/glm-gate-a-20261002/profile-scatter-prefill512`
- BF16 trace: `/home/matteius/experiments/glm-gate-a-20261002/profile-nativebf16-prefill512`
- Baseline and BF16 server logs: `server-rowtile-graph-prefill512-profile.log`
  and `server-rowtile-graph-nativebf16-prefill512-profile.log` in the same
  experiment directory.
- Local quality records: `/tmp/glm-graph-scatter-quality-20261002.jsonl` and
  `/tmp/glm-nativebf16-quality-20261003.jsonl`.

The older `W2_CUBE_MAX_TOKENS=128` cap does **not** govern this serving
profile's resident grouped-expert path. That path is selected before the
per-expert Cube fallback and accepts up to 5,120 routed rows; a 512-token,
top-8 chunk has 4,096 routes. Raising the 128 cap alone would therefore not
address the measured prefill latency.
