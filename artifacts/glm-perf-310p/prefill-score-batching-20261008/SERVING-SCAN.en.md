# GLM serving-path scan

This audit checks the bound 310P serving path, rather than counting Python spellings.
The live four-rank status still matches the frozen `kda_reuse_v2` source, with no
expert reconstruction fallbacks and native selected-state gather/scatter enabled.

| Source site | Current relevance |
| --- | --- |
| `dsa.py:427` `.item()` and DSA reference matmuls | Reference attention; serving retains native paged MLA and kpool indexing. |
| `kda.py` Python recurrence/casts | CPU parity core; NPU forward uses `kda_310.py` and custom operators. |
| `model.py:301/305/392/441/462/613` packing, CPU copies and stacking | Load/preparation work; permanent checkpoint avoids online repacking. |
| `model.py:908` conv transpose/cast | Cached initialization, not per-token transformation. |
| `model.py:1162/1163` FP32 indexer weights | Prepared after loading, outside graph capture. |
| `kda_310.py:148/149` L2norm followed by FP16 conversion | Active, but L2norm already returns FP16; following same-dtype conversion is a no-op. Strided inputs may still need layout copies. |
| `kda_310.py:152/211/212` gate/beta conversions and sigmoid | Active producer work; fusion must preserve existing intermediate precision/rounding. |
| `kda_310.py:223/224` constant FP32 operands | A_log and dt_bias are declared FP32. Float calls do not imply a conversion each forward. |
| `kda_310.py:205/229` carry access | Live hook uses v925 selected rows, not the generic whole-bank indexing path. FP16/FP32 carry boundaries remain. |
| `model.py:794/802` output RMSNorm gate | Active fusion candidate; preserve FP32 activation and final FP16 rounding. |
| `model.py:177` MoE output cast | FP32 combined/reduced result returns to the shipped FP16 activation contract. |
| `moe.py` hc_pre/hc_post helpers | Reference hyper-connection functions; actual model uses shipped MHC operators and the patched mHC path with qualified normalization. |

`.t()`/`split()` commonly create views. `.reshape()` can copy for incompatible
strides; `.contiguous()` copies only when necessary. Same-dtype `.to()` and
`.float()` are not dispatched casts. Graph capture retains real casts/copies as
replayed device work. Counts alone cannot rank the costs.

The last 640-token profiled step includes 1,376 InplaceCopy/Cast-family tasks
summing 21.89 ms and 266 Matmul/Cast tasks summing 34.88 ms, across mixed types
and sites. This is not attribution to each grep line or a wall critical path.
Larger task sums remain experts, sparse attention and KDA scores. The next
candidate batches KDA row vector operations inside the complete prefill kernel.

Two gate projections consume distinct f_a and g_a inputs. A fused/grouped
schedule could reduce launch/layout overhead; naive block-diagonal concatenation
would add zero-padded arithmetic. The cached projection group-list table is
bounded by scheduler shapes in the current 640-token chunk configuration, rather
than growing once per generated token.
