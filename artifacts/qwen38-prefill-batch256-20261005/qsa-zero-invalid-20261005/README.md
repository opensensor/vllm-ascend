# Qwen QSA gather follow-up, isolated 310P probes

This round uses one 310P and the existing coherent Qwen operator package.
It does not start a full Qwen server. The saved
[`benchmark_qsa_table_slice_310.py`](../../../tools/qwen4exp/benchmark_qsa_table_slice_310.py)
compares the full 4,096-entry page table with an aligned visible-page clone,
reusing output buffers and alternating measurement order. The cache has
16,800 physical pages for production-sized geometry in the principal runs.

## Full versus compact page table

All K and V outputs were bitwise identical within each run. Median wall time
for one 64-query tile, 512 selected groups, 24 paired samples:

| Visible tokens | Active groups | Full K | Compact K | Full V | Compact V |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2,560 | 512 | 0.654 ms | 0.676 ms | 0.437 ms | 0.465 ms |
| 10,240 | 512 | 0.662 ms | 0.684 ms | 0.479 ms | 0.510 ms |
| 2,560 | 128 | 0.712 ms | 0.717 ms | 0.521 ms | 0.536 ms |

The narrower table activates the kernel's local page-map staging, but is
slower at all three shapes. A production-source slice was therefore removed.
The raw [2,560-token](visible-2560-cache16800.jsonl),
[10,240-token](visible-10240-cache16800.jsonl), and
[128-active-group](visible-2560-active128.jsonl) records retain all samples.
The 128-active-group probe is slower than the 512-active-group probe even
though it reads fewer distinct cache groups; the kernel still copies fallback
cache data for every inactive group.

## Query tile size

The existing QSA benchmark compared exact outputs for 2,560 queries and 512
selected groups, using a 262,144-token allocated cache and 10,240 visible
tokens. One sweep measured 64/128/256 query tiles at 206.5/205.0/202.8 ms,
but a second sweep measured 256 at 206.6 ms with a 205.9 ms same-run baseline.
Tiles 512 and 1,024 regressed to 213.8 and 214.9 ms. All compared outputs
were exact. The 256 result is not stable enough to change the serving default.
See [first sweep](query-tile-sweep-2560.jsonl) and
[wider sweep](query-tile-sweep-wide-2560.jsonl).

## Masked-group zero-fill candidate

The isolated [`qsa_gather_value_nz_v310.h`](qsa_gather_value_nz_v310.h)
replaces repeated fallback cache reads for fully masked groups with local
zero fill. Valid groups and tail handling retain the existing code. The source
was built in `/srv/ai/src/qsa-zero-invalid-20261005` with its own vendor
package, leaving the serving package intact.

The first package zeroed only one of 16 head-dimension blocks. Its
[quiet 128-group probe](zero-quiet-active128.jsonl) measured a masked-value
zero fraction of 0.0625 and K gather regressed from 0.710 to 2.297 ms. The
unchanged attention hash was insufficient to validate the gather output,
because those groups are masked later. **Do not use that first package.**

The final v4 package clears the entire tile once when it contains masked
groups, before copying valid groups. Fully selected rows keep the original
fallback copies, using a separate compile-time specialization. It is installed at
`/srv/ai/src/qsa-zero-invalid-20261005/opp-v4/vendors/qsa_zero_invalid_310p_transformer`.
The installed source SHA-256 is
`8f7d5c1bb6cb39657a53dc810985ce96e99f8e248514f86f04f1df7b859fea4f`;
the compiled kernel object SHA-256 is
`f7573f522b1b911e5ac70caefbb258a4f464ebd867683799c4cda1bd63537b08`.
The installer SHA-256 is
`a9aba1942e74d668e0ca7e58e995824595f94cbb6b71725f001280c38bc3bb5a`.

After the user authorized the NPU takeover, the resident GLM server was stopped
cleanly. All probes ran on one otherwise idle 310P. No full Qwen server was
started. The final v4 package passed all
[32 gather regressions](zero-v4-regression.log), including CPU-reference parity,
two KV heads, head dimensions 16/256, counts zero and around 16-group tile
boundaries, 5/17/512 selected-group widths, local/full page tables, and tails.
The [test snapshot](test_qsa_gather_value_nz_310.py) remains in this experiment;
the canonical test and kernel were left unchanged.

## Final v4 measurements

Each probe has 64 queries, one KV head, head dimension 256, 512 selected-group
slots, 2,560 visible tokens, and 16,800 allocated cache pages. The full block
table has 4,096 entries. Queued event timing uses eight samples of ten calls;
wall timing synchronizes after each call. Baseline and candidate use separate
processes with the same seeds and coherent dependency packages.

| Active groups | Gather | Baseline event | v4 event | Time change |
| ---: | :--- | ---: | ---: | ---: |
| 128 | K | 0.6145 ms | 0.4707 ms | -23.4% |
| 128 | V | 0.4424 ms | 0.3571 ms | -19.3% |
| 512 | K | 0.5621 ms | 0.5883 ms | +4.6% |
| 512 | V | 0.3627 ms | 0.3778 ms | +4.2% |

Sparse wall medians improved from 0.6942 to 0.5601 ms for K and from 0.5177
to 0.4347 ms for V. Valid K and V hashes and attention hashes match exactly
at both active-group counts. The sparse candidate has exactly zero nonzero
masked elements in both K and V. A device FP32 mean initially reported
0.999998 for the zero fraction; the probe now uses an exact host count outside
the timings. Raw records:
[sparse baseline](baseline-event-active128.jsonl),
[sparse v4](zero-v4-event-active128.jsonl),
[dense baseline](baseline-event-active512.jsonl), and
[dense v4](zero-v4-event-active512.jsonl).

**Do not promote this package to the serving default.** The dense regression
persists in queued event measurements despite specializing the two paths.
Most later long-context rows use the full 512 groups. Short prompts within the
selection budget already take the model's exact dense-attention path, so the
sparse gather gain cannot be applied to all short prompts. It may help the
early query tiles of a larger first prefill chunk, which contains both sparse
and fully selected rows. End-to-end TTFT and decode gains have not been measured.

The [named-operator follow-up](../qsa-selective-gather-20261005/README.md)
now runs both kernels in one process, retaining the baseline for dense tiles.
It improved first-chunk synthetic attention by 1.94% with parallel gather and
exact output parity. Real serving dispatch still needs scheduler-owned host
lengths for multiple requests and prefixes; avoid synchronizing device group
counts for dispatch. Both native operators must be registered before a serving
experiment. The current resident Qwen startup retains the qualified baseline
OPP stack and enables live Python experiments for other candidates.

## Reproduction

Run the saved [`run-qsa-probe.sh`](run-qsa-probe.sh) on `threadripper` from
`/srv/ai/src/qwen38-prefill-batch256-runtime-20261005/results/qsa-visible-table-20261005`:

```bash
bash run-qsa-probe.sh baseline 128 baseline-repeat-active128
bash run-qsa-probe.sh zero 128 zero-v4-repeat-active128
bash run-qsa-probe.sh baseline 512 baseline-repeat-active512
bash run-qsa-probe.sh zero 512 zero-v4-repeat-active512
bash run-qsa-probe.sh zero 128 zero-v4-repeat-regression regression
```

The active-group argument is unused by the regression action. Stage the
benchmark from `tools/qwen4exp/benchmark_qsa_table_slice_310.py`, this directory's
test snapshot, and the runner into the remote result directory before repeating
the probes. All NPUs were verified free after the final run.
