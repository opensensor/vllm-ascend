# M3.1 packed W4 / INT8 compute promotion

The live six-chip service uses `minimax_w4_int8_v13`, TP2/PP3/EP2, 163840-token
capacity and 512-token prefill batches. No worker restart or shard reload was
needed. See `summary.json` for aggregate measurements and the raw request and
activation receipts for individual samples.

Three matched 64-token requests measured median 1.830 tokens/s with FP16 v9,
1.956 with exact FP16 byte-unpack v12, and 2.231 with INT8 v13. These are decode
rates after first output, including per-token weight/activation conversion.
The final gain is 21.9% over the starting v9 service, not a fourfold speedup.

The resident conversion repacked 342 banks (192.375 GiB), compared four routed
real experts per projection bank, and retained all six worker PIDs. Maximum
projection NRMSE was 1.278%; this does not establish broad model quality. Three
small greeting/tool/Python checks passed, alongside the three counting requests.
No expensive broad verification or full 160K run was repeated.

The INT8 kernel uses K32/N16 Cube layout, fused small-row activation conversion,
and one strided UB-to-L1 DMA instead of eight fragment transfers. Larger route
sets use dynamic activation quantization and bounded 128-row expert tiles.
Synthetic sizes through 512 rows and real projections through 128 rows passed;
2048-route geometry is implemented but was not benchmarked here at full size.
On this dav_m200 compiler, half-to-INT8 CAST_RINT and vector ShiftRight did not
produce the required results. The working implementation uses CAST_ROUND and
masked byte fields. Earlier failing variants were not promoted.

Probe v8–v11 used FP16 v9 as baseline. INT8 probes v12/v13 used FP16 v12, although
the older probe script still labeled the baseline variant `v9` in those raw
records. `m31-v12-full-probe.json` compares FP16 v12 against FP16 v9. The current
probe records a neutral baseline label and its explicit path.

`activate-native.sh` is the installed fresh-worker post-start script. Roll back
using `python -m tools.minimax_m3.resident_int8 disable` with the INT8 manifest
and `--baseline minimax_w4_v12`; never dispatch FP16 against INT8-repacked bytes.
CPU rollback and the bounded device byte round trip passed. Full live rollback
was not repeated. Existing image, retrieval, and broad tool results apply to the
previous FP16 service, not this approximate INT8 arithmetic.

The isolated CPU suite passed 197 tests, with 24 hardware skips. The complete
format command was run against the pre-change baseline in an isolated worktree
and encountered existing failures summarized in `format-ci-summary.txt`.
Changed-file hooks are checked separately. Raw traces are diagnostic evidence,
not an lm-eval accuracy measurement or an unprofiled throughput guarantee.
