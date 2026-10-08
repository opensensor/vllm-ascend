# GLM prepared expert offsets and shared QSA cache reuse

Two optional native experiments are implemented and loaded into the real GLM
server: decode v984 / prefill v985 prepare invariant expert offset tables on the
CPU; QSA v991 reuses an aliased K/V tile in L1 and batches accumulator/output
vector operations across NZ blocks. Arithmetic order and precision
remain unchanged. The checkpoint keeps permanent packed weights and FP16 scales
under `fp16_storage_fp32_compute_v1`.

## Changes and evidence

- `--prepared-offset-tables` replaces scalar UB table construction in every expert
  kernel task with aligned DMA into existing UB allocations. Tagged descriptors
  are cached by geometry. The first eight INT64 fields and launch pointer counts
  remain unchanged. CPU tests compile the original table builders and compare
  every byte across multiple K sizes and schedules. Neither extra L1/UB storage
  nor model weight transformations are introduced.
- QSA source staging requires the qualified parent SHA and rejects changed DMA
  sites. The optimization checks raw K/V pointer equality on the AI core. It
  skips the duplicate value gathers and second UB-to-L1 copy when the pointers
  match. The separate-cache path retains both gathers. Existing UB allocations
  remain; this candidate does not claim a cache-memory saving.
- The live runner cache audit found 12 aliased attention cache pairs on each rank.
  The QSA adapter uses the production tiling arithmetic and rounds Q24 scales
  exactly like C++ `llround`. Reversible instance bindings prepare descriptors
  before capture and preserve the original static class getter.
- The direct QSA parent and candidate share the same private launch wrapper.
  ACLRTC requires explicit reads of the original 14-INT64 tiling structure and
  the 310P `KERNEL_TYPE_AICORE` annotation. All 30 full cases match the installed
  operator and compiled parent bit for bit, including three changed-input/cache/
  metadata graph replays per case and separate/aliased caches. Dense cases also
  pass an independent FP32 reference. All four real workers passed 18 bounded
  admission cases before hot capture.
- Expert v984 and v985 each passed 30 independent math cases, 12 real-weight
  cases and 60 paired boundary cases with three changed replays on device 0.
  Both versions then passed 12 bounded admission cases on each serving rank.
  The qualification covers W2/W3/W4 weights and A4/A8 activation arithmetic.

## Measured performance

The expert-table change has no established prefill speedup. Its serving pass
measured 5.36 seconds cold for 640 tokens, 11.39 seconds for 1,280 and 65.13 seconds
for 6,400. C1 synthetic-token generation measured 9.53–10.05 tok/s.

Shared K/V reuse alone (v989) was essentially neutral end to end. Repeated
same-wrapper parent cold times averaged 11.433 seconds for 1,280 tokens and
64.644 seconds for 6,400; shared-only averaged 11.508 and 64.722 seconds. This
control does not support promoting a shared-only throughput improvement.

The stronger v991 batches FP32 accumulator rescale/add and final scale/cast
across NZ blocks. The multiply and add remain separate operations in the same
per-lane order; the final cast remains FP16. At head dimension 512 the rescale
barrier moves from once per 16-element block to once per head, avoiding 32
separate loop/command/barrier sequences for each active head and tile. No extra
UB or model memory is introduced. Independent flags are `--vector-output` and
`--vector-accumulate`. Flag-off preprocessing reproduces the original code.

All 30 complete cases for v990 (output only) and v991 (output + accumulator) are
bit-exact to both the compiled parent and installed operator, including changed
replays. All four real workers each passed 18 admission cases for both versions.
The 640-row v991 dense fixture fell from 12.264 to 4.799 ms; sparse fell from
97.809 to 28.205 ms (71.2% less latency). This is an attention-kernel measurement,
not a 3.5x whole-model speed claim.

The first v991 real serving pass measured 10.95–11.05 seconds for 1,280 tokens
and 57.70 seconds for 6,400, versus the parent control's 11.433 and 64.644 seconds.
That is about 3–4% and 10.7% less cold request time, respectively. The repeat measured 10.75 seconds for 1,280 and 57.93 seconds for 6,400.
Both long cold measurements reduce request time by 10.4–10.7%; they are included
in `qsa-v991-repeat-serving.json`. The short 640-token
prompt remained around 5.42 seconds with no proven speedup. Final C1/C4 records
are in `qsa-v991-final-serving.json`; decode measured 9.77–10.03 tok/s, and C4 full-request aggregate throughput was
10.28 tok/s. This batch does not establish a sustained decode improvement.

Synthetic token-ID outputs are timing fixtures, not language-quality checks.
The final real-text HTTP response verifies functional inference; comprehensive
model quality is unevaluated. Timing includes MTP acceptance effects. C4 reports
full-request aggregate throughput separately from C1 generation.

## Runtime and limits

The server uses four Ascend 310P devices, TP4/EP, FlashComm1, MTP1, prefix caching,
640-token chunks and decode capture sizes 2/8. Configured maximum context remains
311,040 tokens with four request slots. This batch tests up to 6,400 tokens;
configured capacity is not a full-capacity qualification. Prefill still has live
attention/indexer boundaries (46 graph segments / 45 eager breaks for 640 rows).

Real-weight gates and HTTP inference are used; no dummy-only sign-off. This is
an incremental performance experiment, not a new architecture adaptation, so
existing model/tutorial/accuracy configurations remain the applicable baseline.
Multimodal inputs are disabled by the retained GLM launcher. No transformers
upgrade, new environment variable, runner behavior or production model patch is
introduced by this batch. No GitHub issue comment is posted.

Initial expert admission failed because the inventory requested a nonexistent
decode route-input binary. A later optional cache audit incorrectly assumed a
Python type metadata field was a string. Both failed native sessions were
replaced with fresh, identity-checked paused GLM workers before further admission.
The audit predicate was corrected, and all final gates passed. QSA compile
attempts v987/v988 failed on the generated tiling macro / core annotation; v989
compiled and passed the complete gates. The first free QSA probe lacked the
serving libopapi stack; it resumed the server, then passed with the saved stack.
Failure logs are retained with successful receipts.

CPU validation: 1,555 tests passed. `pytest --confcutdir=tests/ut/glm_perf -q tests/ut/glm_perf`.
The top-level UT conftest needs an unavailable upstream Flash Linear Attention
package, so this standalone tooling suite intentionally uses its own directory
as the conftest boundary. Scoped Ruff/manual hooks passed. Required `bash format.sh ci` was run and failed
on existing repository-wide lint/format/import issues. Its 145 unrelated
formatter edits were restored in the isolated checkout; the log and restoration
inventory are retained. Large JSON snapshots use `.json.gz`; decompress to recover their original
bytes. Frozen bundles and executed protocols retain their
original bytes; formatting does not rewrite tested evidence.

See [Chinese report](README.md), [runbook](RUNBOOK.md), raw `measurements/`, frozen
`bundles/`, executed `protocols/` and `SHA256SUMS.json`.
