# Offline GLM follow-up, 2026-10-03

No NPU, serving process, remote OPP, or checkpoint was touched in this pass.
The source of truth for this audit is the three saved, complete 20-case
quality-suite JSONL files under `/tmp` on the development host:

- `/tmp/glm-graph-scatter-quality-20261002.jsonl`
- `/tmp/glm-nativebf16-quality-20261003.jsonl`
- `/tmp/glm-nz-graph-quality-20261003.jsonl`

`python3 -m tools.glm_perf.audit_quality` checks each case against the
current workload's expected answer, preserves the recorded strict result,
and adds only diagnostic classifications. It accepts bounded enclosing
quote/code markers and letter case as *possible* formatting differences;
these do not turn a strict failure into a pass. The three suites each remain
**17/20**, with the same three failed case IDs:

| Case | Scatter graph | Native-BF16 graph | NZ graph | Diagnostic |
| --- | --- | --- | --- | --- |
| `instr_first` (expected `red`) | `Red` | `Red` | `Red` | Possible case-only miss. |
| `code_slice` (expected `lan`) | `` `'lan'` `` | `` `'lan'` `` | `` `lan` `` | Possible answer-wrapper miss. |
| `instr_reverse` (expected `pmal`) | `pial` | Ends in `pmal` after extra explanation | `pial` | Two wrong reversals; one final with extra text. |

All three responses completed rather than running out of answer tokens. This
is not evidence that the graph or NZ layout caused the failures: the same IDs
also fail on the earlier graph path, and the prior eager comparison reported
the same strict misses. It also does not prove the checkpoint's math is
correct; a trusted reference using the *same* quantized checkpoint is still
needed before blaming or exonerating a kernel. Keep the exact-answer Gate-A
unchanged and report the diagnostic labels alongside it.

The existing four-rank trace analyzer now preserves per-category task union
and lists the largest exact kernel-name/input-shape pairs, in addition to
its task sums, and ignores CANN's aggregate `Total` communication rows. It
explicitly labels grouped W2/W4, AI-CPU `Cast`, fused mHC Sinkhorn,
TopK, integer bit operations, and other matmuls. Generic TopK cannot be
assigned to kpool without host/operator correlation. The window's envelope
is an elapsed bound, **not** a dependency-chain critical path; overlapping
task sums cannot be added to predict tok/s.

## Packed-field arithmetic candidate (CPU proof only)

`tools.glm_perf.packed_fields` is a small importable helper, not a device
benchmark or installed operator. Its exhaustive tests compare all 256
possible packed bytes for both signed W2 and W4 against integer-shift
extraction. They prove that FP16 `RINT(byte / divisor - (divisor - 1) /
(2 * divisor))` gives the exact quotient for divisors 4, 16, and 64 on the
tested arithmetic path. Those quotients reconstruct the same low-to-high
two's-complement fields without a runtime integer vector shift. No signed
weight transform, scale multiplication, NZ reorder, or Cube accumulation was
changed.

Qwen already uses a related FP16-biased `RINT` for W4, but its load-time
sign-bit toggle and group-128 scale/offset layout differ from GLM's raw
signed W2/W4 codes and block-32 FP32 scales. The Qwen code also keeps
decoded FP16 tiles in L1; it does not execute Cube directly on packed INT4.
Thus copying its whole kernel would not preserve GLM parity. GLM's default
grouped operator still stages decoded tiles through GM. The opt-in
singleton-L1 candidate is separate and unpromoted.

Only consider a guarded GLM decode-kernel implementation if a complete
four-rank graph trace identifies unpack/dequant as a meaningful fraction of
the actual critical path. It would still need an isolated CANN build, exact
W2/W4 byte and scale parity on device, all route-shape tests, and latency
measurements against the same compiled baseline. CPU arithmetic parity does
not establish that CANN emits a faster instruction sequence or that the
operator's output remains bitwise equal.

## Next GLM hardware session, once the four NPUs are available

1. Freeze the exact W4through32-noclip checkpoint, source, NZ code layout,
   known-good rowtile OPP, BF16 mHC state math, device-resident MLA, parser,
   and graph capture sizes `[1,4]`. Do not load the rejected singleton-L1 or
   shift-extraction candidates. Record hashes and graph replay evidence.
2. Re-run `instr_reverse` several times at temperature 0 in matched eager
   and graph modes, keeping prompt, chat template, parser, reasoning effort,
   seed, and checkpoint identical. Compare full reasoning/final text and
   first divergent token/logits if an answer difference appears. Also re-run
   the complete 20-case strict suite and report diagnostic classes separately.
3. Capture all four ranks for a steady c1 decode, four-stream decode, one
   512-token prefill chunk, and a representative 8K retrieval. Use explicit
   per-step windows with `tools.glm_perf.analyze_trace`; record top kernel
   shapes, rank arrival spread, collective wait/transit, host-side replay
   overhead, HBM, and finished-answer validity. A partial-rank export is not
   enough to select the next optimization.
4. Choose one isolated candidate from that trace. If grouped projection
   remains dominant, attack *per-active-expert* packed-tile traffic with a
   new schedule and exact W2/W4 parity over empty/singleton/repeated/peer
   routes. If the trace points instead to kpool or another path, do not spend
   another cycle on grouped-kernel guesses. The old 128-row W2 Cube cap does
   not govern the measured resident grouped prefill path.

MTP-1 and prefix caching stay later work: GLM's draft loader/state rollback
and recurrent prefix-state correctness are not qualified. This offline pass
does not make a performance or production-quality promotion claim.

Offline verification: 24 focused quality/trace/packed-field CPU tests passed;
45 tests passed across all `tests/ut/glm_perf`. A broader local suite reached
52 passes but 21 kpool import-dependent failures because
this local Python lacks `torch_npu` and its installed vLLM lacks
`kv_cache_spec_registry`; those are environment failures, not NPU parity
results. Re-run the kpool tests in the matching development environment.
