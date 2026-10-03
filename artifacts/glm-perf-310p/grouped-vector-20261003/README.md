# GLM grouped-vector candidates, 2026-10-03

The four-rank GLM server remained on the known-good OPP while separate
single-310P operator packages tested two vector-dequant changes: one Muls per
pair of NZ 16x16 fragments sharing a [32,32] scale, and biased-RINT packed
W2/W4 extraction. The packages retained the same grouped operator ABI and
used the full 4096-output GLM shapes, NZ-packed codes, identical random
inputs/scales and route boundaries, two warmups, and five synchronized
repetitions. The baseline object SHA-256 was
`fd115612355c0dce96983d302806c0422a68a84c9e0a196c5de8464c6118ce10`.

Both fast schedules exposed a W4 packed-byte buffer reuse hazard when built
without a V-to-MTE2 handoff after the byte-to-FP16 Cast. The first scale-pair
binary failed all 15 W4 full-output cases, although all W2 cases passed.
The varying-K, one-hot W4 probe found wrong upper-nibble values in 16-output
blocks even with unit scales; the known-good binary passed. Moving the
handoff directly after Cast fixed the probe and 30-case full-output parity
without the large slowdown incurred by fencing all preceding vector work at
the next tile. The signed build-cache dependency fix `a48a30f9e` ensured each
shared-header change regenerated grouped objects rather than falsely hitting
the old binary.

| Candidate | Grouped object SHA-256 | Bitwise cases | W4 aggregate | W2 aggregate |
| --- | --- | ---: | ---: | ---: |
| Scale-pair plus handoff | `1aace904360b9ffdc2ce01b22a509a3068132e2931dfac8e07dc29b2db71e768` | 30/30 | 9.63% faster | 8.07% faster |
| RINT plus handoff | `8f0151af4b43ff19747f746007d42b6c1f48f217124bd20b06ae9f48fa8e0236` | 30/30 | 8.66% faster | 0.72% faster |
| Combined | `b211354d0b47407722a5805f974bfeb1722d07d252c0adcf830f60f2ca5e1de3` | 30/30 | 18.55% faster | 8.42% faster |

These percentages are the reduction in the sum of case median operator
latencies, not serving throughput. For the combined binary, the 8/32-row
decode-only 20-case subset was **15.12% faster** in one baseline-first paired
sweep; the 416-row subset was 14.38% faster. A second,
candidate-first/baseline-second decode sweep with seven timed repeats per case
passed all 20 bitwise comparisons and reduced the sum of median latencies by
**15.20%** (W4: 19.27%, W2: 8.62%; no individual-case regressions). This
clears the 15% operator gate, narrowly. The matched serving and quality
results below, rather than this operator result, determine the runtime
decision.

The original 30-case comparison files are on the NPU host under
`/home/matteius/experiments/glm-gate-a-20261002/` as
`w2-scale-pair-v2-comparison-nzpacked-20261003.json` and
`w2-rint-scale-pair-comparison-nzpacked-20261003.json`; the repeat is
`w2-combined-decode-repeat-comparison-20261003.json`. The combined package
is at `/srv/ai/src/glm-w2-rint-scale-pair-20261003/opp-combined`; the source
header SHA-256 is
`f1debf371bc794f585cef7f1346f0a6ff2536345c1e8cb97c774238636f6cb62`
and the self-extracting installer SHA-256 is
`46da44c60d8fc6602164b21f58d2fa6a11ddd258ab5f7740ab2524f9b0141e9a`.
The generated compiler options include both
`-DGLM_W2_SCALE_PAIR` and `-DGLM_W2_GROUPED_RINT_UNPACK`.

Matched four-rank serving used the same checkpoint, source tree, FULL_DECODE_ONLY
capture sizes `[1,4]`, and launcher flags, changing only the grouped OPP and
trace directory. Both 256-token workloads were valid with no early EOS:

| Workload | Known-good | Combined | Relative gain |
| --- | ---: | ---: | ---: |
| One stream, 256 tokens | 2.406 tok/s | 2.672 tok/s | +11.09% |
| Four streams, 256 tokens | 5.801 aggregate tok/s | 6.255 aggregate tok/s | +7.82% |
| One stream, 32-token fault smoke | 2.470 tok/s | 2.735 tok/s | +10.71% |
| Four streams, 32-token fault smoke | 5.435 aggregate tok/s | 6.064 aggregate tok/s | +11.58% |

The 32-token fault runs are not speed-qualification samples. The 256-token
four-stream result misses the PRD's 10% end-to-end gate. On October 3 the
user explicitly chose to promote the faster experimental runtime anyway;
this is a documented exception, **not** a claim that the original promotion
gate passed. The candidate's 20-case quality suite was 17/20 with exactly
the same three failing case IDs as the earlier baseline. However, two of
those already-wrong final strings differed from the saved earlier run
(`instr_reverse`: `pial` to `pma`; `code_slice`: backtick-wrapped `lan` to
backtick-wrapped quoted `lan`). An in-situ baseline rerun after restoring the
known-good OPP showed these prompts are output-unstable even at temperature
zero: two `instr_reverse` replies differed from each other, and two
`code_slice` replies reproduced both backtick variants. The candidate
therefore has no *observed strict-score* regression, but deterministic
whole-model answer parity is not established by this suite.

The local request records are `tmp/glm-combined-{baseline,candidate}-serving-20261003.jsonl`
and `tmp/glm-combined-candidate-quality-20261003.jsonl`, each with a
`.summary.json` companion. The candidate server log is
`/home/matteius/experiments/glm-gate-a-20261002/server-mhc-combined-20261003.log`.
The known-good launcher was restored afterward on port 8001 (PID `1059455`),
and its health and grouped-OPP environment were checked. Its restart log is
`/home/matteius/experiments/glm-gate-a-20261002/server-mhc-baseline-restored-20261003.log`.
The user then directed promotion. The combined OPP was rechecked against its
recorded NZ grouped object hash and relaunched using the same tested launcher;
the new log is
`/home/matteius/experiments/glm-gate-a-20261002/server-mhc-combined-promoted-20261003.log`.
The exact launch script is saved beside this README as
`serve-glm-mhc-combined-graph-nz-20261003.sh`.
The repo's grouped-op CMake flags now enable both optimized schedules by
default for that operator, and the promoted shared header is byte-identical to
the tested source hash above. The standalone operator's default build is
unchanged. Keep the known-good rowtile launcher as rollback.

Promoted restart PID `1480146` passed `/v1/models`, captured both
FULL_DECODE_ONLY sizes, and completed fresh real-weight 32-token one- and
four-stream smokes with no early EOS. Their decode rates were 2.744 and
6.051 aggregate tok/s; these are restart checks, not replacements for the
paired 256-token serving results above. The smoke records are
`tmp/glm-combined-promoted-smoke-20261003.jsonl` and its summary.

The combined package passed 16 additional 310P pytest cases for the shared
standalone W2/W4 operator, canonical versus NZ grouped outputs, scale
application, 72-expert routing and the varying-K byte-reuse regression. The
local dirty worktree already contained a default-on scale-pair change; a
minimal V-to-MTE2 handoff was added to that header without staging its other
in-progress edits. That complete dirty-tree build has not been qualified.

## Next kernel hypothesis

The trace's four-stream grouped task sum was 10.93 s in a 25.95 s all-rank
capture envelope before this package. The 15% operator reduction can only
remove part of that envelope, consistent with the measured +7.82% serving
gain. The older resident-L1 path is not a fallback: its 32-output-channel
tile is four times narrower than the current 128-channel GM tile, and its
measured singleton cases were slower. A new *unimplemented* experiment could
stream K in 1024-element blocks while retaining a 128-channel decoded B tile
in L1. A 128 x 1024 FP16 weight tile is 256 KiB; with the existing two
128 x 128 activation stages, this is about 320 KiB within the 512 KiB L1
budget, while two 128 x 128 FP16 B stages use the 64 KiB L0B budget. It would
keep the wider output tile and avoid the GM dequant round trip, but requires
correct partial-K Cube accumulation and explicit V/MTE/L1 lifetime checks.
The earlier L1 parity and latency failures must remain hard negative controls.
