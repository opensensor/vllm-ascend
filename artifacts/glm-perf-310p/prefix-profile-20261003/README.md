# GLM prefix-graph decode profile and 32K context trial (2026-10-03)

The promoted TP4 prefix-graph server was profiled on the same four 310P
devices as the earlier GLM runs. A 36-token prompt generated 48 valid output
tokens at 2.61 tok/s under profiling. The profiler captured 32 worker steps
on all four ranks; parsing and analysis completed without dropped ranks.

The whole-capture cross-rank task envelope was 14.85 s. The first decode
kernel followed the dynamic prefill's last grouped projection at approximately
timestamp 1791051476700000 us. The explicit window in
[`c1-decode-window.json`](c1-decode-window.json) covers the remaining 11.96 s
of device activity. Rank 0 attributed 3.43 s to grouped W2/W4 projections,
1.81 s to AI-CPU BF16 Cast, 1.10 s to transfers/layout, 0.96 s to other
matmuls, and only 0.034 s to collective tasks. Grouped projections account
for 2,605 tasks in this decode window; Cast for 14,105. These task sums are
*not* an additive critical path. The rank-0 task union was 8.53 s, below the
11.96 s envelope, so task-level gaps/host scheduling also merit inspection.
The median matched-rank collective arrival spread was 0.081 ms. The
trace preserves the earlier conclusion that collectives are no longer the
main decode target under graph replay.

Profiler raw and parsed exports are on the NPU host at
`/home/matteius/experiments/glm-gate-a-20261002/profile-prefix-20261003`.
The all-rank reports are
`/home/matteius/experiments/glm-gate-a-20261002/prefix-c1-summary-20261003.json`
and `prefix-c1-decode-summary-20261003.json` in the same directory. The
server log is `server-prefix-22k-20261003.log` there.

The 32K launcher beside this note changes only the context limit, a 5 GiB
explicit KV budget per rank, and the unused profiler directory. It keeps
prefix caching, NZ packed codes, the combined grouped OPP, batched BF16 mHC
rounding, and FULL_DECODE_ONLY graphs unchanged. The previous 22.5K prefix
launcher is the rollback. The separate 32K+Sinkhorn launcher is an unexecuted
follow-up and has no performance or quality claim. The server advertised
32,768 tokens and planned
34,837 KV tokens, or 1.06 full-length requests. A 64-token short smoke
completed at 2.71 decode tok/s. The matched 256-token workload passed all
five requests with no early EOS: c1 was 2.682 tok/s and c4 was 7.567
aggregate tok/s. This is a separate run from the earlier 22.5K sample; no
isolated throughput gain is claimed for the context change.

The real-weight long retrieval crossed the old 22,528-token limit: 23,989
served prompt tokens produced the exact `BLUE-ORCHID-7319` answer. The cold
request took 573.4 s prefill and 581.2 s total in server timing. A repeated
streaming request reused 23,680 tokens, recomputed 309, and returned the
same exact answer with 10.65 s TTFT and 18.38 s client total (about 31.7x
faster end-to-end than cold). The retrieval generator's `27000` target is
not tokenizer-calibrated, so the suite's target-length `passed` field is false
despite the exact answer; served token usage is authoritative here.

The first streaming benchmark client stayed connected after the server
finished; its parser lacked a stop condition at `[DONE]` and waited for
socket EOF. A separate non-streaming request and direct streaming read both
returned correctly, including `[DONE]` in the latter. The
`tools.glm_perf.suite` parser now stops at `[DONE]`, with a regression test
using a non-terminating iterator.

The first 32K strict-quality pass returned valid responses on all 20 cases
but scored 16/20. It had the three known misses plus one incoherent
`arith_div` response; six immediate repeats of that exact prompt all
returned the correct `12`. A second full pass scored only 15/20: the known
three misses plus incoherent `retrieval_short_4` and `retrieval_short_5`.
Six immediate repeats of `retrieval_short_4` passed, but two of six repeats
of `retrieval_short_5` were incoherent at temperature zero. This is a real
quality instability, so **the 32K launch is not promoted** despite its
successful long-context and prefix performance checks. The known 22.5K
prefix launcher was restored for a same-prompt comparison.

On the freshly restored 22.5K server, six repeats of each of those two
retrieval prompts passed (12/12). A subsequent matched 256-token short
workload measured 2.697 tok/s c1 and 6.844 aggregate tok/s c4, with no early
EOS. Its full strict quality run scored 16/20: the usual `instr_reverse`,
`instr_first`, and `code_slice` misses plus one incoherent `arith_add` result.
Thus occasional incoherence also occurs at the shorter context. These few
samples do not isolate the 32K context or its explicit KV budget as a cause;
the 32K configuration stays a trial pending a stronger quality comparison.
The complete control requests and summary are
`/tmp/glm-22k-sinkhorn-off-short-quality-20261003.jsonl` and its
`.summary.json` companion on the development host.

The existing `mhc_sinkhorn_310` OPP is also present in the server's vendor
stack but is not enabled by the control `hf-overrides`. Earlier eager
experiments saw a gain; the graph serving trial below is the relevant
promotion check.

## Global fused-Sinkhorn trial

The fused flag alone changed between otherwise identical 22.5K graph
launchers. Both capture sizes completed and an exact-code real-weight smoke
passed. The matched 256-token benchmark (no early EOS) was **2.849 tok/s c1**
versus 2.697 control (+5.6%), but **6.063 aggregate tok/s c4** versus 6.844
control (−11.4%). The full 20-case strict quality run was 16/20, with the
same four missed IDs as its in-situ control: `arith_add`, `instr_reverse`,
`instr_first`, and `code_slice`. That does not establish answer parity—the
content of some wrong answers changed—but it shows no new failed IDs in this
sample. The global flag is **rejected** because of the c4 regression.

The complete request records are
`/tmp/glm-22k-sinkhorn-on-short-quality-20261003.jsonl` and its summary on
the development host; the server log is
`/home/matteius/experiments/glm-gate-a-20261002/server-prefix-22k-sinkhorn-20261003.log`.
The follow-up hypothesis fuses only when the mHC residual has exactly one
token row. Multi-row decode and prefill retain the upstream Sinkhorn math;
this must be tested as a distinct build and serving configuration.

That first fallback hypothesis was insufficient: mHC runs *after*
sequence-parallel sharding, so four global decode tokens can be one row per
rank. The isolated row-gated server measured 2.843 tok/s c1 but only 6.171
aggregate tok/s c4 (−9.8% versus control). Its quality suite was stopped
after the complete c1/c4 records; no quality claim is made for it. The
corrected candidate gates on the full token count in decoder `positions`
before the mHC calls and has separate source/launcher snapshots.

The corrected full-token candidate passed its focused CPU policy tests and
captured both graph sizes, but still missed the serving speed gate: **2.824
tok/s c1** (+4.7% against the earlier matched control) and **5.800 aggregate
tok/s c4** (−15.3%). All five 256-token requests were valid and had no early
EOS. Its strict quality suite was not run, so no answer-parity claim is made.
The four-rank c4 trace under
`profile-prefix-22k-sinkhorn-globaltoken-20261003` recorded 90
`MhcSinkhornV310` tasks per rank, all within approximately 0.36 seconds at
the beginning of the profiled request. It did **not** show fused Sinkhorn
throughout c4 decode; the early singleton step explains those tasks. The
full-token gate therefore appears to work, while the c4 regression remains
unexplained. The candidate is rejected, and the user asked not to restart
the control server merely for another paired comparison.

## AI Core BF16 mHC rounding candidate

The c1 graph decode trace above attributes 14,105 rank-0 AI-CPU `Cast`
tasks to 1.81 s of task time over 32 worker steps. In the corrected
candidate's trace, pairs of those `Cast` tasks occur directly after mHC
output concatenation. This is attribution, not additive critical-path time.
The existing native FP32→BF16→FP32 cast is bit-exact but dispatches through
AI-CPU on 310P. The next opt-in candidate rounds an FP32 tensor's bit
patterns in a single AI Core op, keeping FP32 storage. For finite values it
uses round-to-nearest-even BF16: `(bits + 0x7fff + ((bits >> 16) & 1)) &
0xffff0000`.

The ACLRTC probe in [`bf16-round-probe`](bf16-round-probe) matched that reference on
4, 16, 4,096, 4,116, 8,192, 8,195, 16,384, and 32,768 elements, including
ties and unaligned tails. Device-event time was 0.0038 ms at four elements,
0.0237 ms at 4,116, 0.0423 ms at 16,384, and 0.0841 ms at 32,768. The
first tail implementation using DAV-M200 `DataCopyPad` silently left the
unaligned final tile zero; scalar GM access for only the final 1–7 elements
fixed it. These micro timings do not predict serving gain by themselves.
To repeat the standalone probe on the 310P host, source the CANN environment,
compile `bf16-round-probe/runner.cpp` against CANN's `include` and `lib64`
directories with `-lacl_rtc -lascendcl`, then run
`./runner round.asc DEVICE_ID ELEMENTS` from that probe directory.

The packaged `MhcBf16RoundV310` OPP and PyTorch binding built in isolated
`/srv/ai/src/glm-bf16-round-20261003`, without changing the running server.
With its separate OPP path, the NPU parity, shape, graph replay, and CPU
policy tests passed **15/15**. The model flag
`ascend_glm_mhc_ai_core_round` is default-off; it selects the custom op for
contiguous FP32 mHC tensors of at most 32,768 elements and leaves larger
prefill tensors on the prior BF16 cast. A 22.5K graph serving trial using
the launcher beside this README captured both `FULL_DECODE_ONLY` graph sizes
and answered the short `BLUE-ORCHID-7319` smoke test exactly. The matched
256-token workload completed all five requests without early EOS: **3.034
tok/s c1** (+12.5% versus the earlier fresh 22.5K control) and **5.804
aggregate tok/s c4** (−15.2%). The c4 speed gate is not passed, so this
build is **not promoted** on the microbenchmark alone. An all-rank c4 trace
was captured under `profile-prefix-22k-ai-core-round-20261003`; its export
shows the exact substitution: rank-0 AI-CPU `Cast` count fell from **14,560
to 8,982** (−5,578), and the new AI Core rounding op ran **2,789** times,
one for each removed cast pair. Cast task time fell from 2,205.94 to
1,503.89 ms (−702.04 ms); the new op contributed 122.35 ms of task time,
for roughly 580 ms less attributed rounding work. The grouped projection
count was 2,688 in both c4 traces and its task time was effectively equal
(14,584.58 versus 14,592.63 ms). These sums overlap with other streams and
are **not** a serving latency prediction. Under profiler overhead, c4
aggregate decode rose from 4.365 to 4.487 tok/s against the corrected
Sinkhorn trace. The strict 20-case quality pass scored **16/20** with the
three recurring `instr_reverse`, `instr_first`, and `code_slice` misses,
plus an incoherent `code_python` result. In the earlier control run the
fourth miss was `arith_add`; three immediate same-settings repeats of each
affected case on the new server all passed. This is evidence of the known
first-token instability, not proof of identical quality distributions.
Gate-A remains unpassed and the candidate remains default-off.

The packaged-op synchronized eager latency test passed on 310P while GLM
was loaded: 4,116 elements took 0.0713 ms with the AI Core op versus
0.3212 ms for the native cast pair; 16,384 elements took 0.1012 versus
0.3409 ms. The serving c1 gain is real, but this isolated ratio should
not be applied to full-model throughput.

An edge-case parity check found that the first package did not match the
native BF16 cast for signaling NaNs: a signaling NaN near the BF16 boundary
could become infinity. A corrected isolated OPP at
`/srv/ai/src/glm-bf16-round-nan-20261003/opp-nan` now canonicalizes every
positive or negative NaN to a quiet NaN with the original sign. The
corrected package passed **13/13** 310P end-to-end op tests, including
bitwise native-cast parity for NaNs and infinities and a changing-input graph
replay. The combined focused model, rounding-policy, and NPU-op suite passed
**22/22** with `pytest --noconftest`; the repository-wide unit-test
`conftest.py` requires `fla_npu`, which is absent from this isolated serving
environment. Synchronized eager latency remained below the native cast pair:
0.1719 versus 0.4059 ms for 4,116 elements, and 0.1726 versus 0.4255 ms
for 16,384. The original serving measurements above used the first package;
they must not be attributed to the corrected build until it is served.

The corrected OPP was then launched on all four ranks at the same 22.5K
configuration. Both decode graph sizes captured, `/health` passed, and the
40-token exact-code smoke returned `BLUE-ORCHID-7319`. Its first matched
256-token short run completed all five requests without early EOS: **3.025
tok/s c1** and **6.482 aggregate tok/s c4**. The c1 gain versus the earlier
2.697 control is 12.2%; c4 is 5.3% below that control but 11.7% above the
first AI Core package's c4 run. That spread is too large to call a stable
c4 effect from one run. A second c4-only repeat completed at **5.382
aggregate tok/s**, also with four valid full-length replies and no early
EOS. This 20.4% spread between consecutive c4 runs is larger than the
candidate-versus-control difference. The four first-run replies also had
different content hashes, so identical prompts did not ensure identical
decode routes. A strong logit bias for token 279 (`" the"`) produced 16
identical forced tokens in a smoke request. A 256-token fixed-output c4
probe using `logit_bias: {"279": 100}` completed at **5.475 aggregate
tok/s**, with four identical output hashes and no early EOS. That is close
to the slower natural run rather than the faster first run, so differing
completions alone do not explain the full spread. A second fixed-output
c4 run gave **6.296 aggregate tok/s** with the same four output hashes and
request settings. The 15.0% gap between fixed-output repeats shows that
output text divergence is not necessary for a large c4 timing swing;
internal routing or execution scheduling remains unmeasured. The corrected
run's JSONL and summary are
`/tmp/glm-22k-ai-core-round-nan-short-20261003.jsonl` and its
`.summary.json` companion on the development host.

The corrected server's full strict-quality run scored **17/20**. The only
missed IDs were the recurring `instr_reverse`, `instr_first`, and
`code_slice`; all other requests were valid and exact. The first two misses
generated incoherent or unresolved text through the 256-token limit, and
`code_slice` returned a wrapped quoted string. This is a better single-run
score than the first package's 16/20, not evidence that the quality
distribution improved. Gate-A is still unpassed. The request records are
`/tmp/glm-22k-ai-core-round-nan-quality-20261003.jsonl` on the development
host.

The residual Cast pattern suggested a c4-specific cap miss. The first
candidate used the AI Core op only for tensors of at most 32,768 elements;
a four-token mHC residual has `4 tokens * 4 streams * 4096 hidden = 65,536`
elements. A separate Python-only candidate raises the cap to 65,536, leaving
larger prefill tensors on the original native path. The corrected OPP itself
needed no rebuild. The new size passed bitwise native-cast parity, changing-
input graph replay, and the full **26/26** focused CPU/NPU tests. At 65,536
elements under the loaded GLM server, synchronized eager latency was **0.2752
ms** for the AI Core op versus **0.3525 ms** for the native cast pair. The
gain is modest at this size; end-to-end serving gain still needs a stable
measurement before the cap can be called a throughput improvement.

The cap-extended source captured both decode graphs and passed `/health` and
the exact `BLUE-ORCHID-7319` smoke. Its matched 256-token workload was valid
with no early EOS: **3.032 tok/s c1** and **6.255 aggregate tok/s c4**.
The c1 result agrees with the prior corrected package (3.025). The c4 result
lies within that package's 5.382–6.482 run-to-run range, so a serving gain
is **not established**. The request records and summary are
`/tmp/glm-22k-ai-core-round-64k-short-20261003.jsonl` and its companion
on the development host. An expanded 310P parity test covering signaling
and quiet NaN payloads of both signs, both infinities, signed zero,
subnormals, and finite exponent boundaries passed. The cap-extended strict
quality run was **16/20**. It kept the three recurring misses and had an
intermittent incoherent `code_python` result. This is not a new stable
regression claim, but Gate-A remains unpassed; request records are
`/tmp/glm-22k-ai-core-round-64k-quality-20261003.jsonl`.

The 32-step all-rank c4 trace confirmed the cap's intended substitution on
**every rank**: AI-CPU Cast calls fell from **8,982 to 3,820** (−5,162),
while `MhcBf16RoundV310` rose from **2,789 to 5,370** (+2,581), exactly one
new op for each removed cast pair. Grouped W2/W4 call count stayed at 2,688.
On rank 0, Cast task time fell from 1,503.89 to 775.07 ms, while rounding-op
task time rose from 122.35 to 696.89 ms, giving 154.28 ms less *summed*
rounding task time over the capture. These task sums are not an additive
critical path. The separate `task_time.csv` exports give a four-rank task
envelope of **29.396 s** before versus **29.074 s** after (−1.1%); routing
and scheduling differ across runs, so this is consistent with a small gain,
not proof of a stable serving improvement. The offline parser produced valid
operator statistics and task-time CSVs for all four ranks but malformed
timeline JSON prevented `kernel_details.csv` generation for ranks 0 and 2.
The scripts [`cast_context.py`](cast_context.py) and
[`task_envelope.py`](task_envelope.py) reproduce the prior complete trace's
remaining-Cast pattern and both task-envelope checks; raw and parsed captures
are on the NPU host under
`profile-prefix-22k-ai-core-round-64k-20261003`.

A whole-capture comparison of the earlier c4 profiling traces narrows the
throughput discrepancy. The all-rank device-task envelope was **30.157 s**
for the corrected Sinkhorn-gated trace and **29.395 s** for the first AI Core
rounding trace, a 0.762 s reduction. Rank-0 task union likewise fell from
24.269 to 23.675 s; grouped W2/W4 task time was flat at 14.585 versus
14.593 s. Thus the AI Core op did not lengthen the observed NPU task
envelope in this profiled request. The lower unprofiled c4 benchmark remains
real for its sampled run, but these traces do not attribute it to rounding;
host-side scheduling or run-to-run effects still need investigation. The
generated full-capture reports are
`/home/matteius/experiments/glm-gate-a-20261002/summary-c4-sinkhorn-globaltoken-20261003.json`
and `summary-c4-ai-core-round-20261003.json` on the NPU host.
