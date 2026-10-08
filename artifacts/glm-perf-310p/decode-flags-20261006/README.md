# GLM decode configuration flags — 2026-10-06

Live serving qualification of DeepSeek's existing code changes. Codex manages
deployment, builds the existing native adapters, and runs serving comparisons.
No runtime optimization implementation is added by this study.

## Deployment

Remote artifacts: `/home/matteius/experiments/glm-decode-flags-20261006`.
Server: `192.168.53.187:8001`, model `glm53-flash-selective-w3`.
Logs: `serve-baseline.log`, `serve-candidate.log`, and, if needed,
`serve-rollback.log` under that directory.

The two serving Python files differed from the local checkout only in the new
decode flags and their dispatch gates. Their previous remote contents and exact
diffs are preserved here. The packed draft slot helper already matched the local
checkout; the new tests do not constitute a new crash fix.

The installed serving extension lacked `npu_w2_swiglu_310` and
`npu_w2_route_combine_310`. A supplemental startup binding registers the exact
repository schemas, Meta functions, and unchanged native adapters. The dedicated
OPP contains only these two operators. All 18 native source files match the local
checkout; see `native-source-provenance.json`.

Both launches retain TP4, 311040 configured context, MTP1, full decode graphs
with capture sizes 2 and 8, 640-token prefill chunks, and the qualified operator
stack including KDA column caching. Qualified Sinkhorn/completed-pool
instrumentation is restored once after each startup. Codex mistakenly stopped
the recovered baseline to launch with both flags enabled. The user clarified
that subsequent experiments must use resident hot swaps. The current loaded
process will be retained: `hot_swap.py` selects either existing dispatch gate
or both, drains requests, recaptures graphs, and verifies unchanged worker PIDs
and weight storage digests. It changes no native math or weight layout.
Worker CPU affinity is recorded and applied identically.

The recovery launcher no longer has a shutdown option. It refuses to launch
while a recorded GLM process is resident; that refusal was verified against
the running server. Use `hot_swap.py swiglu`, `hot_swap.py combine`, or
`hot_swap.py both` for subsequent selections. Add `--probe` for bounded c1/c4
serving checks after recapture.

## Gates and comparison procedure

- Local grouped-bank/MoE tests: 62 passed with isolated collection.
- The seven new slot tests pass when the actual helper and test definitions are
  isolated from plugin imports. Full local proposer-test collection fails on
  missing local plugin/dependency modules; no full proposer suite pass is claimed.
- Resident selector host tests: three passed, including exception restoration,
  CPU offload, and repeated switches without nested dispatch wrappers.
- Actual ACLNN SwiGLU/combine checks: seven token shapes on all four NPUs, plus
  changed-input graph replay at both serving capture sizes. All passed.
- Device slot boundary check: passed on all four NPUs.
- Warm up c1 and c4, then retain c1/c4 repetitions with 256 output tokens per
  request. Fixed prompts, seed, and stream timing rules. The user requested
  stopping additional baseline measurements after the first completed pair;
  the remaining baseline repetitions and baseline quality checks were cancelled.
  Three candidate repetitions are planned. No three-run baseline stability
  qualification is claimed.
- Save Prometheus metrics before/after every repetition, including MTP acceptance
  and preemption counters. Record every repetition, median and spread.
- Run the 20 exact-answer checks, tool-call check, and uneven c4 completion lengths
  on the candidate. Valid requests and correct answers are separate gates.

`analyze.py` writes `comparison.json` and `decode-flags.png` once both sets of
measurements exist. These short-prompt measurements do not prove full-window
performance or context capacity.

## Recovery evidence

The original live process died at 03:46 UTC with an HDC disconnect reported from
draft-token event synchronization. The server log alone does not identify its
root cause. The subsequent fresh launch died during grouped W3 matmul tiling at
04:04 UTC and used a different operator stack. This study restores the recorded
qualified stack; it does not reuse that failed launch script.

The previous fusion experiment's saved quality results contain 21 valid requests
but three strict-answer misses, including an incorrect word reversal. Its
"20/20 quality" characterization cannot be reproduced from those artifacts.
This study records actual strict answers on both baseline and candidate.

## Result

Resident switches to SwiGLU-only and then both fusions passed c1/c4 serving
probes. All four worker PIDs and weight storage digests remained identical
across both switches; receipts are in `hot-swap-swiglu.json` and
`hot-swap-both.json`. Both fusions remain active for the performance run.

The candidate completed 21 valid quality/tool requests. Strict results are
17/20 exact answers and one passing tool-call check. The three answer misses
match the saved previous fusion experiment exactly: `pial` instead of `pmal`,
`Red` instead of `red`, and `` `lan` `` instead of `lan`. These are not a strict
quality pass. No fresh baseline answer comparison was run after the user's
instruction to stop baseline testing.

Throughput repetitions are still in progress. No speedup is claimed yet.
