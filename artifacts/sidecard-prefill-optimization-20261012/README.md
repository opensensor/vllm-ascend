# GLM prefill optimization

**Matched 2K time to first text falls from 26.78 to 23.21 seconds (13.3% less
waiting).** An 8K prompt falls from 96.40 to 80.00 seconds (17.0%). Decode
median is 12.22 tokens/s across three fixed 2K/64-output requests. The service
is running at `http://192.168.53.187:8000/v1`, model `glm-sidecard`.

Batched CUDA KDA convolution removes the Python per-token dispatch loop while
preserving FP32 accumulation, FP16 rounding before SiLU, and convolution state.
Single-row decode retains its existing CUDA graphs. The runtime uses one RTX
PRO 6000 and physical Ascend chips 2 and 3 on one card. The other card and
hardware power, ECC, and clock settings were untouched.

## Qualification

The new prefill gate compares full-vocabulary logits against the original
32-row path on 2K prompt tokens and 64 teacher-forced output steps. It restores
runtime choices on success or failure, writes numerical evidence, and rejects
changes before API readiness. The first 32-row convolution-only trial used an
eight-step version of this gate and passed bitwise.

Optional 64/128-row candidates retain bounded 2 MiB DMA output slots, matching
routing metadata capacity, and cancellation between completed chunks. Hardware
qualification is separate from capacity admission. The 128-row trial failed
bitwise parity with maximum absolute logit error 4.5634765625, was never served,
and closed both workers with intact lifetime guards and zero module references.
The 64-row trial also failed, with maximum absolute logit error 8.16015625, and
closed normally. Both larger chunks remain rejected candidates. A separate CUDA
check at the full 24,576-channel convolution width matches the original loop
bitwise for 32, 64, and 128 rows, narrowing the mismatch beyond convolution.
The final service retains 32-row chunks and enables batched convolution. Its
stronger 64-step gate passes bitwise at all 127 prefill/decode boundaries.
All 104 frozen Python source files match the delivery source byte for byte.
The larger-batch mismatch has not been isolated to a specific operation.

## Measurements and limits

The fixed API fixture repeats the same profiler-test prompt with zero cached
tokens and eight forced output tokens. Three fresh baseline 2K runs have median
time to first text 26.779851219 seconds; the 8K baseline is 96.400062388 seconds.
The initial after timings accidentally used a different code-review prompt.
Those measurements remain under `review-prompt-*` filenames and are excluded
from matched comparisons. The final after fixture uses the exact baseline
prompt: "Inspect this module for profiler test 20261011 and explain the control
flow. " repeated to the requested length. Final matched measurements are
recorded in summary.json.

The final 2K median is 23.214437313 seconds, a 13.3% reduction in waiting and
a 15.4% increase in uncached prompt throughput. The three after samples span
22.8353–23.8546 seconds versus 26.7402–27.0230 seconds before. The final 8K
result is 79.998839544 seconds, a 17.0% reduction (one run per configuration).
All matched generated continuations are identical.

The separate 64-output decode rates are 12.2192, 12.0179, and 12.4619 tokens/s,
median 12.2192. The previous published median was 12.4249 tokens/s; that reference
comes from the previous iteration, not a fresh 64-output control in this one.
This change targets prefill and retains the decode math.

These are measured completion timings, including first-token generation and
stream delivery, rather than isolated kernel throughput. The context admission
limit remains 65,536 tokens; completed 8K prompts do not establish model quality
at the full limit. Short generated continuation text is a determinism check,
not a representative long-context quality evaluation.

## Verification and reproduction

Run `pytest --noconftest -q tests/ut/sidecard`. The relevant suite currently
passes 795 tests in both the research checkout and fork integration. All
changed-file hooks pass, including Markdown checks. Six actual API cases cover
concurrent isolation, stream equivalence, cancellation before prefill and during
decode, a 32K output budget, and over-context rejection. Arithmetic also passes
after the final matched 8K completion. Whole-repository `format.sh ci`
has the same seven failing hooks as the untouched baseline: ruff check, ruff
format, codespell, typos, clang-format, markdownlint, and forbidden imports.
It ran in an isolated worktree; legacy automated edits were not carried back.

Enable `cuda_prefill_conv` in the authenticated runtime configuration. Leave
`prefill_chunk_size` at the measured, qualified value. The full transport,
repeatability, DSA eager/captured, prefill, and startup inference gates run
before the service exposes readiness. Private environment file contents and
model weights are excluded from this evidence directory.

The architecture diagram is design.png. Executed source, source hashes,
full-model gates, worker receipts, completed API results, and check logs are
included alongside this report. Runtime paths refer to the hardware host.
