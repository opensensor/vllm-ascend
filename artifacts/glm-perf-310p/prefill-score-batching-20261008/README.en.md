# GLM cold-prefill score batching, October 8

The new `GLM_KDA_SCORE_ROW_BATCH` experiment batches gate arithmetic and product
casts in the complete KDA prefill kernel. It composes with the qualified column
cache, FP32 beta products, score reductions, vector tail W/U and gated-key reuse.
It uses otherwise dead storage in the existing 128 KiB vector arena: half gates
and products at 16 KiB, FP32 products at 32 KiB, ending before 64 KiB. Immutable
keys/gates start at 80 KiB. **No extra UB allocation** is introduced by this change.

For each valid full-chunk score row, column gate differences retain their FP16
rounding, then Muls/clamp/Exp/key multiplication run across the whole row. K×K
and Q×K remain separate passes. Each pass performs one batched FP16-to-FP32
product conversion; the two original 64-lane FP32 reduction trees and +0/p0/p1
combination order remain. Output fences and later solve stages are unchanged.
Only SAFE_GATE, FP16, K128, BT64 and a live 64-row column cache admit the new path.
Partial chunks retain the already-qualified tail implementation.

## Qualification

- All 17 production safe-gate complete-operator cases preserve every returned
  output byte, recurrent carry, repeat output and input tensor. Twelve outputs
  are compared per case, including both BSND/BNSD, tails and multi-chunk inputs.
- The two nonsafe reference cases are already nonfinite. They are diagnostic
  evidence, not qualified operating modes; production keeps safe gating enabled.
- The first compilation failed because a by-value tensor slice was passed to a
  non-const reference. v2 names the column-product alias. A host syntax regression
  covers the scalar and vector reduction branches.
- The first resident-side standalone attempt failed in `torch.npu.set_device`,
  before any fixture or candidate arithmetic: device subprocess startup timed
  out (507033/E39007). The resident chips were nearly fully reserved. That
  observation suggests insufficient room for another context; the error itself
  does not establish a precise memory cause. The idle, identity-checked GLM tree
  was stopped, and the free-device candidate gate passed. The real server was
  then reloaded and all eight existing native resources restored.
- Saved reference outputs come from the qualified tail-W/U v2 fixture. Deployed
  gated-key reuse v2 already matches that reference on the same 17 cases. The
  reference SHA is recorded; no new baseline benchmark was run.

Complete 640-token KDA timing is 17.70 ms BSND / 17.49 ms BNSD, versus archived
qualified-parent 26.14 / 25.84 ms. The three other gate means take 17.65–17.74 ms.
These are free-device candidate measurements against archived parent timings,
not a fresh paired throughput comparison or an end-to-end speedup claim.

## Serving scope

The existing real FP16-scale disk checkpoint, target/draft A4, MTP1 and eight
native resources are retained. The new package replaces only the previous KDA
OPP vendor slot. No checkpoint transformation, dependency upgrade, new runtime
environment variable or model-runner behavior is introduced.

Serving remains TP4 on four 310P3 chips, port 8001, four request slots,
640-token chunks, prefix caching and full decode graphs for 2/8 tokens. Prefill
still contains captured segments with live attention/indexer eager boundaries;
this is not a completely fused prefill graph. Configured context is 311,040;
this batch does not qualify that maximum or expand the user's four-slot setup
to the skill's 16-slot baseline. EP/FlashComm1 and multimodal support are not
newly qualified; image/video inputs remain disabled. No dummy weights are used.
Generated-text quality is left to the user's evaluation.

The current server log is
`/srv/ai/artifacts/glm-prefill-weight-reuse-20261007/score-row-batch-server-20261008.log`.
Process identity and all-four-rank source/package admission receipts are saved
alongside it. E2E measurements and final checks are recorded below.

## Evidence and reproduction

The [serving scan](SERVING-SCAN.en.md) distinguishes actual casts/copies from
reference paths, load-time work and same-dtype no-ops. See the
[Chinese report](README.md) and [runbook](RUNBOOK.md). Measurements include frozen
source/binary hashes, complete operator gates, archived parent times, failed
build/context logs and the qualified package. Large fixture `.pt` tensors and
model shards stay on the host. Controller protocols are frozen `.py.txt` files;
they are evidence of this exact deployment, not general launchers.

## End-to-end cold prompts

| Prompt tokens | Archived parent TTFT | Row-batch TTFT |
| --- | --- | --- |
| 640 | 5.63 s | 5.40 / 5.36 s |
| 1,280 | 12.32 s | 11.68 / 11.43 s |
| 6,400 | 68.17 s | 65.36 s |

These are cold synthetic token-ID completion requests with one output token and
prefix state deliberately reset. Parent numbers are archived, not a fresh paired
control; repeats are available for the new 640/1,280 requests only. The 6,400-token
request completes approximately 2.8 s earlier than the recorded parent. Short
63-token requests remain 1.26–1.30 s; 639-token TTFT is 6.17 s.

Synthetic C1 generation is 9.83 / 9.62 tok/s. C4 total throughput, including
prompt processing, is 9.58 / 10.37 / 10.84 aggregate tok/s. These do not establish
a decode speedup. Generation timing includes the full request completion tail;
first-to-last visible-fragment rates are saved separately. All four ranks keep
clean graphs, the admitted source digest and zero expert fallbacks. A real chat
smoke returns a coherent photosynthesis explanation with HTTP 200. This is a
functional check, not an accuracy evaluation.

## Local checks

The new nine staging tests pass; 1,503 GLM performance tests and 339 available
GLM W2 tests pass (1,842 total). The broad local W2 run encounters three existing
files requiring unavailable torch_npu or upstream vLLM model/cache modules:
`test_glm5next_w2_assembly.py`, `test_grouped_gate_up.py`, `test_kpool_ops.py`.
Their failures are preserved; they are excluded from the available CPU gate.
A default pytest invocation also encounters the unrelated top-level runtime
conftest; the CPU checks use the focused directory's confcutdir. Hardware evidence
comes from the full KDA gate and real serving runtime, not mocked dependencies.

Required `bash format.sh ci` was run in the isolated checkout. It reports existing
repo-wide style findings and formats legacy files; those unrelated edits were
restored. Scoped hooks pass on the owned changes. Compiler and lint logs are
compressed without changing their bytes, avoiding false secret/spelling findings
on compiler cache digests and quoted existing lint violations.

Prefix replay is also checked: 1,280 tokens take 11.37 s cold and 6.06 s on an
identical replay, with **640 actual prefix hits**. The final 640-token block
still runs. One direct real-text completion measures 9.03 tok/s through full
completion; it is not a chat-template or language-quality evaluation.

## Fresh serving profile and next targets

CANN capture covers decode and two 640-token chunks from a cold 1,280-token
request on all four ranks. In the last rank-zero prefill step, source-confirmed
nine-launch ordinal attribution assigns stage 7 (score finalization) 417.11 ms
across 34 layers, versus the archived gated-key-reuse trace's 702.97 ms. Complete
KDA tasks sum 552.58 ms. The step spans 6,151.34 ms versus archived 6,515.00 ms.
Profiler overhead applies; summed task durations are not an additive critical
path, and this is not a fresh paired control. Decode spans 210.35 ms versus
archived 205.66 ms, consistent with no demonstrated decode improvement.

Remaining last-step task sums: expert gate/up 1,661.68 ms, down 849.94 ms, sparse
attention 758.36 ms, routed reduction 133.04 ms and pack 115.76 ms. AI_CPU still
includes 42 ScatterElements tasks (8.25 ms), 24 FloorDiv tasks (1.71 ms), two
ReduceSum cast tasks (0.36 ms) and smaller operations. We have not eliminated all
AI_CPU work. Mixed cast families include Matmul/Cast 34.85 ms (266 tasks) and
InplaceCopy/Cast 22.69 ms (1,376 tasks); these do not map directly to grep sites.

The next substantial targets are repeated expert reconstruction/layout work and
QSA cache traffic. Within KDA, the new batched matrices also permit a bounded
repeat-based Sub/Mul and two-tree reduction schedule instead of separate
per-column instructions. That follow-up is **not implemented or qualified** by
this batch. Precision and carry-state checks must remain, followed by whole
prompt timing. The serving stack is healthy and unpaused after profiling; its
workers, weights and admitted source digest were retained across source switches.

Raw rank-zero op/step CSVs, all-rank attribution, profile receipts and the frozen
profiler candidate are included. Source SHA for the staged header is
`131bfc255e0e66bf528f51b759cea529af3c2823c442e3dcf1f6f767d1c75a2c`;
the kernel binary SHA is
`f8ed83dc3536500def87dcf49298b3d9941d48b8878471dcc9272d79c162972c`.
The Python serving factory retains SHA
`33b6d50f3ffc2bf6212a3f968082fc5cf5dc3f85f0ee6da3601f9f08d0b35a49`;
all four workers' OPP environments admit the new private KDA vendor.
