# Qwen streaming hardware gate, October 9, 2026

The v5 streaming candidate is rejected. Its real-weight expert partial was
bitwise correct but about 2 times slower at 128 tokens and 3.5 times slower at
2,560 tokens. Native WY failed the downstream GDN output gate. The image-enabled
baseline was resumed for the user's testing; neither candidate was installed in
its workers. Further diagnostic NPU work stopped at the user's request.

## Baseline handoff

- Endpoint: `http://192.168.53.187:8001/v1`; model ID: `qwen38-flash-next`.
- Frozen plugin source: `61fd8ea8cd3c3e81dffc2b82ef52cfe9e23aac59`, runtime
  `/srv/ai/src/qwen38-streaming-gate-20261009`, with retained baseline OPP binaries.
- TP4/EP4 on logical devices 2–5, the original two cards. New card 9 supplies
  logical devices 0–1. ECC is now disabled on all three cards; new-card capacity
  is 47,302/46,717 MB per chip after its isolated SMP reset.
- TP6 is unsupported by this model: 16 GDN key heads do not divide by six.
  Therefore the third card does not increase this service's cache capacity.
- Context limit: 262,144 tokens; four scheduling slots; 937,737 cache tokens
  total, approximately 3.58 full context equivalents, not four full contexts.
- Images enabled: one image per prompt, maximum 1,048,576 pixels. Video disabled.
- Baseline grouped native INT4, builtin FP16 SwiGLU, CANN-v2 finalization,
  2,560-token chunks, MTP2, graph sizes `[3, 12]`, bounded prefix retention 15.
- At handoff the service was unpaused. No streaming/WY candidate hot swap,
  recapture, or further service mutation is planned while the user tests.

Fresh and cached image requests both correctly identified the screenshot as
310P3, three physical cards and six chips. Client completion times were 4.644 s
and 2.835 s; the second request reused 512 prompt tokens. This validates this
single-image workload, not every image shape or multiple-image configuration.

Text/tool smoke passed six of seven checks. The remaining string reversal returned
`DNESCA` instead of `DNECSA`. Do not describe the quality suite as fully passing.

## Candidate results

Standalone tests used the new card's logical device 0, isolated from live TP4.
The baseline was held during diagnostic work and resumed afterward. The frozen
candidate namespace is `qwen_streaming_v5`; its SHA256 is
`e65674ad94c314c52ae76d8da579ff5e1125f2c7f18e877699c85cdb76221bd6`.
Source/binary receipts were checked before loading the standalone bridge.

| Tokens | Activation seed | Baseline median ms | Streaming median ms | Streaming / baseline |
| --- | --- | --- | --- | --- |
| 128 | 1024 | 6.145 | 12.197 | 1.985 |
| 128 | 1025 | 6.107 | 12.168 | 1.992 |
| 2560 | 1024 | 45.194 | 158.202 | 3.501 |
| 2560 | 1025 | 44.878 | 157.668 | 3.513 |

These are layer 0, rank 0 expert partials with real checkpoint weights and
synthetic activations routed through the real router. All four outputs match
bitwise. Timings use three ABBA cycles, six samples per variant, untimed initial
parity/warmup calls and synchronization at timing boundaries. Temperature reads
are outside timed sections. Shared experts, HCCL and the rest of the model are
excluded. These ratios are not whole-model latency measurements.

Ten independent raw projection cases matched exact CPU correction references,
including two seeds, production projection widths, empty active rows and sparse
expert banks. State gather/scatter matched for TP-local and unsharded head counts;
local route gather matched active capacities 0, 1, 320 and 1,280.

Native WY passed component comparisons but failed the downstream GDN output
comparison at TP-local K4/V12, T128. There were 486 mismatches out of 196,608
values, with maximum absolute error 0.0466098 against `atol=0.003, rtol=0.02`.
Final-state and unsharded downstream checks were not reached. The first divergence
and its cause remain unknown because intermediate tensors were not saved. Keep
the reference WY path; do not loosen tolerances to qualify this candidate.

Early harness attempts failed because the CANN Python path was overwritten and
custom operators were not enabled. Corrected harness runs distinguish those setup
failures from the numerical WY failure. All attempted harnesses and logs are kept.

## Thermal control and local validation

The installed driver combines power, temperature and hugepage values in table
cells. The old parser rejected this layout. The parser now supports three-, four-
and five-cell layouts and still rejects missing sensors and invalid values.
Twenty-six CPU regressions and all applicable scoped manual hooks pass.
Archive hashes, relative document links and six static resource calculations
also pass.

The external corrected monitor reads all six chips: hold at any chip >=94C,
resume only when all chips <=85C, with an independent 96C cutoff. The recorded
488 valid samples reached 72C. No real-temperature hold occurred. Sustained
thermal operation and hardware hold/resume are not qualified by these short tests.
Temperature alone does not identify physical memory traffic or barrier stalls.

Required repository-wide `bash format.sh ci` failed on existing unrelated lint,
format, spelling and import findings. Its formatter edits to 144 unrelated files
were restored in an isolated worktree. Full root pytest collection is blocked by
missing installed vLLM flash-linear-attention dependencies. The compressed
[repository CI log](format-ci.log.gz) and [scoped hook log](scoped-hooks.log.gz)
retain the distinction. Scoped checks and
hardware gates are reported independently from these limitations.

## Evidence and next work

[evidence.json](evidence.json) contains exact trials, image responses, thermal
summary and a SHA256 inventory for [raw-evidence.tar.gz](raw-evidence.tar.gz).
The archive preserves actual harness source, logs, receipts and service snapshot
bytes without applying source formatters to captured files. Logs represent the
capture interval, not the user's subsequent testing.

[The offline follow-up plan](FOLLOWUP_PLAN.md) identifies concrete scheduling,
vector instruction and workspace changes. These are source-based hypotheses;
no device profiler attribution, reduced physical bus traffic, improved thermals,
full-model speed gain or candidate promotion is claimed. T9 requires a new
candidate and renewed complete qualification.
