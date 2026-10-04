# Qwen3.8 Flash Next GPQA Diamond: W4 on 310P versus RTX

On 2026-10-02, the four-chip Ascend 310P3 W4 service and the RTX 6000 Pro
`llama.cpp` service completed the same 198 GPQA Diamond questions. Both scored
**140/198 (70.71%)** under the AISBench-style `Answer: X` extractor. This is a
comparison of two quantized serving stacks, not a W8 checkpoint result or a
controlled quantization ablation.

## Protocol and runtime

- Dataset: AISBench `gpqa_gen_0_shot_cot_chat_prompt`, GPQA Diamond version
  `b1ed2c`; the 198 serialized prompts and shuffled options match byte for
  byte (cases SHA-256
  `fba8e938bb45b148f2027eea5976db039a6b182070331920605ed63f94ff9fa3`).
- Both requests used temperature 0, seed 1024, top-k 20, top-p 1, min-p 0,
  repetition penalty 1, presence penalty 0, EOS enabled, and an 8,192-token
  output limit. The server-specific chat templates and reasoning behavior
  differ.
- Ascend: `Qwen3.8-Flash-Next-W4A16-G128-300i`, native W4A8 expert
  computation, two Atlas 300I Duo cards / four 310P3 chips, TP4/EP4, MTP2,
  four concurrent requests, 262,144-token serving limit. The final 106 cases
  used `FULL_DECODE_ONLY` C3/C4 graph capture sizes `[9, 12]`.
- RTX: `Qwen3.8-Flash-Next-UD-IQ4_XS` GGUF on one RTX 6000 Pro, `llama.cpp`,
  three concurrent slots, 262,144 tokens per slot, q8_0 key/value cache.

The Ascend ledger was resumed across server changes. It includes eight original
AISBench results without client timing fields and two retained timeout rows
that were later replayed successfully. Every question has one successful result
on both services. The final 106-case Ascend graph phase ran on one server from
04:28 to 07:21 UTC without a request failure.

## Quality

| Measure | Ascend W4 | RTX IQ4_XS |
| --- | ---: | ---: |
| Correct under the AISBench-style extractor | 140/198 (70.71%) | 140/198 (70.71%) |
| Responses with a final answer | 117/198 | 124/198 |
| Correct answers in final response text | 114/198 (57.58%) | 120/198 (60.61%) |
| Reached the 8,192-token output limit | 81/198 | 74/198 |
| Total output tokens | 949,014 | 957,402 |

The extractor takes the last `Answer: [A-D]` match from the combined response.
If generation stops at the limit before any final `content`, the reasoning
field can still contain text such as “Need final line exactly Answer: D.” The
extractor credited **26 Ascend** and **20 RTX** correct choices from such
unfinished reasoning. The “correct answers in final response text” row instead
searches only text after the final `</think>` boundary. It is a diagnostic
completion measure, not the configured benchmark score.

When both services produced final answers for the same question, all **112
letters agreed**: 109 were correct and three were wrong. On the other 17
questions with only one final answer, Ascend alone completed five and RTX alone
completed twelve; the RTX-only answer on case 113 was incorrect. The paired
correct-final counts therefore differ on five Ascend-only and eleven RTX-only
cases. That six-case difference is modest on this fixed set (two-sided exact
McNemar p=0.210); it does not establish a general model-quality advantage.

| GPQA domain | Questions | Ascend extractor correct | RTX extractor correct | Ascend correct final | RTX correct final | Ascend / RTX output-limit hits |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Biology | 19 | 11 | 12 | 11 | 11 | 6 / 6 |
| Chemistry | 93 | 50 | 48 | 31 | 35 | 62 / 57 |
| Physics | 86 | 79 | 80 | 72 | 74 | 13 / 11 |

Chemistry accounts for most incomplete responses. Raising the token budget or
changing the answer format would answer a different benchmark question and
needs a separately labeled run.

## End-to-end serving performance

On the same final 106 case IDs (92–197), median client-observed output rate
(`output_tokens / request_seconds`) was **12.50 tokens/s on Ascend** and
**61.72 tokens/s on RTX**, a 4.94× ratio. Median request elapsed time was
350.96 versus 80.89 seconds. The Ascend graph phase emitted 507,256 tokens
over 2 h 52 m 46 s, or **48.93 aggregate tokens/s** with four workers. The
complete RTX run emitted 957,402 tokens over 1 h 27 m 9 s, or **183.11
aggregate tokens/s** with three slots. These wall rates describe different
case windows; the paired per-request result above is the direct comparison.
The earlier Ascend cases include a degraded server phase and do not give a
single uninterrupted whole-run throughput figure.

The final Ascend server log recorded 106 completed request timings, no eager
decode fallback, no zero-acceptance interval, and no error. Mean logged MTP
draft acceptance was approximately 69% in both halves of the run. Twelve
prefix-Mamba NPU-to-CPU spill warnings appeared; the run continued without the
previous acceptance collapse. This observation does not identify the cause of
the earlier degraded server state.

Three changes made this run practical: the 310P prefix-Mamba tier now waits
for pending NPU state writers before eviction or slot reuse (`3c4e0a685`), the
loopback GPQA runner saves each completed question and resumes missing IDs
(`5739f1597`), and the dedicated four-request graph profile captures C3/C4
decode shapes (`fdee915e8`). The recurrent-state ordering fix has its own NPU
regression test. The benchmark's healthy final phase is useful operational
evidence, but it does not isolate which change prevented the earlier collapse.

## Evidence and limits

The retained local ledgers are:

- RTX: `/run/media/matteius/ai-drive/benchmarks/results/qwen38-quality/gpqa-diamond-rtx6000pro-iq4xs-c3-rerun-20261002/results.jsonl`
  (198 successful rows).
- Ascend snapshot:
  `/run/media/matteius/ai-drive/benchmarks/results/qwen38-quality/gpqa-diamond-rtx6000pro-iq4xs-c3-rerun-20261002/ascend-complete-results.jsonl`
  (198 successful rows and two superseded timeout rows; SHA-256
  `2185ec59097fabde822b83ae13c1224bdf97105a158fc68234db72ff573e59b5`).
- Paired, question-free audit:
  `/run/media/matteius/ai-drive/benchmarks/results/qwen38-quality/gpqa-diamond-rtx6000pro-iq4xs-c3-rerun-20261002/paired-audit-complete.csv`.
- Final Ascend server log in the same directory:
  `ascend-c3c4-server-complete.log`.

No benchmark questions or response text are reproduced here. The two systems
use different quantizations, runtimes, reasoning templates, and concurrency;
the result is evidence for these configurations rather than a hardware-only
speedup or a W8-versus-W4 quality comparison. The output ceiling dominates a
large share of the measured accuracy, particularly in Chemistry.
