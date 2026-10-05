# Paired Qwen3.8-Flash-Next GPQA Diamond sample (2026-10-01)

The first three questions (IDs 0, 1, 2) from the existing AISBench GPQA Diamond
reference run were sent serially to both live servers. The exact user prompts
and option order came from the frozen RTX reference predictions at
`/run/media/matteius/ai-drive/benchmarks/results/qwen38-quality/gpqa-diamond-deterministic-iq4xs/20260923_103113/predictions/vllm-api-general-chat/GPQA_diamond.jsonl`.
Matching prompt SHA-256 values in `rtx.jsonl` and `ascend.jsonl` verify that
each pair received the same user prompt.

Both requests used a maximum of 8,192 output tokens, temperature 0, seed
1024, top-k 20, top-p 1, min-p 0, no presence or repetition penalty, and EOS
enabled. The servers used their own chat templates and reasoning settings.
Both returned a final `Answer: LETTER`; the last such answer in visible content
was graded against the GPQA key.

- RTX: llama.cpp, one RTX PRO 6000 Blackwell Workstation Edition,
  `Qwen3.8-Flash-Next-UD-IQ4_XS` GGUF.
- Ascend: vLLM Ascend, two Atlas 300I Duo cards / four 310P3 chips, TP4/EP,
  native W4A8 backend with MTP2 and decode-only graphs.

| GPQA question | Gold | RTX answer | RTX output | RTX decode | RTX elapsed | Ascend answer | Ascend output | Ascend decode | Ascend elapsed |
| --- | :---: | :---: | ---: | ---: | ---: | :---: | ---: | ---: | ---: |
| 0, quantum-state lifetimes | D | D | 746 | 97.70 tok/s | 9.04 s | D | 635 | 29.79 tok/s | 22.20 s |
| 1, organic synthesis | C | C | 6,196 | 96.25 tok/s | 64.86 s | C | 3,012 | 27.42 tok/s | 110.63 s |
| 2, spin expectation | B | B | 611 | 98.11 tok/s | 6.78 s | B | 566 | 29.87 tok/s | 19.79 s |

All three answers were correct on each server. Median per-request decode was
97.70 tok/s on RTX and 29.79 tok/s on Ascend, a 3.28× ratio. Total elapsed
time for these three serial questions was 80.69 s versus 152.61 s, a 1.89×
ratio. The wall-time ratio is smaller because the Ascend model produced about
half as many tokens on question 1. Decode rate uses `(completion_tokens - 1) /
(last streamed output timestamp - first streamed output timestamp)` and counts
reasoning tokens. Elapsed time includes prefill and the full response.

This three-question sample checks paired behavior and sustained decode; it is
too small to estimate GPQA accuracy. Quantization, implementation, chat
templates, and reasoning settings differ between the services. The questions
have short inputs and do not test long-context performance.

`run_sample.py` reproduces the requests. `rtx.jsonl` and `ascend.jsonl` contain
the per-question timings, token usage, content, and reasoning output.
