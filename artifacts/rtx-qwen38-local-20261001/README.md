# Local RTX PRO 6000 Qwen3.8-Flash-Next throughput (2026-10-01)

Server: llama.cpp `llama-server`, one RTX PRO 6000 Blackwell Workstation Edition,
`Qwen3.8-Flash-Next-UD-IQ4_XS` GGUF, three 262,144-token slots, Q8_0 KV cache.
The short requests below do not validate long-context performance.

Method: OpenAI-compatible streaming chat, temperature 0, thinking disabled,
`ignore_eos=true`. Decode rate is `(completion_tokens - 1) / (last output timestamp - first output timestamp)`.
The c1 test used the same three coding prompts and 512-token output length as the
Ascend grouped-MTP c1 test. Results include five measured requests after one
32-token warmup.

| Workload | Result |
| --- | ---: |
| c1, five 512-token requests, median | 93.94 tok/s |
| c1, observed range | 93.16–105.21 tok/s |
| c3, three 3×1024-token batches, median aggregate | 206.24 tok/s |
| c3, observed aggregate range | 195.09–210.35 tok/s |

The c3 test used short merge-sort prompts. The llama.cpp service has three
slots, so c3 is not a matched comparison to the Ascend four-stream result.
Quantization and inference engines also differ. Raw timing and token counts
are in `c1.json` and `c3.json`.
