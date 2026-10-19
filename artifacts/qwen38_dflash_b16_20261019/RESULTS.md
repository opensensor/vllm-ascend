# Block-16 DFlash2 and longer chat answers — 2026-10-19

## What was tested

The target was the same Qwen3.8-27B `UD-Q8_K_XL` GGUF on the NVIDIA RTX PRO 6000 Blackwell as in the [earlier fixed-length sweep](../qwen38_dflash_energy_20261019/RESULTS.md). All four arms used the same local llama.cpp rebuild, one slot, 4096 context tokens, full GPU offload, flash attention, and Jinja chat formatting. The checkout's Git worktree metadata is broken, so this rebuilt binary reports `0.4.0-dev (build 0, commit unknown)`; the pre-rebuild binary in the earlier sweep reported build 10906 (`b0dcb8192b`). The temporary logit instrumentation was removed before this sweep.

The original [z-lab DFlash2 Q8_0 GGUF](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2-GGUF) has a trained block size of eight and therefore drafts at most seven new tokens. To test a larger window, the [0xBakeer block-16 fine-tune](https://huggingface.co/0xBakeer/TandemLLM-Qwen3.8-27B-DFlash2-b16) was downloaded and converted with this llama.cpp checkout to Q8_0 GGUF. The safetensors SHA-256 was `843d7698e043118863847a560b0375e93db253f00352282d20ff0d6b6816356a`, matching its model card; the converted GGUF SHA-256 is `4628aef9267139219f07db25b51d21fcf2c888b98224814a639ae6f0b574af15`. The [conversion log](convert.log) records 81 tensors and the 2.05 GB output. The converted model is stored at `/run/media/matteius/20TB-drive/models/dflash2-qwen38-27b-b16/dflash2-b16-q8_0.gguf`.

The server was launched with `--spec-draft-n-max 16`; its [log](b16-k16-server.log) confirmed `block_size=16` and **clamped the effective maximum to 15 draft tokens**. A second run requested 12 with the same block-16 weights. The earlier block-8 Q8_0 draft was run at seven. Plain decoding used the identical target without a draft.

The [harness](../../tools/dflash_research/long_chat_bench.py) sent seven chat-formatted tasks: code, PostgreSQL, math, queue debugging, a story, translation, and service capacity. It disabled thinking with `chat_template_kwargs={"enable_thinking": false}`, used greedy decoding and seed 42, disabled prompt caching, allowed up to **1,536 output tokens**, and let EOS stop naturally. Each arm had one warmup and one measured response per task. All 28 measured responses stopped normally; none reached the token cap or exposed `<think>` text.

## Performance

| Arm | Output tokens | Aggregate decode tok/s | Speedup vs plain | End-to-end output tok/s | GPU-board J/output token | Exact text matches vs plain |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Plain | 2,866 | 46.87 | 1.00× | 44.87 | 11.63 | 7/7 |
| Block 8, K7 | 2,924 | 129.20 | 2.76× | 114.47 | 4.44 | 2/7 |
| Block 16, K12 | 2,846 | 115.37 | 2.46× | 102.96 | 3.91 | 1/7 |
| Block 16, K15 | 2,846 | 112.66 | 2.40× | 100.95 | 3.90 | 1/7 |

The block-16 K12 and K15 arms produced **identical text on all seven tasks**. The extra three draft positions did not improve aggregate speed in this one run. Draft acceptance over the seven measured tasks was 2,177/5,222 (41.7%) for block-8 K7, 2,141/8,424 (25.4%) for block-16 K12, and 2,141/10,530 (20.3%) for block-16 K15. Longer draft windows increased attempted positions much more than accepted tokens on these prompts.

GPU energy was integrated from NVML board-power samples during each HTTP request, including prefill and response overhead, excluding model loading and warmup. It does not include CPU or wall power. The answers have different lengths and sometimes different content across arms, so the GPU J/token comparison is descriptive, not a controlled energy saving attributable only to draft width. The block-8 and block-16 checkpoints also have different weights; only K12 versus K15 holds the draft checkpoint fixed.

| Task | Plain tok/s | Block-8 K7 tok/s | Block-16 K12 tok/s | Block-16 K15 tok/s |
| --- | ---: | ---: | ---: | ---: |
| Code | 46.7 | 224.5 | 237.7 | 256.8 |
| SQL | 46.7 | 121.3 | 117.9 | 114.3 |
| Math | 46.7 | 183.0 | 196.8 | 194.7 |
| Debugging | 46.7 | 103.0 | 86.1 | 83.9 |
| Story | 46.7 | 94.1 | 78.0 | 75.6 |
| Translation | 46.8 | 116.2 | 99.1 | 92.6 |
| Systems | 46.9 | 161.8 | 152.4 | 148.9 |

The code answer is identical across all four arms, and K15 reaches 256.8 decode tokens/s there, versus 46.7 plain. On debugging and the story, block-16 K15 is slower than block-8 K7. More draft positions therefore did not provide a general speed gain here.

Raw responses, timings, integrated energy, and power-sample counts are in [plain.json](plain.json), [b8-k7.json](b8-k7.json), [b16-k12.json](b16-k12.json), and [b16-k15.json](b16-k15.json). Draft acceptance and loading details are in their corresponding server logs.

## Answer review

All four arms delivered complete, readable final answers in this chat setup. The LRU code is identical and passed basic capacity, recency, and eviction checks. All arms gave the same core seven-calendar-day PostgreSQL window query, the collision formula and birthday approximation, and the 100 ms / 10 RPS / 50 RPS / five-in-flight service calculation. The stories all had dialogue and an ending, though all exceeded the requested 250–350 words (431–476 words). The block-16 translation has a heading typo, `Revisó vs. Revisó`, but correctly explains `examinó` in the body. The queue-shutdown answers are plausible descriptions, but none specifies producer admission and close as one atomic protocol, so they should not be treated as verified implementations.

This is an informal manual review of seven tasks, not a blinded quality evaluation. It supports the narrower observation that changed tokens often yielded coherent, task-relevant complete answers. It does not establish semantic equivalence or statistical quality parity.

The previous Flash-Next server was restored after the sweep; [its log](restored-server.log) and `/health` check confirm it is serving again. No Ascend hardware was used in this experiment.
