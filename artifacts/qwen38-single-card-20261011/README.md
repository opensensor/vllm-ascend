# Qwen3.8 Flash-Next: one-card 128K qualification

Runbook: [clone, launch and native activation](../../docs/source/developer_guide/performance_and_debug/qwen38_single_card.md).
Frozen runtime: [release assets](https://github.com/opensensor/vllm-ascend/releases/tag/qwen38-single-card-20261011).

Selected hardware: one Ascend 310P3 dual-chip card (TP2/EP, chips 0 and 1),
247 GiB host RAM. Original W4 G128 checkpoint, FP16 scales; no new dense or
expert quantization. Text-only 131,072 total tokens, 512-token chunks, one
sequence, MTP2, seven FP32 recurrent slots, sequential QSA gathers.
Automatic memory profiling admitted 134,614 cache tokens.

The [qualification receipt](qualification.json) includes source hashes,
physical memory accounting, native-worker gates, comparisons and limitations.
Raw near-limit and decode measurements are retained alongside it.

| Selected measurement | Result |
| --- | --- |
| Cold 8,192-token prefill | 28.591 s TTFT; 286.52 tokens/s |
| Identical 8K replay | 8,064 cached; 0.906 s TTFT |
| Median three 512-token generations | 20.94 tokens/s |
| Cold 131,008-token recall | Correct; 528.366 s TTFT; 247.95 tokens/s |
| Identical long replay | 130,944 cached; 4.932 s TTFT |
| 130,560 input + 512 output | Complete 131,072-token window; 19.73 generation tokens/s |

The changed/shortened full-window prompt had zero cached tokens; its original
cache-hit expectation failed despite complete correct capacity/generation
checks. Both results remain in the receipt. The final stage is
`passed-with-known-cache-miss`. Deterministic text/tool smoke is 6/7, retaining
the baseline reverse-word error. Reasoning arithmetic and 13 generated code
cases passed; no fresh broad accuracy evaluation is claimed.

The 32K parallel-gather short profile was faster (22.24 median decode), but its
128K long-prompt trial stalled after 6,144 tokens. The selected sequential
schedule completed two long cold prompts. Logging-pool warnings also occurred
in the progressing run; they alone do not establish the cause of the stall.

Offline validation: 128 scoped CPU tests passed; 28 bounded-scheduler tests
skip against the local upstream version and pass on the pinned host vLLM.
Two real AscendConfig admission tests passed. Both chips passed synthetic
96-step spill/restore/CoW/recycling gates with seven, eight and 16 slots.
Each worker passed 70 byte-exact native residual cases before graph recapture.
Scoped manual pre-commit checks pass. Whole-repository format CI has unrelated
archived-artifact and v2 lint failures recorded in the earlier work.

The relocated release snapshot itself ran the successful near-limit service.
The bundle excludes model weights, the Python environment and CANN/ATB; it
requires the recorded existing environment. A fresh OS install, vision,
two-session serving and 262K capacity remain unqualified.
