# Qwen3.8-27B DFlash2 energy and correctness sweep — 2026-10-19

## Setup

This follows the [initial RTX experiment](../qwen38_dflash2_20261019/RESULTS.md). The same NVIDIA RTX PRO 6000 Blackwell GPU, Qwen3.8-27B `UD-Q8_K_XL` target GGUF, DFlash2 `Q8_0` draft GGUF, and llama.cpp build 10906 (`b0dcb8192b`) were used. The tested arms were plain decoding (K0), DFlash2 with at most three draft tokens (K3), and DFlash2 with at most seven draft tokens (K7). Each arm used one slot, 4096 context tokens, full GPU offload, and flash attention.

The fixed workload in [energy_bench.py](../../tools/dflash_research/energy_bench.py) has ten varied raw-completion prompts. Each arm had one 128-token warmup, then one 256-token greedy completion per prompt with seed 42, prompt caching disabled, EOS ignored, and raw output token IDs requested. Measurements were made sequentially; K0 was repeated after K7 and before K3, and that second K0 is the comparison baseline below.

GPU power was sampled through NVML approximately every 50 ms during each HTTP request and integrated with trapezoids. Energy includes prefill, decode, and request overhead, but excludes the warmup and model loading. It is GPU-board energy only; CPU, storage, and wall-power use were not measured.

## Matched results

| Arm | Decode tokens/s | Speedup vs K0 | GPU joules/output token | Mean observed GPU power | Exact 256-token matches vs K0 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Plain K0 | 46.85 | 1.00× | 11.67 | 524 W | 10/10 |
| DFlash2 K3 | 101.68 | 2.17× | 5.59 | 515 W | 2/10 |
| DFlash2 K7 | 117.27 | 2.50× | 4.94 | 547 W | 3/10 |

Each arm generated 2,560 measured output tokens. Total measured GPU energy was 29.88 kJ for K0, 14.32 kJ for K3, and 12.66 kJ for K7. K7 used about **58% less GPU energy per output token** than K0 in this run. The speed gain was larger than the increase in power draw.

Per-prompt decode rates and energy are in [k0.json](k0.json), [k3.json](k3.json), and [k7.json](k7.json). The server logs contain draft-acceptance measurements: [K3](k3-server.log), [K7](k7-server.log).

Each cell below is decode tokens/s followed by GPU joules/output token. Rates ranged from 79.9 to 179.8 tokens/s for K7, depending on the prompt.

| Prompt | K0 rate / J | K3 rate / J | K7 rate / J | K7 draft acceptance | K7 exact match |
| --- | ---: | ---: | ---: | ---: | :---: |
| Speculation | 46.7 / 11.3 | 96.6 / 5.7 | 95.6 / 5.9 | 27.2% | No |
| Python | 46.7 / 11.4 | 125.0 / 4.5 | 179.8 / 3.2 | 65.9% | Yes |
| SQL | 46.8 / 11.5 | 114.0 / 5.0 | 135.6 / 4.2 | 45.1% | No |
| Debugging | 46.5 / 11.6 | 74.1 / 7.6 | 79.9 / 7.2 | 20.5% | No |
| Math | 46.7 / 11.7 | 114.1 / 4.9 | 159.7 / 3.6 | 56.1% | No |
| Systems | 46.6 / 11.8 | 117.3 / 4.8 | 125.0 / 4.6 | 41.3% | No |
| Translation | 46.7 / 11.8 | 117.2 / 4.9 | 145.9 / 3.9 | 50.4% | Yes |
| Story | 46.7 / 11.9 | 91.4 / 6.3 | 99.9 / 5.8 | 29.7% | No |
| Summary | 46.7 / 11.9 | 98.4 / 5.8 | 109.2 / 5.3 | 33.6% | Yes |
| Formatting | 46.6 / 12.0 | 89.3 / 6.5 | 106.1 / 5.5 | 32.5% | No |

## Correctness finding

With raw token IDs requested, only three K7 completions and two K3 completions matched plain greedy decoding exactly. K7 repeated three times with identical token IDs for the traced debugging prompt. For the seven K7 prompts that differed, the first divergence occurred at output positions 2–248. Replaying each baseline prefix with top-token probabilities reproduced the baseline choice. At those first divergence points, the baseline top-two log-probability gaps were only 0.0017–0.0515; details are in [divergence-top2.json](divergence-top2.json).

The follow-up [same-prefix logit trace](LOGIT_TRACE.md) captured one of these divergences directly. At output token three of the systems prompt, plain decoding ranked `9150` first by 0.0140 raw logit, while an eight-output verifier batch ranked `22903` first by 0.0035. A shorter verifier batch retained the plain ranking. This establishes the target ranking flip in this runtime; batch-dependent numerics are plausible, but the trace does not independently rule out verifier position or state bookkeeping. Do **not** claim bit-identical or distribution-identical output for this local quantized-target runtime on the basis of these measurements. The first three-prompt experiment established identical text on its nine cases; it did not request token IDs, and its report has been corrected.

A [manual review of the saved text](QUALITATIVE_REVIEW.md) found that most divergent passages remain readable and often carry similar ideas, but the forced 256-token cap makes many answers incomplete. The K7 SQL answer introduces an unsupported SQL Server `RANGE` claim, and its math derivation contains malformed notation. The saved outputs do not establish semantic equivalence.

## Next engineering gate

1. Run a fixed-prefix target-only `llama_decode` probe in one-token and eight-token batch shapes. This will separate batch-dependent kernel numerics from speculative state or position bookkeeping.
2. Add a deterministic greedy fallback for uncertain verification positions only if the divergence is numerical, then remeasure energy and acceptance. Preserve the current fast path for high-margin positions.
3. Re-run quality and energy gates with chat-formatted prompts, natural EOS handling, longer contexts, and concurrency before making deployment claims.
4. Port the draft and selector path to the 310P runtime separately; these NVIDIA measurements do not validate the Ascend W8A8 checkpoint.

The prior Flash-Next server was restored after this sweep; its startup log is [restored-server.log](restored-server.log).
