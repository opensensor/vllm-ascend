# Qwen3.8-27B DFlash2 RTX experiment — 2026-10-19

## Setup

- GPU: NVIDIA RTX PRO 6000 Blackwell Workstation Edition, 96 GiB class.
- Runtime: local llama.cpp build 10906, commit `b0dcb8192b`.
- Target: local `Qwen3.8-27B-UD-Q8_K_XL.gguf` from the base Qwen3.8-27B GGUF directory, revision `fdd03b8bbd279c1694563650e79d85a237d9934`.
- Draft: `z-lab/Qwen3.8-27B-DFlash2-GGUF` at revision `2d9571f8ce46e151f61c6499c99dee6079e1d610`, `Q8_0` file.
- One slot, 4096-token context, full GPU offload, flash attention on, draft maximum 7 tokens (8-token block).
- The original 27B model was held fixed between arms; the only generation change was enabling the DFlash2 draft.

## Method

Run the fixed prompts in [bench.py](bench.py) against `/completion`. For each of three prompts, make one warmup request and three measured requests. Use greedy decoding, seed 42, disable prompt caching, ignore EOS, and generate exactly 128 tokens. Compare both token IDs and text. The reported rate is `tokens_predicted / predicted_ms` from the server's decode timings, aggregated over the nine measured requests. The warmups are excluded.

## Results

| Prompt | Plain tok/s | DFlash2 tok/s | Speedup | Draft acceptance | Mean accepted length |
| --- | ---: | ---: | ---: | ---: | ---: |
| Explanation | 46.82 | 97.21 | 2.08× | 29.1% | 2.95 |
| Code | 46.77 | 226.36 | 4.84× | 90.1% | 7.06 |
| Analysis | 46.83 | 126.35 | 2.70× | 41.8% | 3.85 |
| All 1,152 output tokens | 47.18 | 133.68 | 2.83× | — | — |

All nine paired completions had **identical token IDs and text**. The measured decode times were 24.420 s plain and 8.618 s with DFlash2. GPU memory use after loading was about 29.3 GiB plain and 33.2 GiB with the draft.

## Reproduction

Use the same target for both runs. The DFlash2 run adds:

```text
--spec-draft-model /run/media/matteius/ai-drive/models/gguf/Qwen3.8-27B-DFlash2/Qwen3.8-27B-DFlash2-Q8_0.gguf
--spec-type draft-dflash --spec-draft-n-max 7 --spec-draft-ngl all
```

For either run, start `llama-server` on port 8199 with `--ctx-size 4096 --n-gpu-layers all --flash-attn on --parallel 1`, then run:

```bash
python artifacts/qwen38_dflash2_20261019/bench.py --out artifacts/qwen38_dflash2_20261019/RESULT.json
```

Raw responses and server logs: [baseline.json](baseline.json), [dflash2.json](dflash2.json), [baseline-server.log](baseline-server.log), [dflash2-server.log](dflash2-server.log).

## Scope

This is a greedy, single-request microbenchmark with three short prompts. `ignore_eos` keeps every completion at 128 tokens for a comparable decode measurement; it is not a quality evaluation. The NVIDIA run used a base-model GGUF target, **not** the `tsii42` Ascend W8A8 checkpoint, which requires the Ascend quantization runtime. The results establish DFlash2 behavior on this RTX setup; 310P compatibility, output quality across tasks, and speed under longer contexts or concurrency remain unmeasured.
