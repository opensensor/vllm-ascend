# Same-prefix target-logit trace — 2026-10-19

## Question

Does the Qwen3.8-27B target itself rank tokens differently at the first plain-versus-DFlash2 greedy divergence? This trace uses the `systems` prompt from the [energy sweep](RESULTS.md), whose first two output token IDs are `271, 1206` in both arms. The third output token differs.

## Method

The temporary [llama.cpp instrumentation patch](../../tools/dflash_research/llama_server_logit_trace.patch) logs the target's unsampled top-two raw logits at each of the first four output positions. It reads `llama_get_logits_ith` after sampling has synchronized the target context. The trace was built from the same local llama.cpp source and CUDA configuration as the benchmark runtime, then removed; the original source and `libllama-server-impl.so` were restored afterward.

Both decisive requests used the exact same 256-token `/completion` payload: the same prompt, `temperature=0`, `seed=42`, `cache_prompt=false`, `ignore_eos=true`, and `return_tokens=true`. The server settings were the same as the energy sweep. DFlash2 used `--spec-draft-n-max 7`. Neither decisive request used `n_probs`; the logging patch accessed the raw target logits directly. Responses are [plain](trace-k0-response-full.json) and [K7](trace-k7-response-full.json); raw server logs are [plain](trace-k0-full-server.log) and [K7](trace-k7-server.log).

## Result at output token 3

| Path | Actual prefix token IDs | Target batch outputs | First-ranked token and raw logit | Second-ranked token and raw logit | Chosen output token |
| --- | --- | ---: | --- | --- | ---: |
| Plain serial | `271, 1206` | 1 | `9150` (` compare`), 21.9720306 | `22903` (` analyze`), 21.9580307 | `9150` |
| DFlash2 verification | `271, 1206` | 8 | `22903` (` analyze`), 21.9426403 | `9150` (` compare`), 21.9391327 | `22903` |

The serial margin is **+0.014000** in favor of `9150`. The verifier margin is **+0.003508** in favor of `22903`. The raw target ranking therefore flips at the same output prefix. The saved response token arrays reproduce the earlier sweep's first ten tokens in each arm.

The K7 log also contains a `pos=3 idx=1` line from the *preceding* verification batch. That line scores a rejected draft continuation and is not the target distribution for the actual `271, 1206` prefix. The relevant K7 line is the later `pos=3 idx=0 batch_outputs=8` after token `1206` was emitted.

## Verification shape check

A four-token K7 request on the same prompt produced `271, 1206, 9150, 37972`, matching plain decoding. Its relevant verifier line was `pos=3 idx=0 batch_outputs=2`, with `9150` first at 21.9645195 and `22903` second at 21.9530315. The [short response](trace-k7-response-no-probs.json) and the same [K7 log](trace-k7-server.log) preserve that control. The shorter request changes the draft/verification shape, so it cannot replace the matched 256-token comparison.

This directly establishes a target-logit ranking change between serial decode and the eight-output verifier path on this local quantized GGUF and llama.cpp build. Batch-dependent GPU numerics are a plausible cause, especially because the two logits are close and the shorter verification shape retains the serial ranking. The trace does **not** independently exclude a position or state bookkeeping difference inside the speculative path. A fixed-prefix `llama_decode` test that constructs one-token and eight-token batches without a draft model would isolate that remaining question.

The user's Flash-Next server was restored and passed `/health` after the trace; see [restoration log](restored-after-trace-server.log).
