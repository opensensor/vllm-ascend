# 2.5× faster Qwen3.8-27B decoding and 58% less GPU energy/token with DFlash2. Maybe the bigger bottleneck is how we generate.

Every local LLM hardware discussion seems to circle back to NVIDIA generations, VRAM capacity, and memory bandwidth. Those matter. But I've been wondering how much of the pain comes from treating one target-model forward pass per new token as inevitable.

I tested a [DFlash2 draft](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2-GGUF) with the same [Qwen3.8-27B Q8 GGUF target](https://huggingface.co/adaliboy/Qwen3.8-27B-GGUF/blob/main/Qwen3.8-27B-UD-Q8_K_XL.gguf) on an RTX PRO 6000 Blackwell using llama.cpp. DFlash2 proposes a block of tokens, and the target verifies them in parallel. The [DFlash paper](https://arxiv.org/abs/2602.06036) explains the approach.

I ran ten varied prompts, 256 generated tokens each, one request at a time, greedy sampling, with the target and all other server settings held fixed. The decode rates below come from llama.cpp timings. GPU energy was integrated from NVML board-power samples during each request, so that number includes prefill and request overhead but excludes model loading; it is not wall-plug energy.

| Mode | Decode tokens/s | GPU joules/output token |
| --- | ---: | ---: |
| Plain target | 46.85 | 11.67 |
| DFlash2, up to 3 draft tokens | 101.68 | 5.59 |
| DFlash2, up to 7 draft tokens | 117.27 | 4.94 |

So the seven-token setup was **2.50× faster on decode** and used **about 58% less measured GPU energy per output token** on this workload. The gain varied a lot by prompt: roughly 80–180 decode tokens/s with DFlash2, depending on draft acceptance. An earlier three-prompt sweep got 2.83×; the broader result is the one I'd quote.

**Important correctness caveat:** this is not yet an exact greedy replacement in my setup. Only 3 of 10 full 256-token DFlash2 completions matched the plain target token for token. I traced the earliest mismatch: after the same first two output tokens, serial decode put ` compare` ahead of ` analyze` by 0.0140 raw logit, while the eight-output verifier batch put ` analyze` ahead by 0.0035. A shorter verifier batch kept the serial ranking. That points toward a batch-shape numerical issue, though I still need to rule out verifier state or position bookkeeping. I have not run a quality evaluation, so I'm not claiming the changed outputs are worse—or that this path is lossless.

The other reason I'm interested is hardware diversity. I'm also working with Ascend 310P cards on a Threadripper machine and this [Qwen3.8-27B W8A8 vLLM-Ascend checkpoint](https://huggingface.co/tsii42/Qwen3.8-27B-W8A8-Dynamic-Ascend-310P-vLLM). **The numbers above are entirely from NVIDIA and a GGUF target.** I have not measured DFlash2 on the Ascend cards yet. That's a separate port and validation job.

My take is that we are still early in practical LLM architecture and inference design. Autoregressive next-token prediction has good reasons behind it, but today's one-token-at-a-time serving path looks woefully inefficient. Faster memory and tensor cores help; algorithms that reduce serial target invocations may buy more. I want to spend more effort on that side of the problem across both NVIDIA and Ascend.

Has anyone else seen a quantized target flip a near-tied greedy token between serial decode and batched speculative verification? I'm especially interested in fixed-prefix tests that separate kernel numerics from verifier bookkeeping. I can share the benchmark harness and trace logs.
