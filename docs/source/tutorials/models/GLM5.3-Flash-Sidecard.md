# GLM-5.3 Flash on CUDA and one Ascend sidecard

## Introduction

This experimental runtime joins one RTX PRO 6000 Blackwell with one Atlas
300I Duo card (two 310P3 chips). It keeps the complete 45-layer GLM text model
resident across devices: CUDA layers 0–17 and 33–44, Ascend layers 18–32.
The second Ascend card remains available to another workload.

The authenticated local checkpoint is
`/srv/ai/storage/models/GLM-5.3-Flash-one-card-sidecard-candidate`.
Native W2/W3/W4 expert codes and FP16 scales are preserved. A4 expert
activations, FP16 projections, and BF16-rounded residuals in FP32 storage
are explicit parts of this runtime's numerical policy. It does not establish
parity with the original unquantized model.

See the [design PNG](../../../../artifacts/sidecard-layer-optimization-20261012/architecture.png)
and [measured report](../../../../artifacts/sidecard-layer-optimization-20261012/README.md).

## Supported features

The Ascend middle block uses TP2 attention/shared MLP and EP2 experts, with
all 288 experts retained per layer. Text chat/completion, streaming, greedy
sampling, sparse DSA, owner-local recurrent/KV state, and bounded prefill
overlap are implemented. Requests serialize; the RTX executes prefix and
suffix in order while overlapping known prompt chunks with Ascend.

This is an independent actor server. Optional one-token KDA span graphs
retain eager DSA cache control and eager multi-token prefill. CUDA expert
kernels can run on the current Torch stream without per-layer barriers;
Ascend linear weights use a prepared FRACTAL_NZ layout. See the
[optimization report](../../../../artifacts/sidecard-layer-optimization-20261012/README.md)
for measured candidates and the promoted configuration. flashcomm1, MTP,
vision, tools, batched scheduling, and prefix caching remain outside scope. The model's theoretical position range is
1,048,576; the deployed cap and completed capacity gates are documented
separately in the report. A 128K/16-sequence profile is not validated.

## Environment preparation

Use separate CUDA and NPU Python environments on the qualified 310P host.
CUDA uses Torch 2.14.1+cu130; NPU uses Torch/torch-npu 2.13 and CANN 9.1.0.
Transformers remains 5.17.0 in both environments. The model-specific 310P
native kernels and CUDA SM120 compact expert library are required. Standard
A2/A3 container recipes do not describe this mixed host runtime.

The [runbook](../../../../tools/sidecard/LAYER_RUNTIME.md) gives exact host
paths, authenticated kernel build steps, and source isolation. It also
explains the private NPU environment file and the Linux parent-death guard.
Do not change the other card's device mapping or reset it.

## Deployment

A single `vllm serve` process cannot own this pair of architecture-specific
Torch environments. Run from a checkout using the dedicated launcher:

```bash
/srv/ai/venvs/sidecard-cuda-20261012/bin/python \
  -m tools.sidecard.layer_runtime \
  --specification artifacts/sidecard-layer-optimization-20261012/runtime-specification.json \
  --cuda-python /srv/ai/venvs/sidecard-cuda-20261012/bin/python \
  --output /srv/ai/storage/artifacts/sidecard-layer-new \
  --port 8000
```

Use a new output directory and free loopback rendezvous port. Startup
qualifies every middle expert layer, TP2 projections, complete-model
pipeline parity, and real generation before exposing readiness.

## Functional verification

```bash
curl -fsS http://192.168.53.187:8000/health
curl -fsS http://192.168.53.187:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"glm-sidecard","messages":[{"role":"user","content":"Calculate 7 * 8. Reply with only the number."}],"reasoning_effort":"low","max_tokens":64,"temperature":0}'
```

The answer is in `choices[0].message.content`; generated reasoning is
separate in `reasoning_content`. Prompt plus requested output must fit the
server's total context budget. Streams support final usage and `[DONE]`.

## Accuracy evaluation

The [test configuration](../../../../tests/e2e/models/configs/GLM5.3-Flash-Sidecard.yaml)
is for the custom API gate, not the standard lm-eval harness:

```bash
python -m tools.sidecard.layer_validate \
  --evaluation-config tests/e2e/models/configs/GLM5.3-Flash-Sidecard.yaml \
  --output /tmp/sidecard-api-gate.json
```

It runs five elementary arithmetic prompts, three fixed-length greedy
repeats, a completed 2K/64-token benchmark, and admission rejection checks.
Real-weight dense/TP2 components, clipped routed kernels, and sparse index
selection also have independent CPU-reference gates. See the report for
receipts and exact tolerance scopes. Arithmetic smoke and capacity tests do
not establish general model accuracy or long-context recall quality.

## Performance

The hardware report contains completed cold-prompt and decode measurements,
complete-model serial/overlap timings, and context-capacity receipts. It
separates internal first-token time from client first-text latency and
records the output token count. Decode timing excludes prompt processing.
Overlapped middle roundtrip measurements include overlapped CUDA work;
their medians must not be summed to estimate prefill wall time.
