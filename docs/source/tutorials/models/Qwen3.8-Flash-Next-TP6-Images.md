# Qwen3.8 Flash-Next: TP6 images on 310P

## Introduction

This profile enables image input on the existing three-card QFN service at
`192.168.53.187:8001`. The real checkpoint is
`/srv/ai/models/Qwen3.8-Flash-Next-W4A16-G128-TP6-MTP-20261010`.
Its `Qwen4ExpForConditionalGeneration` architecture combines the Qwen3-VL
vision frontend with the existing W4 language model. This is a host-specific
serving configuration, not an A2/A3 installation recipe.

## Supported Features

The configured profile retains TP6/EP6, MTP2, full decode-only graphs,
prefix caching, paced prefill, native PLE, and the qualified custom collective.
The vision encoder uses data parallelism because its 16 heads do not divide
across six TP ranks. Images are limited to one per request, with processor
bounds of 65,536 to 1,048,576 pixels. Video and FlashComm1 remain disabled.
The runtime accepts 262,144 tokens per request and six request slots; this
is not a certification of six fully populated context windows.

## Environment Preparation

Use the existing matched Python 3.12.13, CANN 9.1.0, torch 2.13.0+cpu,
torch_npu 2.13.0rc1 and transformers 5.17.0 environment. Preserve the qualified
operator-package precedence and the shared PLE table owner across restarts.
Do not replace dependencies or use A2/A3 container defaults for these 310P3s.
The [runtime runbook](../../developer_guide/performance_and_debug/qwen38_310p_runtime_runbook.md)
describes the complete runtime and ownership constraints.

## Deployment

The canonical command is rendered by
`artifacts/qfn-tp6-scaleout-20261016/start-tp6.sh`. Compared with the previous
text-only profile, remove `--language-model-only` and set:

```bash
--mm-encoder-tp-mode data \
--limit-mm-per-prompt '{"image":1,"video":0}' \
--mm-processor-kwargs '{"min_pixels":65536,"max_pixels":1048576}'
```

On the qualified host, use a fresh run directory:

```bash
cd /srv/ai/storage/artifacts/qfn-tp6-scaleout-20261016
python3 launch.py NEW_VERSION
# After readiness, restore and verify the retained owners once:
bash run.sh restore.py resident-NEW_VERSION
```

The shared-prefill PLE adapter needs its additional gate and enable transaction;
see the [image validation report](../../../../artifacts/qfn-tp6-images-20261016/REPORT.md).
A launcher-only restart does not restore every retained optimization.
No eager or TorchDynamo fallback is prescribed for this image profile.

## Functional Verification

Use the real-image smoke runner from the validation artifact, with a fresh
output directory and the actual API URL. It checks text arithmetic, changed
images, a repeated image, the maximum configured image resolution, and OCR.

An OpenAI-compatible request carries image data in `image_url`:

```bash
curl --fail http://192.168.53.187:8001/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen38-flash-next","messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"data:image/png;base64,REPLACE_WITH_PNG_BASE64"}},{"type":"text","text":"Describe this image."}]}],"max_tokens":128,"chat_template_kwargs":{"enable_thinking":false}}'
```

Local `~/.config/kilo/kilo.jsonc` should keep image input enabled and advertise
context/input/output limits of 262144/229376/32768. Its QFN provider remains
`vllm-310p-w4` for compatibility; the display label identifies TP6.

## Accuracy Evaluation

The manual runtime diagnostic records small real-weight smoke checks. It is
not an lm-eval task or a general vision accuracy score. No dummy-only result
qualifies this runtime. Long-context image retrieval, multiple images per
request, and video are outside this validation.

## Performance

Vision adds encoder execution and changes profiled cache headroom. Historical
text-only throughput and cache capacity must not be presented as measurements
of this image-enabled run. See the validation report for current evidence.
