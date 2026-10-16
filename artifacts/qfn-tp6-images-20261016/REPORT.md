# TP6 image serving and Kilo alignment — October 16, 2026

## Change

The Kilo profile advertised image input, but the running TP6 launch explicitly
selected `--language-model-only` and allowed zero images. The checkpoint already
contained the vision frontend and 333 vision weight keys. No model implementation,
package, driver, or firmware change was necessary.

The canonical TP6 launcher now loads vision, permits one image per request,
keeps video disabled, and bounds preprocessing to 65,536–1,048,576 pixels.
The vision encoder uses data parallelism: its 16 attention heads cannot be
partitioned evenly across six ranks. The 48-layer, hidden-size-2560 language
model retains native W4 experts, MTP2, EP6, and full decode-only graphs.

Local `/home/matteius/.config/kilo/kilo.jsonc` now identifies three-card TP6,
with context/input/output limits 262144/229376/32768 and text+image input.
The existing provider ID, endpoint, credentials, sampling options, and unrelated
providers remain unchanged. The original file was backed up with mode 0600.
`kilo debug config` confirms the effective values in `kilo-profile.json`.
The remote host's unrelated llama.cpp Kilo profile was not changed.

## Runtime and capacity

- Host: `matteius@192.168.53.187`; API: `http://192.168.53.187:8001/v1`.
- Served model: `qwen38-flash-next`; real TP6 W4 checkpoint, no dummy weights.
- API PID: 1822369; run: `resident-imagesv1` beneath
  `/srv/ai/storage/artifacts/qfn-tp6-scaleout-20261016`.
- Server log: `<profile root>/resident-imagesv1/server.log`.
- Model configuration and runtime per-request maximum: 262,144 tokens;
  no YaRN extension. Six request slots, 2,048-token batch ceiling.
- Image-enabled planner: 1,634,964 tokens, 6.24 times the per-request limit.
  Original text-only planner: 1,710,425 tokens. Neither number proves six
  fully populated windows or long-context vision accuracy.
- Matched Python 3.12.13, CANN 9.1.0, torch 2.13.0+cpu,
  torch_npu 2.13.0rc1, transformers 5.17.0; no version changes.

The prior API and its workers exited after clean collective retirement.
The sealed 95.37 GiB shared PLE owner survived. Thermal monitoring and CPU
worker affinity remained enabled for the new run.

## Validation

The baseline image suite passed all seven requests before fast-path restoration:
text arithmetic, red/blue/red images, a 1024×1024 green image, OCR of `NPU-4729`,
and another text arithmetic request. Every request returned HTTP 200 and the
expected answer. The larger image produced 1,024 image tokens, confirming that
vision ran rather than treating the prompt as text only.

The full restoration runs the real 5,704-token prefix test followed by six
cached branches (5,632 cached tokens each), six concurrent 256-token decodes,
and exact native PLE/HC and early-draft/complete-prefix verification before
selecting fast mode. Shared-prefill PLE is restored through its separate
admission and shadow gates. Final receipts accompany this report.

The image suite is then repeated with retained optimizations active. An
additional streaming gate sends 1024×1024 red/blue/red images with the same
question, requires correct SSE answers and `[DONE]`, and requires nonzero KV
prefix reuse on the final red request. A two-image request must return HTTP
400, after which `/health` must still return 200.

Final results: baseline **7/7**, retained **7/7**, streaming **3/3**. The
streamed red/blue/red sequence returned the correct colors; the final red
request reused **1,024 cached tokens** and completed in **0.578 seconds**.
The changed blue image had zero cached prefix tokens, confirming this sample
did not alias the prior red image. Two images returned HTTP 400 and health
remained HTTP 200.

All six workers reported clean graphs, active custom collectives, native PLE
in fast mode after **191** checks per rank, and shared-prefill PLE after **4**
checks per rank with no remaining shadows or failures. Early draft and complete
prefix selection passed **848** comparisons per rank before fast mode. The
maximum observed temperature was **72°C**, with zero thermal holds.

Receipts: `image-smoke-baseline.json`, `image-smoke-retained.json`,
`streaming-cache.json`, `over-limit.json`, `overlap-retained.json`, and
`final-audit.json`. The latter two retain the relevant audit fields; complete
transaction records remain in the host run directory.

These are runtime smoke and cache-isolation checks, not a general vision
accuracy evaluation. The manual diagnostic YAML is not an lm-eval task.
No full 131K or 256K prompt was submitted. The user's six-slot profile takes
precedence over a generic 16-slot capacity baseline.

## Features and limits

| Feature | Status in this profile |
| --- | --- |
| Images | One image/request; real color, OCR, changed-image and replay checks |
| Video | Disabled, not tested |
| TP6 / EP6 | Retained; all six workers participate |
| ACL graphs | FULL_DECODE_ONLY, sizes 3/6/12/18 |
| MTP | Two draft tokens retained; target handles images |
| PLE | Shared host table, native decode gather/DMA, shared-prefill adapter |
| Collective | Qualified custom binomial SUM retained |
| FlashComm1 | Disabled in favor of the existing custom collective |
| Long-context vision | Not qualified |
| Six full 256K windows | Not qualified |

The draft model logs its existing text-only multimodal fallback; this does
not disable the target vision encoder. The real image responses and image
usage counters are the relevant evidence. No eager or TorchDynamo fallback
was needed, and no dummy-only pass is claimed.

## Reproduce

Use the matched runtime and fresh owner directory. The canonical launcher and
ordinary restore steps are in `tools/qwen4exp/TP6_SERVING_PROFILE.txt`.
After `restore.py`, copy the shared-prefill controller from
`qfn-tp6-mixed-20261016/control.py`, replacing only its old
`resident-v8/control.token` path with the new run's token path. Run
`shared-gate-v4.py`, then `shared-enable-v3.py --apply` through that controller.
Warm the adapter until all six ranks report at least four checks, no remaining
shadows, no failures, and clean graphs. Preserve the PLE table owner throughout.

Run the attached validation scripts with the matched Python/Pillow environment:

```bash
python image_smoke.py --url http://127.0.0.1:8001 --output /tmp/qfn-image-smoke-new
python streaming_cache_gate.py --url http://127.0.0.1:8001 --output /tmp/qfn-image-stream-new
```

Output directories must be new. The small fixtures are generated locally; no
external image download or data upload is needed.

## Local checks

The rendered-launcher regression passes and checks image bounds, data-parallel
vision, context, six ranks, MTP, graphs, and custom collective configuration.
Scoped manual pre-commit checks pass. `bash format.sh ci` also ran: it still
fails existing repository-wide Ruff, spelling, formatting, Markdown, and
forbidden-import checks. Its unrelated auto-edits were discarded in the
isolated worktree. Default unit collection is blocked by the existing missing
`vllm.third_party.flash_linear_attention` dependency; the independent launcher
check passes with its focused `--confcutdir`.

This configuration change does not establish a new text throughput record.
The small image response timings are latency observations, not a statistically
qualified benchmark or a broad multimodal accuracy score.
