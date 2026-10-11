# Qwen3.8 Flash Next: experimental 1M YaRN on six 310P chips

## Introduction

This profile extends the existing TP6 W4 checkpoint from 262,144 to 1,048,576
tokens with static YaRN factor 4. Both QSA attention and its learned indexer
rotate the same leading 64 dimensions. The 48-layer model retains its GDN,
PLE, routed experts, shared experts, and two-token MTP draft strategy.

The larger limit is experimental. Component parity at million-token positions
does not establish retrieval quality across a million-token conversation.
The user elected to test long histories interactively instead of waiting for
the progressive retrieval gate.

## Supported Features

| Feature | Profile |
| --- | --- |
| Hardware | Three Atlas 300I Duo cards, six Ascend 310P3 chips |
| Parallelism | TP6/EP6; unequal expert shards retain global router IDs |
| Context | 1,048,576 tokens per request, including generated tokens |
| Active requests | Two scheduler slots sharing one KV pool |
| Generation | MTP2, breakable FULL_DECODE_ONLY graphs with capture sizes 3 and 6 |
| Multimodal | One image per request; encoder data parallelism; existing image limit |
| Cache | Prefix caching, chunked prefill, and compact recurrent checkpoints |
| Thermal control | Hold at 94 C; resume when all six chips are at or below 85 C |
| FlashComm1 / asynchronous scheduling | Disabled in the retained profile |

Two slots do not guarantee two simultaneously full 1M contexts. Cached tokens
still count toward each request's context limit. Reused prefixes can share KV
blocks, but distinct histories compete for pool capacity.

## Environment Preparation

Use the pinned serving environment and the converted checkpoint already on the
development host. This is a 310P profile; A2/A3 container examples do not qualify
this hardware. Preserve the coherent custom OPP libraries and upstream vLLM
fork documented in the [runtime runbook](../../developer_guide/performance_and_debug/qwen38_310p_runtime_runbook.md).
Do not independently upgrade Transformers or change the shard iterator.

The complete runtime snapshot is
`/srv/ai/src/qwen-yarn-1m-runtime-20261011`. Its profile and receipts are in
`/srv/ai/src/qwen-yarn-1m-tp6-20261011`.

## Deployment

On the development host:

```bash
bash /srv/ai/src/qwen-yarn-1m-tp6-20261011/start-server.sh --check-runtime
bash /srv/ai/src/qwen-yarn-1m-tp6-20261011/start-server.sh
```

The profile supplies the nested `text_config.rope_parameters` override:

```json
{
  "rope_type": "yarn",
  "factor": 4.0,
  "original_max_position_embeddings": 262144,
  "rope_theta": 10000000,
  "partial_rotary_factor": 0.25,
  "mrope_interleaved": true,
  "mrope_section": [11, 11, 10]
}
```

It also sets the existing upstream `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1` override
because the checkpoint's native maximum remains 262,144. The override alone
does not implement YaRN. The plugin must apply the rotary policy in both QSA
attention and indexer paths.

Startup restores the qualified runtime modules and checks text, arithmetic,
code, fresh image processing, and repeated image processing before releasing
maintenance. The long retrieval benchmark is not part of automatic startup.

## Functional Verification

```bash
curl http://127.0.0.1:8001/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen38-flash-next","messages":[{"role":"user","content":"What is 17 times 19? Answer with the number only."}],"temperature":0,"max_tokens":32,"chat_template_kwargs":{"enable_thinking":false}}'
```

Read `server-8001.log`, `bootstrap.log`, and `bootstrap-ready.json` in the
profile directory. A successful HTTP health check alone does not establish
that the engine passed image processing or finished restoring its graphs.

## Accuracy Evaluation

| Gate | Evidence / scope |
| --- | --- |
| CPU YaRN formula | Exact parity with installed Transformers for the supported policy parameters |
| Million-position rotary math | All six chips; maximum absolute table error about 4.41e-6 |
| Text graph replay | Three changed-position replays per chip |
| Image rotary coordinates | Component parity on all six chips |
| Real checkpoint startup checks | Five of five passed: marker, math, code, fresh image, cached image |
| Full 1M retrieval / language quality | Not qualified; interactive testing requested |

The optional `tools/qwen4exp/validate_yarn_context.py` gate checks retrieval at
280K, 550K, and 1M synthetic background tokens, then verifies a cached repeat.
Run it only in a controlled maintenance window. It acquires maintenance and
leaves it active for the caller to release after inspecting the receipts.

## Performance

The implementation retains only 32 FP32 inverse frequencies per QSA layer;
it does not allocate million-position rotary tables. Three-part FP32 angle
reduction avoids the observed large-angle NPU trigonometric error without a
CPU transfer or FP64 computation.

QSA top-k gathers use compact INT32 indices on NPU, with an INT32 intermediate
before the index sort's FP32 cast. CPU selection retains INT64 indices. The
initial component probe measured about 1-6% lower selection time; that is not
a whole-model throughput claim.

Exact dense QSA paths pass only the page-table prefix covering their 2,048-token
selection budget. Passing the padded 1M table triggered a paged-attention MTE
burst error during the first image probe. Regression tests preserve all
addressed pages and compact multiple request rows without device readback.

The separate `native_int8_group_probe.cpp` is an arithmetic experiment, not a
serving backend. Exact INT32 G128 results and graph replay passed on chip 0.
Full-K gate/up/down projections also passed exact reference math on chip 5
using real checkpoint weights at rows 1, 3, 16, and 17. Quantization, routing,
weight storage, and real-model throughput require measurement before adoption.
The direct A8 packer plus projections also matched the native W4A8 operator
exactly on chip 5 at rows 1, 3, and 17. No performance measurement accompanied
those live-traffic correctness probes; the serving workers retain their W4 path.

Kilo compaction follows its own model limits. Configure its matching model
entry with context 1,048,576, input 1,000,000, and output 32,768, then reload
the client. vLLM handles request limits and KV allocation; it does not summarize
conversation history for Kilo.
