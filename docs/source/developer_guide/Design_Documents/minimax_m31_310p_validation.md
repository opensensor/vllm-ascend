# MiniMax-M3.1-Flash: initial six-chip 310P validation

October 24, 2026. This is a conversion and component-validation milestone,
not a running M3.1 service or a full-decoder accuracy qualification.
The user authorized stopping existing NPU workloads and using all six chips.

## Source and memory

The source is [MiniMaxAI/MiniMax-M3.1-Flash](https://huggingface.co/MiniMaxAI/MiniMax-M3.1-Flash/tree/54e4e69dfc452b1e94a426c24a4fdeeaa8e3ad5e),
pinned to revision `54e4e69dfc452b1e94a426c24a4fdeeaa8e3ad5e`.
Its root index has 89,632 tensors in 48 shards, with 250,103,411,200 bytes
of tensor payload. The separate DSpark directory is excluded from this budget.

The header audit independently accounts for every root tensor, including
NVFP4 companions and MXFP8 scales. The proposed conversion contains 23,428
logical weights and **219.16 GiB** of payload. TP2, PP3 and stage-local EP2
retain the best contiguous partition of **21/20/19 decoder layers**.

| Stage | Weights per chip, GiB | 8K cache, GiB | Including provisional 3 GiB reserve |
| --- | ---: | ---: | ---: |
| 0 | 37.07 | 0.205 | 40.28 |
| 1 | 38.69 | 0.195 | 41.89 |
| 2 | 37.98 | 0.186 | 41.17 |

These are estimates from this checkpoint's headers, not measured full-model
allocations. They conservatively replicate vision on every rank. All 60
layers retain a full main KV cache and an index-K cache. At an assumed
45.6 GiB per chip and 95% utilization, the busiest stage has 1.43 GiB beyond
the provisional reserve. Load peaks and actual runtime workspaces still need
measurement. The 128K/concurrency-16 skill baseline remains infeasible: KV
alone would require 300 GiB across the six chips.

## Conversion implementation

`tools.minimax_m3.export_mixed_w4` reads local source shards in bounded row
chunks and reuses the existing signed block32 W4 MSE quantizer. NVFP4 uses
low-nibble-first E2M1 decoding, E4M3 block scales, and division by the stored
weight global scale. Input global scales are validated and retained in source
provenance; they are not weight multipliers in the FP16-activation target.
MXFP8 uses E4M3 weights and raw E8M0 exponent bytes in groups of 32.

Dense quantized matrices become FP16; existing F32 tensors stay F32. The
completed config removes the source quantization configuration and carries
the existing W4 layout marker. The original config and index are preserved.
The production manifest reader checks normalized logical weight coverage
against the original mixed source index and rejects missing companions.
Partial exports do not publish a complete manifest. Resume checks output
hashes and source identity. Per-record source hashes for quantized matrices
refer to decoded FP32 rows, as explicitly recorded in the manifest.

A real-weight fixture converted layer 3 expert 0's three full projections
and its full attention Q projection. The production manifest reader accepted
the fixture, and expert codes/scales matched the independent preparation
used by the device probe exactly. This reduced fixture is not a complete model.

## Hardware evidence

The host is `threadripper` (`192.168.53.187`). Device pairs `[0,1]`, `[2,3]`
and `[4,5]` share PCIe identities, confirming the three dual-chip cards.
The three stopped system services are `qwen3-vl-model`, `qwen3-vl-card2` and
`qwen3-vl-thinking`. Their unit definitions are saved on that host under
`/srv/ai/storage/artifacts/minimax-m31-20261024/previous-services/`.
They remain stopped for M3.1 work; no existing model files were removed.

The attention probe uses **real layer 3 weights with seeded synthetic hidden
states**, FP16 projections, Gemma norms, partial rotary positions, sigmoid
routing with selection bias, and the existing eager paged sparse-attention
implementation. It compares CPU and NPU projections, routing, selected block
sets, attention, and output projection. A separate attention run supplies
identical CPU-generated projections to isolate attention arithmetic.

The 8,193-token test has three queries following an 8,190-token prefix,
crosses a 128-token page boundary, uses different main/index physical page
maps, and poisons unused cache slots with NaNs. All six chips passed using
Torch 2.13 and CANN 9.1. Selected block sets and router IDs matched exactly.
Attention NRMSE was approximately **0.000379** and output-projection NRMSE
**0.000445**; the declared output tolerance was 0.01. The shorter 2,177-token
probe also passed. These checks do not cover every layer, realistic hidden
state distributions, full prefill scheduling, or end-to-end decoding.

The expert probe uses real converted layer 3 expert 0 repeated across 64
slots, exercising `[64,3072,3072]` packed gate/up banks and
`[64,6144,1536]` packed down banks. The grouped operator passed on chips 0
and 2 with two routes per expert, all 128 routes in the last expert, and
zero local routes. Maximum projection NRMSE was **0.0000145** against the
CPU reconstruction of the stored W4 format, below the declared 0.003 limit.
Repeated slots test production geometry; they do not qualify all 128 real
experts or stage-local collectives.

Requantizing the NVFP4 source adds weight reconstruction error: **11.52–11.77%**
NRMSE for the sampled w1/w2/w3 matrices. Small device error relative to W4
does not establish model quality relative to the released checkpoint.
The composed CPU W4 expert output differs from the reconstructed source
expert by **17.85% NRMSE** on four seeded inputs (cosine 0.9840). This is
not a language-quality score. Both expert references use FP16 activations;
they do not reproduce the release's NVFP4 activation quantizer.
The device clipped-SwiGLU check with identical
inputs also passed. Calibration and held-out language/multimodal evaluation
remain required.

Two environment failures were resolved before accepting expert results:

- The base Torch 2.10 interpreter could not load the Torch 2.13 custom
  binding. Use the matching isolated venv.
- Supplying tensors on chip 2 while leaving the current device at chip 0
  produced zero custom-op output. Explicit `torch.npu.set_device(device)`
  fixed the probe, confirmed on chips 0 and 2. Workers must establish their
  current device before invoking this binding.

The matching OPP package must supply both the grouped kernel and op-api
library. Merely finding the Torch registration is insufficient. Built-in
attention operators emitted a storage-offset warning, although the bounded
comparisons passed; this is not a blanket qualification of other view shapes.

## Reproduction and continuation

Local source and output directories are
`/run/media/matteius/20TB-drive/models/MiniMax-M3.1-Flash` and the sibling
`MiniMax-M3.1-Flash-W4`. The preparation supervisor in
`artifacts/minimax-m31-20261024/prepare_pipeline.py` finishes the pinned
download, runs conversion, and checks the completed manifest/budget. It
records progress in `pipeline-status.json` and errors in its logs. It does
not start a server. Check that status rather than assuming completion.

```bash
python -m tools.minimax_m3.mixed_checkpoint /path/to/MiniMax-M3.1-Flash \
  --cache /path/to/range-cache --output /tmp/m31-budget.json

python -m tools.minimax_m3.export_mixed_w4 /path/to/MiniMax-M3.1-Flash \
  /path/to/MiniMax-M3.1-Flash-W4 --threads 16 --mse-backend numba

python -m pytest --noconftest tests/ut/minimax_m3 -q
```

To prepare the attention bundle, run `tools.minimax_m3.layer_probe` with
`BUNDLE --prepare --model SOURCE --cache RANGE_CACHE`. Prepare the expert
bundle with the same arguments to `tools.minimax_m3.expert_probe`.
Both default to layer 3; the expert probe defaults to expert 0.

On the NPU host, source `/srv/ai/bin/ascend-env.sh` and use
`/srv/ai/venvs/qwen38-w4-test-ce1862/bin/python`. Run the attention probe with
`BUNDLE --ops PATH_TO_MINIMAX_M3_OPS --device npu:0 --length 8193 --output RESULT`.
For the expert probe, additionally pass `--binding` with
`/srv/ai/src/qwen-quality-repair-tp6-v9-20261010/vllm_ascend/vllm_ascend_C.cpython-312-x86_64-linux-gnu.so`.
Its resolved binding SHA-256 is
`7dc95bea62a7b2e2f245af8993e87ade02b69a029a0c0f39b714dedcc9061275`.

For that expert probe, set `ASCEND_CUSTOM_OPP_PATH` to
`/srv/ai/storage/artifacts/npu-collective-qfn-20261014-024609/qfn-runtime/vllm_ascend/_cann_ops_custom/vendors/custom_transformer`
and prepend its `op_api/lib` directory to `LD_LIBRARY_PATH`. This is an
isolated probe configuration, not a change to any existing service launcher.

After conversion, the remaining gates are full-bank loading, all-layer
composition, TP2 reductions and PP3 transport, then a real-weight text service
at 8K/one request with measured per-rank load and cache peaks. Require
`/v1/models`, a short generation and a prompt longer than one sparse block,
followed by held-out quality evaluation. Full service, full-model accuracy,
TTFT and decode throughput are **not yet validated**. EP collectives,
ACLGraph, FlashComm1, DSpark/MTP and multimodal API validation are pending;
the first service candidate remains eager and text-only.

Evidence is under `artifacts/minimax-m31-20261024/`: header budget,
per-chip attention JSON, grouped expert JSON, conversion fixture receipt,
unit-test output, and runtime failure logs. Large weight fixtures and downloaded
checkpoints are local artifacts and are not committed. The focused CPU suite
passed **102 tests**, with four optional ModelSlim tests skipped. This is not
a full-repository or full-model test-suite result.
