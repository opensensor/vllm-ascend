# Qwen W4 prefill SwiGLU package staging, October 4

The fused SwiGLU-plus-INT4-pack candidate is built and staged for an isolated
310P gate. No NPU inference, operator benchmark, or Qwen service was started
in this handoff. NPU use remains deferred at the user's request.

## Staged artifacts on Threadripper

| Item | Path or SHA-256 |
| --- | --- |
| Isolated build source | `/srv/ai/src/qwen38-prefill-swiglu-src-20261004` |
| Isolated runtime | `/srv/ai/src/qwen38-prefill-swiglu-runtime-20261004` |
| Coherent OPP vendor | `/srv/ai/src/qwen38-prefill-swiglu-opp-20261004/vendors/qwen38_swiglu_prefill_transformer` |
| Five-operator installer | `725e5bfe346acf4c35e42919762c85159d90bf9c25f1bf21fa4ba7af0cdae63c` |
| Runtime host extension | `b23c5f3b7da77daa75d9664482c847a01c2873b6d33d1acde22d3c919636b229` |
| Coherent host API library | `f1a9515baa78ccf4318415ec4bad710d79ad6390515d834ff74217b3e5ea89c6` |

`results/build-bindings.log`, `results/build-five.log`, and
`results/install-five.log` are under the isolated build source. The first
build produced the host extension and four native W4 operators. The second
produced a coherent OPP with those four operators and
`RecurrentGatedDeltaRuleV310`. The latter was installed into the isolated
OPP path above. The runtime launcher enables
`grouped_activation=cann_swiglu_pack`; its only change against the retained
batch-1536 launcher is in [candidate-launcher.diff](candidate-launcher.diff).

## Host checks completed

- The installed host API exports both workspace-size and execution symbols
  for all four W4 operators and the recurrent operator.
- The recurrent kernel configuration includes FP16 and FP32 state variants.
- The runtime's host extension includes the SwiGLU-pack Torch registration;
  its SHA-256 matches the isolated build output.
- `ldd` resolves the host API and extension dependencies after adding the
  selected CANN, Torch, and torch-npu library paths.
- The isolated runtime's `w4_moe.py` and the SwiGLU adapter and tiler match
  commit `a4f42f565` by SHA-256. The built native-matmul source uses the
  selected 128-row-per-expert switch, and its grouped chunk is 1,536 tokens.
- `bash -n` and the launcher's `--check-runtime` passed. `--show` selects TP4,
  a 2,048-token scheduler batch, the fixed cache, MTP2, decode graphs, and
  the opt-in activation setting. These checks do not establish NPU parity.
- The one-layer benchmark passed `py_compile`, Ruff, and its CPU-only
  `--dry-run` path.

## Queued NPU gate

Use the isolated runtime and the coherent OPP above, with the OPP first in
`ASCEND_CUSTOM_OPP_PATH` and `LD_LIBRARY_PATH` as required by the
[310P runtime runbook](../../docs/source/developer_guide/performance_and_debug/qwen38_310p_runtime_runbook.md).
Run the 5,120-, 15,360-, and 20,480-row
`benchmark_w4_prefill_swiglu_pack_310.py` exact-parity/timing check first.
If it passes, run
`benchmark_w4_prefill_swiglu_layer_310.py` at 1,536 tokens with real layer-0
weights. Only after exact layer parity and a favorable layer timing should
the isolated TP4 service be started for matched cold prompts using
[service_client.py](service_client.py). Reuse the saved batching baseline;
there is no need to start a baseline server. Keep the candidate service up
after measurement until the user hands the NPUs elsewhere.

No tok/s or TTFT result has been measured for this fused activation package.
