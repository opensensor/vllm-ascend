# Deferred Qwen hardware gates

No commands in this runbook have been used to launch or change the live server.
The existing image-enabled baseline and its 94C hold / all-chip 85C resume /
independent 96C cutoff remain in control. Do not run these gates concurrently
with user service testing without a new hardware handoff.

## Offline tools

Generate a reviewable six-chip config/profile from the saved checkpoint config:

```bash
python -m tools.qwen4exp.three_card_profile \
  artifacts/qwen38-memory-audit-20261008/model-config.json /tmp/qwen-tp6-profile
```

For a later canonical checkpoint, create a separate append-only asset overlay:

```bash
python -m tools.qwen4exp.three_card_profile \
  /path/to/canonical-checkpoint/config.json /path/to/new-tp6-overlay \
  --checkpoint /path/to/canonical-checkpoint
```

The overlay links the original weights/tokenizer/processor/index. It does not
export or re-quantize weights. Reject TP-specific prepacked checkpoint sources;
verify the canonical per-expert loader filter against all six ownership ranges.
The saved example config is not a standalone weight checkpoint.

## Standalone streaming gate

Use the contained append-only builder with `projection_variant=m32n160_v2`,
`layers.native_wy=false`, and `windows.tile_columns=160`. Namespace version,
contract SHA, source assets, binaries and bridge must match. A projection/gather
bundle is not an admitted whole-model deployment manifest.

Source the installed CANN `set_env.sh` before host compilation. The builder
preserves SDK loader paths and prefixes explicit CANN, driver and Torch library
directories in the contained compiler child. Parent process settings are unchanged.

The new benchmark compares the real layer-0 expert partial against the baseline
with independent seeded activations, identical routes, warmup and three ABBA
cycles. It excludes shared experts and HCCL, and uses native candidate gather
versus the baseline load_layer default gather policy. Record stage attribution
separately before attributing any complete-path improvement to projection alone.

```bash
ASCEND_RT_VISIBLE_DEVICES=0 python -m tools.qwen4exp.benchmark_streaming_next_310 \
  --bundle /path/to/verified-bundle --model /path/to/canonical-checkpoint \
  --output /path/to/new-evidence.jsonl --tp-size 4 --execute
```

Device 0 is appropriate only if it is an isolated granted device in the current
inventory. The harness selects visible device 0, reads six sensors and stops at
90C; it never discovers or acquires an NPU lease. No benchmark has been run for
this follow-up. Require exact raw projection/packing parity and changed-input
graph stress before real-weight trials. Preserve v5 as the rejected comparator.

The queued raw parity and changed-input graph replay tests are separate from the
real-weight benchmark. Run only after an isolated device handoff:

```bash
ASCEND_RT_VISIBLE_DEVICES=0 pytest \
  --confcutdir=tests/e2e/nightly/310p/single_node/ops/streaming_next \
  tests/e2e/nightly/310p/single_node/ops/streaming_next \
  --qwen-streaming-next-bundle /path/to/verified-bundle \
  --qwen-streaming-next-execute
```

Without both options these tests skip before any NPU library load. Gate/up tails,
85/86-expert boundary caches, two down windows and changed-input graph replay
are covered. These tests have not been executed on hardware.

For WY, use `capture_wy_trace` only in an isolated diagnostic process. It performs
explicit diagnostic CPU readback. Supply the exact reference/native preparation
callbacks and a downstream callback returning both output and final state.
Save normalized Q/K identity and clone initial states. Retain FP64 stage oracles
and first-divergence metrics. Reference WY remains selected until the original
native downstream output and state gates pass.

## Six-chip full model gate

Start with the baseline projection backend rather than combining TP6 and the new
streaming candidate. Required settings for the new model profile are TP6, PP1,
EP enabled, `--mm-encoder-tp-mode data`, images enabled and MTP disabled. Omit
speculative configuration entirely. Keep the source config's trained heads and
vocabulary; padding is local loader/allocation policy. External KV transfer is
unsupported by this candidate. Use a new frozen runtime and the existing known
OPP stack; changing flags against already imported workers is insufficient.

Before submitting inference, check all rank expert ranges, QSA KV ownership,
nine-head GDN state descriptors, 1,920-channel convolution state, 41,408-row
embedding/head shards, image encoder replication, and collective participation.
Measure actual weights, state/cache, graphs/HCCL, vision and route scratch peaks
against each chip's usable capacity. Keep meaningful reserve; total installed
RAM cannot establish context capacity. Do not claim more windows from the routed
weight ledger alone.

Then run text/image cold and cached requests, all-rank real activation/state
comparison, cancellation and prefix CoW, graph replay with changing inputs, and
sustained mixed workloads with thermal holds included in timings. Record startup,
TTFT, decode, per-rank skew, memory peaks and temperatures. Compare six-chip
non-MTP throughput with the actual TP4/MTP2 baseline explicitly; the arithmetic
and drafting profiles differ. Do not present an isolated padded-head CPU proof
as full model accuracy or six-rank support.
