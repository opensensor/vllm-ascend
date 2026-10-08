# GLM-5.3-Flash context on two Ascend cards: quantization decision

Started 2026-10-03 as a checkpoint and source audit. The host has four 48 GiB
310P devices on two cards; the GLM service uses TP4. Results below distinguish
the initial offline estimates from later device measurements.

## Packed W3 graph result, 2026-10-04

The same selective-W3 checkpoint and NZ-packed source as the paired eager
run below loaded at 192K with prefix caching disabled and four sequence slots.
`FULL_DECODE_ONLY` captured decode sizes 1 and 4 in four seconds, using about
0.19 GiB of graph memory per rank. A real chat request answered `42` for
`17+25`. The 256-token short suite completed all five requests without early
EOS:

| Matched packed-W3 run | Eager | Graph | Change |
| --- | ---: | ---: | ---: |
| c1 decode | 3.131 tok/s | 3.550 tok/s | +13.4% |
| c4 aggregate decode | 7.092 tok/s | 7.556 tok/s | +6.5% |

The strict 20-case answer gate remained **17/20**, with exactly the same
three misses as eager W3: `instr_reverse`, `instr_first`, and `code_slice`.
Thus graph capture works and improves decode modestly, but it does not fix
quality or create the Qwen-scale graph speedup. Client records are in
[`packed-w3-paired/`](packed-w3-paired/). The server log is
`/home/matteius/experiments/glm-w3-20261004/server-w3-nz-graph-correct-20261004.log`
on the NPU host. The standard remote launcher had not yet received the local
graph-mode arguments, so this run used a verified staged copy named
`serve-selective-w3-graph-eval-20261004.sh`; its process command line and
engine config both showed graph mode before any timing was accepted.

A tokenizer-calibrated 8K retrieval also passed: 7,877 served prompt tokens,
the exact secret-code answer, and 180.5 s TTFT (43.7 effective prompt tok/s).
This is a real prefill result, not a 192K prompt test. The engine admitted a
192K configured window and reported 239,781 cacheable tokens (1.22× maximum
single-request concurrency at that limit) with 5.95 GiB KV memory on rank 0.
Actual long-window prefill remains unvalidated and is clearly the larger usability
problem. The 8K client record is `packed-w3-paired/window8k-w3-nz-graph-20261004.jsonl`.

### 512-wide QSA Cube prefill candidate, 2026-10-04

A two-step Torch/CANN profile of the 640-token graph server found that eleven
512-wide QSA calls consumed 5.162 s of rank-0 device task time, 26.6% of the
summed task time. The existing Cube QSA specialization only selected 256-wide
queries. Main now also instantiates that kernel for GLM's 512-wide latent
queries. A one-card operator probe measured 470.044 to 63.131 ms for a dense
640-query continued-prefill call, with at most 7.63e-6 absolute output
difference from the vector path. The candidate also passed a short FP32
reference and changing-input graph replay.

At the same 640-token scheduler chunk and 192K configured window, the combined
QSA Cube512 and opt-in histogram candidate kept the baseline's 239,781-token
cache capacity. The matched 7,877-token retrieval returned the exact code in
**99.04 s TTFT**, compared with **180.49 s** for the 640/compare vector-QSA
control, a 45.1% reduction in this pair. Its strict answer suite scored
**17/20** with the same three misses as the prior graph W3 run. The full-model
pair changes QSA and route counting together; the route-count microcheck's
0.025 ms saving per 640-token call indicates that QSA supplies most of the
gain. See [`../glm-perf-310p/qsa-cube512-20261004/README.md`](../glm-perf-310p/qsa-cube512-20261004/README.md)
for the profile, operator evidence, package path, and remaining gates.

### Prefill route candidate, 2026-10-04

The device-grouped MoE dispatcher currently counts routes with an explicit
`[routes, local_experts]` equality and reduction tensor. At the 640-token
chunk limit and top-8 routing, that is up to 5,120 routes compared with each
local expert at every MoE layer. Main now has an **opt-in** fixed-bin
`torch.histc` count path for prefill. It bounds peer IDs to the existing
sentinel before FP32 conversion, drops that sentinel's bin, and falls back to
the comparison path if an exact FP32 key or count cannot be guaranteed.
Decode and the default serving path still use the existing comparison count.
CPU tests cover empty, skewed, peer-owned, large-ID, and tiled routing,
including exact output parity with the old path. The focused suite passed
43/43. The hardware regression
`tests/e2e/nightly/310p/single_node/ops/test_glm_prefill_route_histogram_310.py`
passed 4/4 on a 310P: eager parity at 9, 128, and 640 tokens plus
changing-input graph replay. At 640 tokens, the synchronized route-count
median fell from 0.253 to 0.228 ms and the complete dispatch from 0.519 to
0.497 ms. That saving alone is too small to explain the 8K prefill time.

Main also raises the grouped W2/W3/W4 route limit from 5,120 to 6,144 in the
310P tiler and Python caller, allowing a 768-token top-8 chunk in one grouped
call. The two grouped kernel binaries were byte-identical to the 5,120-route
package; only tiling admission changed. With 72 experts and real GLM W3
projection dimensions, the single 768-token call matched the old 640+128
schedule bitwise and took 128.5 versus 238.7 ms. A separate 6,144-route NZ W3
hardware regression passed bitwise, as did 31 focused CPU tests.

The matched four-rank 8K retrieval was correct with the same 7,877 served
prompt tokens in both configurations. TTFT changed from **180.49 s** at
640/compare to **174.51 s** at 768/histogram, only 3.3% faster. The configured
192K window was admitted, but cacheable capacity fell from 239,781 to
215,507 tokens, reducing single-request margin from 1.22x to 1.10x. These
are single matched runs, and histogram and chunk size were changed together.
The candidate is **unpromoted**: the capacity loss and small TTFT gain do not
yet justify a server change. Client records are
`packed-w3-paired/window8k-prefill-compare-640-20261004.jsonl` and
`packed-w3-paired/window8k-prefill-route6144-histogram-768-20261004.jsonl`;
server logs are in the NPU host's `glm-w3-20261004` experiment directory.
The launcher defaults remain 640/compare; its optional eighth and ninth
arguments select chunk size and route-count mode, and its tenth argument can
capture a two-iteration Torch profile.

## Packed W3 result, 2026-10-04

The main-tree candidate was staged as two deployment copies on the NPU host.
They used the same checkpoint, final custom OPP package, KDA mask fix, KDA NZ
projection, W2/W4 paths, 192K context, four sequence slots, eager execution,
and port **8001**. The only source difference was that the canonical copy
kept W3 resident codes in row order while the candidate repacked W3 into
plane-major 16×256 NZ tiles. Neither deployment copy is a separate source of
truth. Both used real weights and prefix caching disabled.

| 256-token short suite | Canonical W3 | Packed NZ W3 | Change |
| --- | ---: | ---: | ---: |
| c1 decode | 2.075 tok/s | 3.131 tok/s | +51% |
| c4 aggregate decode | 4.354 tok/s | 7.092 tok/s | +63% |
| Valid requests | 5/5 | 5/5 | All reached 256 tokens |

At 192K, the packed server loaded 34.1214 GiB of weights per rank, reserved
5.95 GiB of KV cache, and reported 239,781 cacheable tokens with maximum
concurrency 1.22 at that context length. Device memory while idle was about
43.3–43.5 GiB per rank. The one-time CPU repack increased model loading from
about 160 to 240 seconds in this pair. A separate grouped-operator benchmark
in the same OPP package found bitwise equal outputs and 4.04–4.53× lower
median call latency across gate/up and down shapes with singleton and
repeated routes. The seven-case NPU operator probe passed.

The earlier W4 service measured 3.884 tok/s c1 and 8.935 aggregate tok/s c4,
but it used graph execution and a different checkpoint and context setting.
It is not a paired comparison. The current W3 pair isolates the packing
change. Short responses differed between the two runs despite bitwise equal
isolated operator outputs, so packed W3 still needed its own strict 20-case
quality run before promotion. The later graph run above completed that gate
at 17/20. No packed-W3 CANN decode trace has been taken; remaining time has
not been assigned to particular operators.

The five request records, summaries, and paired operator benchmark are in
[`packed-w3-paired/`](packed-w3-paired/). Full server logs remain at
`/home/matteius/experiments/glm-w3-20261004/server-w3-canonical-paired-v2.log`
and `server-w3-nz-paired.log` on the NPU host. Both servers were stopped at
the user's request at the end of that pair. A later graph server now occupies
port 8001 and all four NPUs, as described above.

The local launcher accepts an optional seventh argument, `graph`, selecting
`FULL_DECODE_ONLY` with capture sizes 1 and 4; its default remains `eager`.
The standard remote launcher was stale during the later graph test, so a
verified staged copy was used for the successful result above.

## Earlier native W3 result and candidate preparation

The later native Cube run loaded the selective W3 checkpoint at 192K with
prefix caching disabled. Live weights were 34.1213 GiB per rank and the engine
reported 242,265 cacheable tokens. Its strict 20-case suite finished 17/20,
with the same three failing case IDs as the prior W4 candidate. The user
accepted that personal quality gate. Decode was about 2.0–2.1 tok/s on the
short quality prompts, so performance remains below the prior W4 service
result. These are not paired speed measurements.

Main contains a candidate that repacks resident W3 matrices into
plane-major 16×256 NZ tiles while retaining the same three bytes per eight
codes. The grouped Cube kernel decodes those tiles without the canonical W3
gather tables, output gather, or 16 tile transposes. Canonical W3 remains
available for the standalone operator and non-NZ paths. The CPU layout and
bank tests pass. The
640-token KDA-mask experiment uses a separate W4 runtime source and
does not exercise this W3 path. The KDA mask fix and W2 route tiling from that
experiment are already in main.

The candidate CANN package compiled from main's W3 header and grouped tiler,
plus main's grouped wrapper and CMake flags for decode-table reuse, scale
pairing, and RINT unpack. A byte comparison caught the older wrapper in the
initial build staging directory before this final package was installed.
The final installer is in the build-only staging directory at
`/srv/ai/src/build-only-glm-w3-nz-csrc-20261004/build/cann-ops-transformer-custom_linux-x86_64.run`
with SHA-256 `a30c922ed73914f62baf017ec36b056f9cb2b729a4b72dcb97de831b5e2a8e43`.
It is installed only under that staging directory as `opp-w3-nz-candidate`;
the active server's OPP search path was not changed. A one-thread CPU repack
of a 2048×4096 W3 projection took about 0.04 seconds locally after replacing
the general sign-extending unpack with direct unsigned byte extraction. This
is a load-time microcheck, not a decode throughput result.

The device probe used `probe-native-w3.sh` with the main-tree deployment copy,
device 0, and the package above. It passed NZ W3 parity at 256×256, 2048×4096,
and 4096×2048, as well as the canonical W3 cases. The paired real-weight
serving result is recorded above.

## Earlier 2026-10-04 device result and native W3 work

The selective W3 checkpoint loaded on four 310P devices with prefix caching
disabled and `--max-model-len 196608` on port **8001**. Live weights were
34.1213 GiB per rank. The server allocated 5.83 GiB for KV and reported
236,364 cacheable tokens, enough to admit the configured 192K window. A
real-weight chat request returned `42` for `17+25`. A strict quality run was
stopped after three completed cases at the user's request to free the NPUs;
all three arithmetic cases passed. This is not a 20-case quality pass.

The eager W3 path took about 36–37 seconds to prefill each 25-token quality
prompt and about 4–5 seconds per decode token. It is not usable for coding
despite the larger cache allocation. The device server and quality run were
stopped, and `npu-smi` reported no running NPU processes. A 256K window has
not been launched.

The next candidate extends the existing W2/W4 310P Cube operator to decode
eight signed W3 codes from three canonical bytes inside its output tile. It
uses the resident grouped expert bank, bounded NZ workspace, and an opt-in
native OPP package built in the isolated source tree. Host bit-layout and
dispatch tests pass. A device operator parity and throughput gate is still
required before another full-model launch. The launcher takes a fourth
argument for the KV fraction; a future 256K attempt can use `262144 0.75`
after the native operator passes.

The native package compiled and installed at
`/srv/ai/src/glm-selective-w3-20261004/opp-w3-native`. Its installer SHA-256
is `83eee8320472b96839a7995ec22cdb8a5797f74b81ff7056da9eb883bcbde4b0`.
Both standalone and grouped kernel objects were rebuilt with the W3 input
bound and compiled successfully. Local tests: 24 packed-method/layout tests
and the W3 resident-bank test passed; Ruff and shell syntax checks passed.
The native W3 operator parity probe subsequently passed all four 310P tests:
grouped and standalone small-shape CPU reference checks, plus grouped
2048×4096 gate and 4096×2048 down-projection checks. At this stage full-model
native W3 speed and answer quality were unmeasured. A server launch was stopped at the
user's request before model loading; port 8001 is closed and `npu-smi` shows
no running NPU processes.

When one device is available, run the staged parity probe before serving:

```bash
bash /home/matteius/experiments/glm-w3-20261004/probe-native-w3.sh \
  /srv/ai/src/glm-selective-w3-20261004 0
```

The full-model launcher remains on port **8001** and requires the native OPP
package. For example, after parity and speed checks pass:

```bash
bash /home/matteius/experiments/glm-w3-20261004/serve-selective-w3.sh \
  /srv/ai/src/glm-selective-w3-20261004 \
  /srv/ai/models/GLM-5.3-Flash-selective-W3-310p 262144 0.75
```

## Decision

The current W4-through-32 server has now reached 128K with prefix caching
disabled. In this GLM cache implementation, that setting switches the 34 KDA
layers to a fixed live state pool, leaving only the 11 MLA histories and
compressed indexer histories proportional to context. The selective W3
candidate should be tested above that measured 128K baseline. Repeated coding
turns still need separate timing because prefix reuse is disabled.

If four 128K windows need more resident HBM, investigate a *selective* packed
W3 revision of the current W4 expert layers. Keep W4 on layers whose downgrade
changes full-model answers or causes residual growth. Do not replace the
working checkpoint with the all-W2 artifact: prior full-model W2 experiments
were incoherent, and even W4-through-26 had substantial deep-layer drift.

With prefix caching enabled, quantization alone is unlikely to produce a 128K
window. That mode retains historical KDA states per scheduler block and pads
the shared physical page to the KDA size. The 32K trial explicitly reserved
5 GiB of KV and reported only 34,837 cacheable tokens. Preserving prefix reuse
at 128K therefore needs a separate KDA-state archive/tier design or another
cache-layout change, with state restoration parity; W3 weight savings alone
cannot bridge the observed capacity gap.

## Measured checkpoint bytes and projected weight budget

The selected tensor bytes below were counted from the safetensors index and
the indexed shard headers. Overlay shards contain superseded tensors, so the
index's stale `metadata.total_size` is not the active checkpoint size.

| Expert map | Selected checkpoint, GiB | TP4 checkpoint lower bound, GiB/device | Calibrated live weights, GiB/device |
| --- | ---: | ---: | ---: |
| All W2 | 90.657 | 22.664 | 23.997, estimate |
| W4 layers 3–26, W2 27–44 | 131.157 | 32.789 | 34.122, estimate |
| Current W4 layers 3–32, W2 33–44 | 141.282 | 35.321 | **36.654, measured** |
| W3 layers 3–32, W2 33–44 | 115.970, projection | 28.992 | 30.325, projection |

The calibrated estimates add the current difference between measured live
weights and the indexed TP4 lower bound (1.333 GiB/device). They are planning
figures, not startup measurements. Replacing one W4 expert layer with an
ideally packed W3 layer saves 0.210938 GiB/device; changing all 30 W4 layers
would save 6.328 GiB/device. Eight selective layers would save 1.688 GiB/device.

The current idle graph server used about 42.7–42.9 GiB/device and had about
3.9–4.5 GiB physically free in one profile. That number includes cache,
runtime, graph, and allocator effects and cannot be added directly to the
weight estimate as a safe allocation budget.

## Cache geometry and context implication

For the no-prefix physical layout, 11 MLA layers at 512 FP16 latent values
cost 11 KiB per historical token per rank. The compressed indexer costs
704 bytes/token per rank. Thus one 128K window needs about 1.461 GiB of
history and four 128K windows need about 5.844 GiB, before the fixed KDA
pool, incomplete-pool tail, graph, workspaces, fragmentation, and safety
margin. One 256K window needs about 2.922 GiB of history. These are physical
history estimates, not admission or latency results. The 1,048,576-token
position setting in the checkpoint is not a validated serve length.

The prefix-enabled 32K trial's 5 GiB / 34,837-token ratio is about
150 KiB/token of *configured cache budget* (including padding/fixed charges).
Even a linear extrapolation would require roughly 19 GiB for one 128K window.
This explains why saving 6.328 GiB of weights is insufficient in that mode.

## FP8 source weight probe

Read expert 0's gate, up, and down FP8 tensors at layers 3, 7, 19, 27, 32,
and 44. For each tensor, dequantized a 128×512 top-left tile with its source
FP32 `[128,128]` scale, then requantized with signed integer codes and FP32
`[32,32]` block scales. Mean metrics across the 18 tiles:

| Scheme | Weight cosine vs source | Normalized squared error |
| --- | ---: | ---: |
| W2, no-clip scale | 0.8352 | 0.3946 |
| W2, MSE scale | 0.9212 | 0.1514 |
| W3, no-clip scale | 0.9637 | 0.0762 |
| W3, MSE scale | 0.9793 | 0.0411 |
| W4, no-clip scale | 0.9914 | 0.0173 |

This is a weight-only screening result from one expert and one tile per
projection. It does not predict coherent full-model output. Prior W2 MSE
scales improved weight cosine but produced correlated clipping bias and
residual growth. Start a W3 candidate with the no-clip scale, then compare
MSE only after layerwise residual and answer checks.

The existing `pack_codes` uses `8 // n_bits` codes per byte. Passing `n_bits=3`
would store only two codes/byte, effectively using four bits per weight and
saving **no HBM**. A real W3 candidate needs eight signed codes packed into
three bytes, an explicit format manifest, loader shape checks, NZ conversion,
and a 310P unpack/Cube path. The current W2/W4 kernel cannot consume that
format unchanged. A CPU/eager prototype may establish quality but cannot
establish deployable speed or memory.

## Next isolated gates, after the current quality/profile run

1. Preserve the exact current checkpoint, source and OPP hashes. The current
   no-prefix server reached 128K. On the selective W3 candidate, use
   `--max-num-seqs 1` and record per-rank allocator and cache ledgers,
   startup admission, and full answers at 192K, 224K, and 256K. Confirm the
   reported KDA cache mode is live-only.
2. Compare cold and repeated-prompt prefill, 256-token decode, four-request
   throughput, and the existing strict 20-case quality suite against a paired
   prefix-enabled control. The current quality gate must finish before this
   comparison is interpreted. Include code editing, tool calls, retrieval,
   and long-context position tests; reject truncation and incoherent finals.
3. If measured HBM needs another 1–3 GiB/device for the desired concurrency,
   prototype W3 on 8–12 low-sensitivity W4 layers in an isolated artifact.
   Rank layers by one-layer quantization ablation against full-answer and
   layerwise-logit/residual evidence, not by weight cosine alone. Preserve
   W4 where errors matter. Check actual packed bytes before NPU conversion.
4. Require 310P operator parity, real-weight full-model quality, per-rank peak
   HBM, the declared context tier, and paired throughput before promotion.
   If users require prefix reuse at long context, separately implement and
   qualify a bounded KDA prefix-state archive; do not label a no-prefix result
   as equivalent for repeated coding turns.

Local evidence: `artifacts/glm-progress-summary.md`,
`artifacts/glm-perf-310p/prefix-profile-20261003/README.md`,
`tools/glm_w2/context_offload_plan.md`,
`vllm_ascend/models/glm5next/cache_config.py`, and
`vllm_ascend/_310p/model_runner_310p.py`.

## Selective packed-W3 candidate prepared offline

The first candidate uses signed W3 for KDA expert layers **8, 9, 10, 12, 13,
14, 16, and 17**, W4 on the other selected early expert layers through 32,
and the existing W2 on layers 33–44. This is a heuristic first map, not a
layerwise quality ranking. W3 has eight signed codes in three bytes and keeps
the existing FP32 `[32,32]` scales. The converter uses a no-clip scale. The
existing W2/W4 Cube operator is excluded for W3; W3 currently runs through
device FP32 dequantization and matmul, so throughput remains to be measured.

The local overlay checkpoint is
`/run/media/matteius/20TB-drive/models/GLM-5.3-Flash-selective-W3-310p`.
It uses eight W3 shards and symlinks the unchanged base shards. The index
selects **144,452,890,300 bytes (134.532 GiB)**, versus **151,700,647,612
bytes (141.282 GiB)** for the current checkpoint. Exact selected payload
saving is **6.750 GiB total**, or **1.6875 GiB per TP4 rank** if resident
ownership follows the existing contiguous expert placement. This is a weight
budget; startup HBM and admitted context are still unmeasured.

Against the *measured* 128K no-prefix baseline, the ideal extra history from
1.6875 GiB/rank is about 151K tokens at full allocation, or **106K tokens**
when the 0.70 KV fraction is applied. That puts the memory-only estimate near
**232K total tokens**. The W3 eager path may need larger temporary FP32
workspaces, and the baseline may have spare cache beyond its configured 128K.
Probe 192K and 224K first; treat 256K as a stretch tier until it runs.

The metadata loader audit passed: 41 indexed shards, 75,575 indexed tensors,
all 42 decoder MoE layers covered, and the final shard for every tensor
matches the index. The opt-in `glm_w2_filtered` loader is required for a clean
measurement because it skips both peer experts and superseded W4 tensors before
reading their payloads. A real-weight check on expert 0 gate/up/down at layers
8 and 17 gave cosine 0.9640–0.9652 against FP8 and relative MSE
0.0750–0.0777. Those weight metrics do not establish answer quality.

To reproduce the local artifact, run:

```bash
python3 -m tools.glm_w2.convert_w3_overlay \
  --source-dir SOURCE_FP8 --base-dir BASE_W4THROUGH32 --out-dir CANDIDATE
```

The default layer list is the eight layers above. Repeating this command
verifies and reuses complete W3 shards. The manifest records shard SHA-256
values and source/base index hashes. The W3 shards are self-contained; the
base shard symlinks must be relinked after moving the directory. In the
staged source tree, run:

```bash
python3 -m tools.glm_w2.convert_w3_overlay --relink-base \
  --base-dir REMOTE_BASE --out-dir REMOTE_CANDIDATE
```

This checks the base weight map before changing links. The small tokenizer
and config files are copied into the overlay.

When all four devices are free, stage the current source changes in an
isolated source root and use [`serve-selective-w3.sh`](serve-selective-w3.sh)
with arguments `SOURCE_ROOT CHECKPOINT 32768`. It listens on local port 8001,
uses one sequence and no prefix cache, and starts in eager mode. First confirm
one short answer and the fixed 20-case strict suite with:

```bash
python3 -m tools.glm_perf.suite --base-url http://127.0.0.1:8001 \
  --model glm53-flash-selective-w3 --workload quality --output QUALITY.jsonl
```

Then compare exact finals to the current paired baseline and inspect per-rank
peak HBM and loader skip counts. If coherent, retry with 192K, 224K, and 256K
model lengths, one long coding prompt, and a 256-token decode slice. Eager W3 speed
is a gating result; a packed Cube W3 kernel is needed if it is too slow.
