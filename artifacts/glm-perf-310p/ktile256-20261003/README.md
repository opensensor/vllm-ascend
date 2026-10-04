# GLM grouped W2/W4 256-K dequant tile, 2026-10-03

Status: **experimental promotion** at the user's request, prioritizing the
measured 18–20% serving decode gain while an intermittent four-request
graph-mode correctness regression is investigated. This is not a quality
sign-off: a calibrated 2K four-request retrieval was 2/4 exact on the first
256-K graph server, but two full 256-output-token runs after a clean restart
were 4/4 each, including fresh variants after quality/decode warmup; the
128-K graph control was 8/8 across two runs. Do not mix the 256-K
OPP with a 128-K NZ-packed model loader, or vice versa: the packed-code tile
shape is an internal ABI. The last-known-good 128-K launcher remains
`/home/matteius/experiments/glm-gate-a-20261002/serve-glm-prefix-22k-ai-core-round-64k-20261003.sh`.

The 310P vector dequantizer previously processed one 16-output-channel by
128-K packed tile per iteration. This candidate processes 256 K at once and
reorders the lossless NZ-packed codes into corresponding 16×256 tiles at
weight load. The Cube path, W2/W4 signed-code semantics, block scales, and
external operator signature are unchanged; the packed-code layout is not.
The optional L1 matmul retains its separate
128-K Cube stage size so both L0A stages still fit. W4 scratch ends below the
192 KiB AtlasA2 UB limit for the tested GLM K widths.

The isolated kernel source is under `/srv/ai/src/glm-w2-ktile256-20261003`
on the NPU host, with local experiment copies in
`tmp/glm-ktile256-isolated/`. The candidate header SHA-256 is
`62f4b5d91d33da0e5b89e64ec2b776dc770c29c2fab3e791e314d3580f9c4cfe`;
the NZ-packed grouped object SHA-256 is
`2889251f3afb5de10ef781cb75f36eeac87e0941acce511cce7ff4e3a88835fe`.
It used `-DGLM_W2_SCALE_PAIR` and `-DGLM_W2_GROUPED_RINT_UNPACK`, matching
the known-good combined package apart from the tile change. The hardware
sweep used only the new compiled grouped/standalone objects in an isolated
copy of that package, `opp-dev`; the live package was untouched. The model
packing change was tested in a separate copy of the live source tree at
`/srv/ai/src/glm-bf16-round-ktile256-20261003`.

Both packages ran the same 72-expert W4 gate/up and W2 down logical weights,
inputs, scales, route boundaries and seed (`20260930`) in separate processes.
The candidate uses `nzpacked256` in the isolated benchmark harness; the
baseline uses `nzpacked` (128 K). Their packed-byte hashes necessarily
differ, so comparison checks the **logical** code hash and all output bits
instead. The baseline grouped object SHA-256 is
`b211354d0b47407722a5805f974bfeb1722d07d252c0adcf830f60f2ca5e1de3`.

| Routed rows | Cases | Bitwise parity | Sum of baseline medians | Sum of candidate medians | Reduction |
| --- | ---: | ---: | ---: | ---: | ---: |
| 8 | 10 | 10/10 | 22.482 ms | 16.572 ms | 26.3% |
| 32 | 10 | 10/10 | 71.100 ms | 51.856 ms | 27.1% |
| 416 | 10 | 10/10 | 132.627 ms | 99.193 ms | 25.2% |

The separate seven-repeat decode-only sweep also passed **20/20** bitwise
comparisons and reduced the sum of case medians from **93.306 to 68.637 ms
(26.4%)**. The W4 cases improved by roughly 20–23% and W2 by 27–32%.
These are synthetic operator latencies, not GLM tok/s or TTFT.

The repacked loader's focused host suite passed **9/9** tests. With the
candidate OPP on a loaded 310P, the grouped W2/W4 regression suite passed
**19/19**, covering canonical and NZ inputs, 72 experts, varied route
groups, 128+ row groups, and the real GLM widths. The byte-reuse one-hot
probe passed **2/2** across K columns straddling both 128 and 256 boundaries.

The full `--pkg` build completed and its isolated install at
`/srv/ai/src/glm-w2-ktile256-20261003/opp-ktile256` has the same three
operator-object SHA-256 hashes as the tested `opp-dev` copy: NZ grouped
`2889251f...`, canonical grouped `719d210a...`, and standalone
`d4387772...`. The candidate TP4 launcher is
`serve-glm-ktile256-graph.sh` in this artifact directory. It retains the
known-good 22,528-token serving settings, device-resident MLA, prefix cache,
and `FULL_DECODE_ONLY` graph capture sizes `[1, 4]`.

The TP4 candidate reached healthy API startup, and both decode graphs were
captured. Its first strict 20-case quality run scored **17/20**, failing only
the pre-existing `instr_reverse`, `instr_first`, and `code_slice` cases. The
previous 128-K live server scored 16/20 on its most recent run, with those
three plus an intermittent `code_python` failure; an earlier 128-K run also
scored 17/20. This single trial shows no new quality regression, not a
reliable quality gain.

An additional maximum-route operator sweep compared the two packaged
binaries at 5,120 routes with `uniform_all_experts` and `mixed` patterns for
both W4 and W2. All **4/4 outputs were bitwise equal**, including a mixed
group with 5,118 rows on one expert. The candidate remained faster in each
case. Thus an isolated high-route arithmetic or bounds error was not found;
the serving failure appears to require the full model or interleaved requests.
The comparison is
`/home/matteius/experiments/glm-gate-a-20261002/w2-ktile256-maxroutes-comparison-20261003.json`
on the NPU host.

The matched 256-output-token `SHORT_PROMPT` suite completed without early
EOS. Against the previous live server with the same context and graph settings:

| Decode | 128-K live baseline | 256-K candidate | Increase |
| --- | ---: | ---: | ---: |
| c1 | 3.032 tok/s | 3.568 tok/s | 17.7% |
| c4 aggregate | 6.255 tok/s | 7.473 tok/s | 19.5% |

These are request-level streamed decode rates, not synthetic operator rates.

The first four-way 2K retrieval trial got three exact answers and one
incoherent 256-token continuation (`variant=2`). That uncalibrated trial also
failed the suite's *separate* prompt-token target check, since its actual
prompts were about 200 tokens shorter than target. After supplying the exact
checkpoint tokenizer (SHA-256 `19e77364...`), the calibrated candidate trial
got two exact answers, one incoherent continuation (again `variant=2`), and
one answer with extra bold markup. The same calibrated `variant=2` prompt
passed when run alone.

The matched 128-K control used the same 22,528-token *per-request* context, graph capture,
prefix cache, model, seed, and calibrated four-way workload. Its launcher
differs only in the source root, grouped OPP root, and profiler output path.
It passed **4/4 exact answers twice (8/8)**, including `variant=2` in both
runs. Thus the candidate has a strong concurrent correctness regression
signal despite its 18–20% serving decode gain. These were four independent
~1,735-token requests running concurrently, not four shares of one context
window. The 128-K graph server was restored after this A/B; it was later
replaced by the 256-K graph server for forward debugging at the user's
explicit request.

To isolate graph replay, the 256-K source and OPP were restarted with
`--enforce-eager` while retaining the same model, prefix cache, context,
seed, and calibrated four-way prompts. Its launcher is
`serve-glm-ktile256-eager.sh` in this artifact directory; compared with the
candidate graph launcher, it differs only in the profiler path and graph/eager
flags. Eager passed **4/4 exact answers twice (8/8)**, including `variant=2`
on both runs. The graph candidate's repeated incoherent answer therefore
appears to require graph-mode execution or its scheduling interaction with
the widened packed layout; isolated operator arithmetic and eager full-model
serving were not sufficient to reproduce it. This is an inference from the
controlled runs, not a proven kernel-level race. The eager variant is slower
than the known-good graph baseline, so it is not a serving replacement. The
128-K graph server was restored after the diagnostic.

One-NPU probes then captured the candidate grouped operator by itself and
replayed it with changing activation tensors and device-side expert
boundaries. W4 and W2 each passed **24/24 bitwise comparisons** with eager.
A captured W4 gate/up → SiLU/multiply → W2 down chain also passed **24/24**
changing-input comparisons. These probes used four routed rows, the real
4096-output widths, and alternating expert distributions. They make a simple
single-op capture failure less likely, but do not cover GLM's KDA/MLA/QSA
state or its full-forward graph. The next diagnostic is a per-layer
graph-versus-eager activation trace on the same concurrent prompts, starting
at the first decode token. The 256-K loader and kernel are being promoted
together, with the regression retained as a known experimental limitation.

After the promotion restart, the real-weight graph server reached `/health`
HTTP 200 and answered the calibrated four-request 2K retrieval **4/4 exact**
on its first fresh run. This does not erase the prior 2/4 failure; it makes
intermittency explicit. The 256-K captured operator was then tested at the
real four-request decode route width (32 routed rows): W4 24/24, W2 24/24,
and a W4 gate/up → activation → W2 down chain 24/24 all matched eager
bitwise under changing inputs and expert boundaries. The first divergence
has not yet been localized to a full-model layer or stream boundary.

The promoted server's subsequent strict quality/decode warmup reproduced
**17/20** quality (the same three misses: `instr_reverse`, `instr_first`,
`code_slice`), **3.56 tok/s c1**, and **8.60 aggregate tok/s c4** with no
early EOS in the 256-output-token `SHORT_PROMPT` runs. The short c4 result
varies with runtime load; the earlier paired 7.47 tok/s remains the direct
controlled comparison to the 128-K 6.26 tok/s run. An additional one-NPU
probe used the same breakable-graph mechanism as GLM, with **11 eager graph
breaks** across repeated W4→activation→W2 calls at 32 routes. It also
matched eager bitwise for **24/24 changing-input replays**. This further
narrows the intermittent full-serving failure to scheduling, attention/state
metadata, or an interaction not represented by the expert-only probes.
The graph-enabled runtime also depends on the matched GLM-specific 310P MLA
projection, kpool fixed-shape writer, sparse-backend selection, and KDA
metadata builder and model installer. These production graph files in the
promotion commit match the live source byte-for-byte. Targeted remote host
suites passed **21/21**
MLA, **29/29** kpool, and **8/8** KDA/backend-selection checks. A third
four-request retrieval with a 32-token cap and fresh variants passed 4/4;
that cap intentionally changes scheduling and is not counted as a full
256-token quality gate.

A fresh four-rank CANN capture of the promoted server used one c1
`SHORT_PROMPT` request and profiled 32 worker iterations. The complete
`op_statistic.csv` files agree on **2,688 grouped W2/W4 calls per rank**.
Their grouped task totals are **3.54–3.80 s**, or **41.5–43.0%** of the
reported per-rank operator time. The next categories are `TransData`
(1.05–1.08 s), `MatMulV2` (0.85–0.87 s), KDA (0.68–0.69 s), and AI-CPU
`Cast` (0.51–0.58 s). These are whole-capture summed task times, including
prefill; they are neither decode-only times nor an additive critical path.
The per-event `kernel_details.csv` exports are **truncated at different
timestamps**, so their apparent rank imbalance and cross-rank envelope are
invalid. In particular, the complete operator summaries show equal grouped
call counts, contrary to the truncated event files. Capture files live under
`/home/matteius/experiments/glm-gate-a-20261002/profile-ktile256-20261003/`
with token `20261004012034999`.

Local request records:
`/tmp/glm-22k-ktile256-windows2k-calibrated-20261003.jsonl`,
`/tmp/glm-22k-128k-windows2k-calibrated-20261003.jsonl`, and
`/tmp/glm-22k-128k-windows2k-calibrated-repeat-20261003.jsonl`, plus
`/tmp/glm-22k-ktile256-eager-windows2k-calibrated-20261003.jsonl` and
`/tmp/glm-22k-ktile256-eager-windows2k-repeat-20261003.jsonl`.

Hardware records are on the NPU host under
`/home/matteius/experiments/glm-gate-a-20261002/`:
`w2-ktile256-dev-nzpacked-20261003.pt`,
`w2-ktile256-dev-comparison-20261003.json`,
`w2-ktile256-prefill-nzpacked-20261003.pt`, and
`w2-ktile256-prefill-comparison-20261003.json`.
