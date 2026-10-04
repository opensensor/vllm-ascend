# GLM grouped W2/W4 256-K dequant tile, 2026-10-03

Status: **rejected for serving** after a concurrent retrieval A/B. The
known-good 128-K graph server was restored. Do not mix this candidate OPP
with 128-K NZ-packed checkpoint banks: the model loader and grouped operator
must be changed together because the packed-code tile shape is an internal
ABI. Its source and package remain isolated for root-cause work.

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

The matched 128-K control used the same 22,528 context, graph capture,
prefix cache, model, seed, and calibrated four-way workload. Its launcher
differs only in the source root, grouped OPP root, and profiler output path.
It passed **4/4 exact answers twice (8/8)**, including `variant=2` in both
runs. Thus the candidate has a strong concurrent correctness regression
signal despite its 18–20% serving decode gain. The known-good 128-K server
was left healthy on port 8001. No 256-K loader/kernel change was promoted to
the main source or committed.

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
at the first decode token; only after locating the first divergence should
the 256-K optimization be revised.

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
