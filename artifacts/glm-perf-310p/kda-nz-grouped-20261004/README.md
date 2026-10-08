# GLM KDA input projection: transposed NZ grouped matmul

Status: experimental serving promotion candidate, 2026-10-04. The TP4
`FULL_DECODE_ONLY` server is healthy on port 8001 with both decode graph sizes
`[1, 4]` captured. This change retains the 256-K grouped W2/W4 package,
device-resident MLA, prefix cache, and 22,528-token per-request limit.

## Why this projection

The 256-K baseline CANN trace assigned about 1.05–1.08 s of summed task time
to `TransData` over 32 profiled worker steps. The decode event sample showed
many conversions of the KDA fused input-projection weight with logical shape
`[6416, 4096]` before `MatMulV2`. A one-NPU probe found that the relevant
`F.linear` call took about 0.94–0.98 ms for one or four rows, versus
0.30–0.31 ms for one-group `npu_grouped_matmul` with a transposed NZ weight.
The probe matched all output bits at 1, 4, 288, and 640 rows; a simple
changing-input graph replay also matched eager output bits on all six repeats
at each of 1 and 4 rows. Those isolated timings are not server tok/s.

The opt-in `ascend_glm_kda_nz_grouped` HF override packs each KDA projection
weight as transposed NZ after model loading. It stores a transposed logical
view back in the existing parameter, retaining its public `[N, K]` shape and
avoiding a second complete model copy. The KDA forward uses the packed
`[1, K, N]` view and precreated device group lists for decode sizes; variable
prefill row counts create a group list on first use. Packing the `[N, K]`
weight as NZ and only then transposing was explicitly rejected by the probe:
it gives incorrect values on 310P. The new path is off unless requested by
the launcher.

## Full-model gate

The isolated source tree is
`/srv/ai/src/glm-kda-nz-grouped-20261004` on the NPU host. Start it with
`serve-glm-kda-nz-grouped.sh` from this directory, copied to
`/home/matteius/experiments/glm-gate-a-20261002/serve-glm-kda-nz-grouped-20261004.sh`.
The live server log is
`/home/matteius/experiments/glm-gate-a-20261002/server-kda-nz-grouped-20261004.log`.

| Gate | Candidate result | Last promoted 256-K result |
| --- | ---: | ---: |
| 256-token `SHORT_PROMPT`, c1 | 3.884 tok/s | 3.56 tok/s |
| 256-token `SHORT_PROMPT`, c4 aggregate | 8.935 tok/s | 8.60 tok/s |
| Calibrated concurrent 2K retrieval | 4/4 exact | 4/4 exact on two post-restart runs |
| Strict 20-case quality | 17/20 | 17/20 |

All five short-prompt responses reached 256 tokens without early EOS.
The quality run returned 20 valid responses and failed only the known
`instr_reverse`, `instr_first`, and `code_slice` cases. Nineteen of twenty
final strings matched the previous promoted-server quality run exactly; the
`instr_reverse` answer differed but remained incorrect. This single gate does
not resolve the existing first-token instability or establish 20/20 quality.
The speed comparison is successive serving runs, not a tightly paired A/B;
the c4 gain especially should not be overinterpreted.

The calibrated 2K requests had about 1,733–1,735 served prompt tokens each.
Their four-way prefill and decode interleaved and queued in the scheduler, so
that suite's aggregate tok/s is **not** a peak decode measurement. Its purpose
here was the graph-mode concurrent correctness gate.

Local request records are `/tmp/glm-kda-nz-short-20261004.jsonl`,
`/tmp/glm-kda-nz-quality-20261004.jsonl`, and
`/tmp/glm-kda-nz-windows2k-20261004.jsonl`, with adjacent summary JSONs.
The checkpoint tokenizer used to calibrate the 2K prompts has SHA-256
`19e773648cb4e65de8660ea6365e10acca112d42a854923df93db4a6f333a82d`.
Focused KDA host tests passed 7/7 both locally and in the isolated NPU-host
source tree; the model and test files pass Ruff lint and format checks.

The new four-rank CANN capture lives under
`/home/matteius/experiments/glm-gate-a-20261002/profile-kda-nz-grouped-20261004/`
with token `20261004021821877`/`878`. Its operator-level export is being
parsed offline; no trace-attribution claim is made here yet.
