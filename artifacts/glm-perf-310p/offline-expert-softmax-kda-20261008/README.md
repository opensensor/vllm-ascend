# GLM: next three offline candidates

This directory records the offline phase. Subsequent device qualification and
serving comparisons are recorded in
[the device handoff](../expert-softmax-kda-device-20261008/README.md).

The expert, QSA and KDA candidates are implemented and opt-in. All 1,585 GLM
CPU tests passed at this snapshot. At that point, no CANN build, NPU execution,
real-weight qualification or measured speedup had occurred, and the candidates
had not been installed. The linked device handoff records the later results.

| Candidate | Work removed or batched | Additional UB per core |
| --- | --- | --- |
| Expert product readback | Two full-pipe drains become explicit Cube/vector dependencies | 0 |
| Expert route store | One strided FP16 DMA replaces per-row stores and lifetime fences | 0 |
| QSA softmax | Head max/sum reductions and previous-max exponentials are batched | 256 B |
| KDA score matrices | Column subtraction, two product passes and both reduction trees use repeat instructions | 0 |

## Expert scheduling

`build_reconstruction.py` accepts independent `--product-pipe-events` and
`--bulk-route-store` flags. Both default off. Product readback uses M/V and V/M
handshakes; the latter protects CO1 against a subsequent Cube overwrite. Other
activation/weight lifetime fences remain. This does not add double buffering.

Bulk route stores require fused MoE, native column order and FP16 route storage.
The compiler define is emitted only for down projections, including specialized
W4 down binaries. Decode continues through the original combine branch. The
workspace, rounding boundary and FP32 weighted reduction remain the same.

Host tests compile the actual Product/Combine bodies. They check readback event
order, full route placement, unchanged decode reduction and partial rows across
256/2048/4096 output widths. The host Cube and pipeline primitives are stubs;
the tests do not establish hardware scheduling correctness or throughput.

## QSA softmax heads

`build_qsa_shared.py --vector-output --vector-accumulate --softmax-head-batch`
adds softmax batching to the v991 source schedule. The two-float maximum/index
result layout and one-float sum layout have distinct destination strides.
Scalar per-head online-softmax state retains the original FP32 expressions.
Padding and the FP16 probability boundary remain. Event 7 is reserved only when
unused by the parent; source drift and duplicate staging are rejected.

Host tests compile the actual complete SoftmaxTile and helpers for 1/2/3/4/12/16
heads, five partial/full token widths and three successive online-softmax steps.
They compare state and complete output buffers. The nightly operator probe now
includes six partial-head dense/sparse cases, for 36 full cases with changed
ACL graph replays. Those hardware cases have not been run for this candidate.
Vector exponential accuracy and event support still need CANN/NPU qualification.

The reduction stride contracts were checked against Huawei's official
[WholeReduceMax reference](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/81RC1alpha001/apiref/ascendcopapi/atlasascendc_api_07_0079.html)
and
[WholeReduceSum reference](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/83RC1alpha002/API/ascendcopapi/atlasascendc_api_07_0081.html).
API documentation is not a substitute for compilation on the serving SDK.

## KDA score matrices

`stage_kda_score_matrix.py` stages only the cached FP16 K=128, BT=64 full-chunk
branch of the qualified row-batch parent. It broadcasts the row operand through
repeat source stride zero. Both independent 64-lane FP32 sums and the parent's
`+0 + p0 + p1` combine remain. FP16 intermediate products still round at the
same boundaries. Other chunk shapes and tails keep their original code.

Existing scratch is reused: half products occupy 16–32 KiB; FP32 products occupy
32–64 KiB; cached columns begin at 80 KiB. No buffer is added. Host tests execute
the actual full-row branch for four input patterns, compare both 64x64 matrices,
check the immutable column-cache region, and verify disabled preprocessing and
invalid feature admission. Hardware vector/reduction semantics are simulated.

The frozen deployed-parent header has SHA256
`131bfc255e0e66bf528f51b759cea529af3c2823c442e3dcf1f6f767d1c75a2c`.
It retains all earlier KDA fusions. The candidate's compiler define requires
row batching, vector reduction, gated-key reuse and the column cache together.

## Validation and handoff

The standalone CPU suite uses:

```bash
pytest --confcutdir=tests/ut/glm_perf -q tests/ut/glm_perf
```

The parent test bootstrap imports unavailable upstream attention packages in
this host environment; this command exercises the isolated GLM suite directly.
The executable tests are host emulation, not a claim of NPU or model quality.

[SUMMARY.json](SUMMARY.json) identifies flags and frozen source hashes.
`frozen-sources.tar.gz` preserves the exact parent/candidate headers and expert
source without formatting or identifier edits to the archived SDK sources.
[RUNBOOK.md](RUNBOOK.md) gives the later hardware gate and installation order.
[README.zh.md](README.zh.md) contains the Chinese summary. Full-repository
format results and scoped results are saved under `checks/`; inherited failures
are distinguished from changed-file checks.
