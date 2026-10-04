# GLM 512-wide paged QSA Cube candidate, 2026-10-04

Status: one-card parity and graph replay passed; four-rank 192K admission,
2K/8K retrieval, and decode gates passed. The strict answer suite retained the
accepted 17/20 result. The checked-in one-card regression passed. The
candidate is running on port 8001.

The packed-W3 192K server spent 180.49 s on a correct 7,877-token retrieval.
A fresh four-rank Torch/CANN profile captured its first two 640-token prefill
steps. Their server times were 7.95 and 12.75 s; the all-rank device-task
envelope was 20.70 s. On rank 0, eleven `QsaSparseAttentionV310` calls with
input query `[640,16,512]` took **5.162 s**, 26.6% of summed device task time.
The 512-wide GLM latent query used the existing vector QSA kernel; the main
source's Cube QSA path only selected 256-wide queries. The next largest rank-0
task categories were grouped W2/W3/W4 projections (5.334 s) and KDA plus
convolution (3.449 s). Task sums are attribution, not an additive critical
path. Raw profile and all-rank summary are under
`/home/matteius/experiments/glm-w3-20261004/profile-w3-prefill640-20261004`
and `profile-w3-prefill640-summary-20261004.json` on Threadripper.

Main now keeps the existing 256-wide Cube instantiation and adds a 512-wide
instantiation using the same algorithm and separate compile-time buffers.
The 512 path is selected only when the existing head-count and group-width
bounds permit it. The isolated QSA package is
`/srv/ai/src/build-only-glm-w3-nz-csrc-20261004/opp-qsa-cube512-candidate`,
built from main's QSA files and installed without replacing the current
`glm-qsa-padded-20260930/opp` package. Installer SHA-256:
`277d91f6ac1fa44bc901958708f5180b9fc0aeb8c3e7e0db5122b51f52bfb38f`.

One-card synchronized operator probes used the same 512-wide queries and
cache layout in separate processes with the control and candidate OPPs:

| Case | Vector control | 512-wide Cube | Change |
| --- | ---: | ---: | ---: |
| Dense continued prefill, 640 queries | 470.044 ms | 63.131 ms | 7.45x faster |
| Sparse selected pages, 16 queries | 19.527 ms | 2.750 ms | 7.10x faster |

The dense output had maximum absolute difference 7.63e-6 from the vector
control; the sparse output differed by at most 3.81e-6. Both were finite.
A separate 35-token FP32 reference check passed at the GLM width, and
changing-query/visible-length graph replay matched eager output bitwise.
The probe is `tools/glm_perf/probe_qsa_cube512_310.py`; JSON and tensor
records are in `/home/matteius/experiments/glm-w3-20261004` on Threadripper.
These operator timings motivated a full-model matched run.

The four-rank candidate retains the 640-token scheduler chunk, 192K configured
window, packed W3 OPP, and graph settings. It enables the opt-in histogram
route count and swaps only the QSA OPP. Startup admitted the same 239,781
cacheable tokens as the 640/compare control, 1.22 times the configured window.
The calibrated 2K retrieval returned the exact secret code in 22.85 s TTFT.
The matched 8K prompt had 7,877 served tokens and returned the exact code;
TTFT fell from **180.49 s** with vector QSA and comparison route counting to
**99.04 s** with Cube512 QSA and histogram counting, a 45.1% reduction in one
run. The route-count microcheck saved about 0.025 ms per 640-token count, so
the QSA change is the likely main contributor, but this full-model pair did
not isolate the two changes. Client records are in
`artifacts/glm-quant-context-2card-20261003/packed-w3-paired/`.

The strict 20-case answer suite scored **17/20**, with the same misses as the
earlier graph W3 run: `instr_reverse`, `instr_first`, and `code_slice`.
The 256-token short suite completed all five requests without early EOS:

| Metric | Earlier graph W3 | Cube512 + histogram | Change |
| --- | ---: | ---: | ---: |
| c1 decode | 3.550 tok/s | 3.548 tok/s | flat |
| c4 aggregate decode | 7.556 tok/s | 7.939 tok/s | +5.1% |

The checked-in `test_glm_qsa_cube512_310.py` passed 1/1 on device 0 while the
server was idle. It checks the 512-wide output against an FP32 reference and
checks changing-input graph replay bitwise against eager output. The server
remained healthy after the test.
A full-length 192K prompt has not been tested.
