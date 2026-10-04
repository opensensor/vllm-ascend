# GLM grouped W2/W4 128-channel L1 streaming trial, 2026-10-03

This isolated trial kept a decoded 128-output-channel by 1024-K FP16 B tile
in L1 and accumulated successive K chunks in L0C. It was enabled only for
NZ-packed experts with two to four local routes and at most 32 total routes;
the singleton and larger-group paths retained the known-good GM workspace
schedule. The live four-rank GLM server was not changed.

The candidate source is in the NPU host snapshot
`/srv/ai/src/glm-w2-l1stream-20261003` and local temporary copies under
`tmp/glm-l1stream-isolated/`. Header SHA-256 is
`fbaf97caf4636466f961bf8be3434d80f79d445fbdb5d0f9777a7424b9b44038`;
grouped kernel SHA-256 is
`fc65b458993235f76ae6c7c992382173547780622d8a101abc5257ea9bbb0a97`.
It was compiled with `-DGLM_W2_SCALE_PAIR` and
`-DGLM_W2_GROUPED_RINT_UNPACK`; the NZ-packed candidate object SHA-256 is
`2bdc63085175dcb5d795950c896d39b1822ffe1cceddf3926d1eaa9aa7419edf`.
For the hardware sweep, only the two grouped objects and metadata were
substituted into an **isolated copy** of the known-good OPP, `opp-dev`. The
known-good package and live service were unchanged. A full candidate package
also built successfully but was not used in this sweep.

The matched CANN 9.1, one-310P NZ-packed operator sweep used the full
4096-output W4 gate/up and W2 down shapes, 8/32 routed rows, five route
patterns, seed `20260930`, two warmups and seven synchronized repetitions.
It compared against the saved known-good combined-package run with the same
options and binary hash `b211354d0b47407722a5805f974bfeb1722d07d252c0adcf830f60f2ca5e1de3`.
All **20/20** FP16 outputs were bitwise identical, and the candidate object
was rehashed after measurement.

| Scope | Known-good median | L1-stream median | Result |
| --- | ---: | ---: | ---: |
| Sum of 20 case medians | 93.306 ms | 95.371 ms | 2.21% slower |
| W4, 32-route distributed | 6.471 ms | 7.360 ms | 13.7% slower |
| W4, 32-route peer-owned | 6.426 ms | 7.343 ms | 14.3% slower |
| W2, 8-route distributed | 1.202 ms | 1.136 ms | 5.5% faster |

Decision: **reject**. Correctness passed, but eliminating this GM round trip
with sequential L1 K chunks does not overcome the extra dequant/Cube staging
cost for the primary W4 decode cases. This supports focusing the next
candidate on the vector dequant path rather than promoting this L1 schedule.
The complete hardware records are on the NPU host at
`/home/matteius/experiments/glm-gate-a-20261002/w2-l1stream-dev-nzpacked-20261003.pt`
and `w2-l1stream-dev-comparison-20261003.json` in the same directory.
