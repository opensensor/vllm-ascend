# GLM grouped W2/W4 biased-RINT unpack, 2026-10-03

An isolated Ascend 310P experiment replaced the grouped kernel's per-field
integer mask/FP16 conversion with FP16 biased-RINT quotients and adjacent
quotient subtraction. The default kernel and the live four-rank GLM service
were not changed. The source was the known-good
`/srv/ai/src/glm-w2-rowtile-20261002` header plus only the
`GLM_W2_GROUPED_RINT_UNPACK` guarded change. The final experimental header is
`tmp/glm-rint-isolated/w2_blocked_dequant_matmul_v310.h` (SHA-256
`347e5bf4b15f4ce02e03339061e6b9be5d22e74a1d2e0ad6092fc95db455691b`),
compiled from `/srv/ai/src/glm-w2-rint-20261003` with CANN 9.1 and
`--ops-compile-options -DGLM_W2_GROUPED_RINT_UNPACK`.

Three variants were tested, always in separate OPP processes:

| Variant | Result | W4 8-row distributed median |
| --- | --- | ---: |
| Baseline | Known-good | 2.091 ms |
| v1, RINT only | Wrong W4 outputs; 15/30 full-output cases failed | 1.895 ms |
| v2, wait for *all* preceding vector work before MTE2 | Column parity restored, but slower | 2.493 ms |
| v3, release packed-byte UB immediately after its Cast | 30/30 full-output cases bitwise equal | 1.903 ms |

The unsafe v1 produced intermittent 16-output W4 blocks whose upper nibble
matched a tile 64 bytes away. A uniform-across-K byte probe missed this; a
varying-K, one-hot probe found it. The same probe passed on the baseline and
v3. This is consistent with reuse of the packed-byte UB before its preceding
vector read completed. v3 places a V-to-MTE2 event immediately after that
read, allowing the remaining unpack/dequant vector work to overlap the next
DMA. The new 310P regression test
`tests/e2e/nightly/310p/single_node/ops/test_w2_grouped_byte_reuse_310.py`
passed 2/2 on v3. The exhaustive first-byte W2/W4 probe passed all 256 byte
values for each field on baseline and candidate; the varying-K probe was
necessary to expose the hazard. The regression test failed 2/2 on the unsafe
v1 binary, both at K-column 1, confirming it distinguishes the two builds.

The full sweep used 4096-output W4 gate/up and W2 down shapes; 8, 32, and
416 routed rows; distributed, repeated, singleton, mixed, and peer-owned
routes; NZ-packed codes; identical seeds, two warmups, five synchronized
repetitions; and separate processes/OPP roots. `tools.glm_perf.operator_bench`
verified packed-code/input/scale hashes and all FP16 outputs with zero
tolerance. The 30-case comparison is at
`/home/matteius/experiments/glm-gate-a-20261002/w2-rint-v3-comparison-nzpacked-20261003.json`
on the NPU host. Baseline NZ grouped object SHA-256 was
`fd115612355c0dce96983d302806c0422a68a84c9e0a196c5de8464c6118ce10`;
v3 was `8f0151af4b43ff19747f746007d42b6c1f48f217124bd20b06ae9f48fa8e0236`.

| Projection | Cases | Aggregate of case medians | Median case speedup | Range |
| --- | ---: | ---: | ---: | ---: |
| W4 gate/up | 15 | 8.66% faster | 8.80% faster | 3.36–9.64% faster |
| W2 down | 15 | 0.72% faster | 0.51% faster | 1.77% slower to 1.57% faster |

Decision: **revise, not promote**. Although v3 passes exact operator parity,
it misses the GLM performance plan's 15% isolated projection threshold and
therefore did not justify a disruptive four-rank serving restart or a decode
throughput claim. The next grouped-kernel candidate must attack decoded-weight
GM traffic rather than only scalar/vector unpack. The live GLM service retains
the known-good OPP.

The experiment also exposed a build-cache false HIT: the grouped operator did
not declare its included standalone W2 header as a source dependency. The
scoped fix is signed commit `a48a30f9e`; after it, v2 and v3 shared-header
edits each generated a fresh grouped build-cache key and binary. The 8
host-side packed-field/dependency tests passed. No unrelated dirty worktree
changes were staged.
