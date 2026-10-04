# GLM grouped scale-barrier removal, 2026-10-03

This isolated Ascend 310P candidate removed the vector `PipeBarrier` after
each `Muls` of disjoint NZ fragments in the grouped W2/W4 dequantizer. It
used the known-good 128-K packed layout and the same compile options as the
live combined package. The live TP4 server was not changed.

The source is at `/srv/ai/src/glm-w2-scale-barrier-20261003` on the NPU host
and under `tmp/glm-scale-barrier-isolated/` locally. The NZ-packed grouped
object SHA-256 is
`2c3235424a2a163505a8dd4c6a3c14196ae5f3a241cd9619cf7378d48540b3e3`.
It was built with `--opkernel` and tested in an isolated OPP copy.

The matched one-NPU decode sweep covered W4 gate/up and W2 down projections,
8 and 32 routed rows, five route patterns, two warmups, and seven timed
repetitions. All **20/20** FP16 outputs were bitwise equal to the known-good
binary. The sum of case medians fell from **93.306 to 92.141 ms (1.25%)**;
19 of 20 individual cases were faster.

Decision: **reject for serving**. The gain is below the GLM performance
plan's isolated-projection threshold and does not justify a four-rank model
restart. It remains an optional follow-up to a larger, independently
qualified kernel change.
