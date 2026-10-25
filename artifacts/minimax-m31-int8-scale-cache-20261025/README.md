# INT8 output-tile scale-row reuse

`minimax_w4_int8_v14` retains four scale rows in local scratch through all K
slices of an output tile. The old path loaded a complete scale row for every
N16 slice at every K1024 boundary. The build enables the existing scale-cache
allocation and supplies the correct cached-row offset to the INT8 decoder,
including its debug path. Arithmetic and packed-byte layout are unchanged.

All twelve real-weight projection comparisons against v13 matched exactly.
Isolated speed gains were small and mixed. The end-to-end serving measurements
were 3.697570, 3.699850 and 3.691333 tokens/s after first output, compared with
3.594553 median for v13 with INT32 routing. All three messages matched exactly.
The candidate is retained, with partial graphs and the fused activation.

`manifest.json` and `build-options.json` identify the sealed remote build.
Build with `--wide-l1 --rint-unpack --int8-compute --scale-cache`. The loader
validated the native kernel on all six workers before selecting it. Worker
PIDs and weight storage were unchanged; the native-load transaction cleared
request caches. No full 160K prompt or broad quality evaluation was run.

`activate-native.sh` is the installed fresh-worker post-start script. It still
uses the v13 layout controller to repack weights, then selects v14, activation
v2, the installed Python module and partial graphs. Never rerun the whole
startup script on already-repacked workers. The CPU suite passed 223 tests
with 24 hardware skips. Raw controller/probe snapshots preserve executed text.
