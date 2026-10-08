# Live dense-projection probe and decode trace

The server retained its weights, worker PIDs, completed-pool prefill candidate,
MTP1, and 311040 configured context throughout these two completed checks.

## Dense projections: rejected

Real KDA `f_b_proj`, `g_b_proj`, and `o_proj` weights were already NZ format.
The graph probe compared existing dispatch with grouped NZ matmul at rows
2, 8, and 640 on all four ranks. Outputs matched exactly.

Gate projections were approximately flat at decode (~0.03 ms); at 640 rows
the alternative was slower (~0.038 to 0.057 ms). Output projections were
slower at decode (~0.063–0.070 to 0.123 ms) and approximately flat at 640 rows
(~0.329 to 0.322 ms). No dispatch change was promoted. Temporary transpose
copies were released. Details are in `probe-results.json`.

## Fresh decode trace

A single 32-token request was traced, then the completed-pool graph candidate
was restored and resumed. No baseline serving sweep was run. Request result
is `profile-request.json`; restoration receipt is `profile-restored.json`.

Raw four-rank traces live remotely under
`/home/matteius/experiments/glm-projection-nz-resident-20261005/trace`.
`decode-task-summary.json` aggregates from the first recurrent KDA task until
trace end. This is an approximate decode interval, not exact per-step ranges.
Task durations may overlap and must not be interpreted as additive latency
savings.

All ranks report zero native Sinkhorn tasks and 65120 ReduceSum tasks with
65120 associated MemSet tasks. This motivates the neighboring normalization
kernel study. Packed expert projection remains the largest individual task
family. Native Sinkhorn has **not** been enabled by this study.
