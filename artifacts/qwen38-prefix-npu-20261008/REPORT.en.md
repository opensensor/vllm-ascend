# Qwen bounded-prefix NPU qualification and image serving

Current status: sustained use later reached the 96°C watchdog cutoff at
23:37:55 UTC. Restart is deferred by the operator. The short tests below remain
historical functional results, not sustained thermal qualification. See the
[thermal incident and 94/85 hold report](../qwen38-thermal-incident-20261008/REPORT.en.md).

## Initial 1,024-token profile

The operator re-granted NPU access on October 8. A complete qualified recovery
snapshot was copied to `/srv/ai/src/qwen38-prefix-bounded-runtime-20261008`.
The signed bounded-checkpoint scheduler and tier helper were backported with
only the matching `_update_states` method and import in the runner. The four
qualified model source files remained byte-identical. The pinned vLLM source
is `3ab5dda29`; no dependency upgrades were made.

Real W4 weights ran on four 310P devices with TP4/EP4, MTP2, decode graphs
`[3,9]`, three request slots, 262,144 maximum context, one image per prompt,
built-in FP16 SwiGLU and CANN finalization. FlashComm1 remained disabled in the
qualified 310P profile. Dummy weights were not used. The serving endpoint was
port 8001; Kilo configuration was not changed.

## Validation results

All 78 focused host regression checks passed. All scoped pre-commit hooks
passed for the changed code, scripts and runbooks. `bash format.sh ci` ran in
an isolated worktree and failed on existing unrelated Ruff, codespell, typos,
clang-format, markdownlint and forbidden-import issues. Its automatic edits
were confined to that worktree. See `host-regression.log`, `scoped-format.log`
and `full-format-ci-complete.log`.
Raw terminal/lint records retain their original spelling and carriage returns.
Codespell and typos flag copied historical lint diagnostics and opaque IDs in
those records; only these two text-spelling hooks are skipped for raw evidence.
Both hooks passed on every changed source file and report.

Real-model text/tool smoke scored 6/7, matching the earlier selected baseline.
The known failure remains: reversing ASCEND returns DNESCA instead of DNECSA.
Five other text cases and the tool call passed. Fresh red, blue and DEMO42 OCR
images all passed. These checks do not establish broad model accuracy.

The exclusively owned server was drained and both scheduler and worker caches
were cleared for a diagnostic pressure comparison. Each rank used its actual
loaded model checkpoint tensors and 96 rounds of three changing state IDs.
This exceeds the initial device-tier capacity. Both policies preserved latest
values and all primary/archive/swap storage addresses. The baseline also
restored the oldest spilled value correctly. All four ranks acknowledged.

| Diagnostic policy | Spill count across ranks/groups | Restore count | Slowest timed rank |
| --- | ---: | ---: | ---: |
| Baseline retention | 495 | 12 | 2.187 s |
| Bounded retirement | 0 | 0 | 0.739 s |

Each group checkpoint is 9,744,384 bytes. Baseline spill payload was
4,823,470,080 bytes, or 4.492 GiB. Bounded retention retired 3,096 states.
Elapsed times exclude correctness checks; counters include the oldest-state
restore probe and its associated admissions. This is one ordered diagnostic
pair, with parallel rank acknowledgments, not independent timing repeats or
an LLM throughput benchmark. It does not establish thermal causality.

A cold 256-token prompt generated 256 tokens at 27.412 decode tok/s. Three
independent cold 16,384-token prompts generated 128 tokens each. Server-side
prefill durations were 43.712, 51.023 and 50.422 seconds, excluding queue time,
corresponding to approximately 375, 321 and 325 prompt tok/s. Client TTFTs were
43.735, 94.756 and 142.439 seconds because later requests queued. Their decode
rates were 1.258, 2.388 and 17.707 tok/s; the first two overlapped other cold
prefills. The batch completed in 149.619 seconds with zero additional spills
or restores on every rank and 194 additional retirements per rank across
its groups. Weights and clean graph status were preserved.

The 16K repeat batch was interrupted at the operator's request to release the
server for interactive use. It is not a completed repeat-prefix qualification.
The separate mixed-image/decode probe was prepared but not run. The maximum
recorded temperature in the initial phase was 80°C, with no thermal shutdown;
the existing 96°C watchdog remained active. This short run does not qualify
sustained thermal operation or three full 256K sessions.

## Historical comparison and requested switch

The saved native HC residual trial's three 8,192-token cold samples had a
19.973-second median (about 410 prompt tok/s). The later 23,410-token residual
control recorded a 59.400-second median (about 394 prompt tok/s). The initial
image recovery profile is therefore not established as the best long-prefill
variant. Prompt lengths, concurrency and starting conditions differ.

Historical raw sources are recorded in `historical-comparison.json`.
The operator requested a live switch closer to that configuration while keeping
images. The replacement uses a 2,560-token scheduler batch and native HC
residual, retaining three slots, MTP2, `[3,9]`, bounded checkpoints and images.
Changing the scheduler batch required new startup-sized buffers and one engine
restart; native HC selection uses the existing resident load/switch/recapture
transaction after startup. Full graph/KV-budget hot reconfiguration remains
unimplemented. Final qualification and service state are recorded below.

## Final image-enabled profile

The replacement loaded successfully with a 2,560-token scheduler batch and
selected `native_hc_residual` through resident controls. The native library and
kernel matched their qualified SHA256 manifests. All four workers passed the
native value probe, retained their PIDs and weight-storage digests through the
selection, recaptured both graphs and returned clean graph status. An initial
controller attempt rejected a non-UUID generation string before changing Python
dispatch; it was corrected and the existing native load was reused safely.

The final real-weight smoke again scored 6/7 with the same known reversal
failure. Red, blue and OCR image checks again passed 3/3. The exact-token
23,410 prompt uses the historical `qwen-latest-matched-23410-v1` label.
Server prefill was **59.688 s**, approximately **392.2 prompt tok/s**, with zero
cached tokens; client TTFT was **59.716 s**. The immediate repeat hit **23,296
cached tokens**, with server prefill **1.900 s** and client TTFT **1.917 s**.
This is close to the saved 59.400-second cold median, not an established new
speed record. Cold/repeat output hashes differ; this is not an exact output
parity gate. Historical unchanged-runtime trials also recorded text drift.

Across this cold/repeat pair every rank recorded zero additional spills and
restores, 49 retirements across its groups, clean graphs and unchanged weights.
Maximum recorded temperature in this phase was **76°C**; end-of-test readings
were 74/73/76/75°C. There was no thermal shutdown during these short checks.
This does not qualify sustained thermal operation, full-window quality,
three concurrent 256K sessions, or concurrent cold-prefill throughput for the
final profile. No further load is submitted after the operator handoff.

The server is left running and unpaused for the operator at
`http://192.168.53.187:8001/v1`, model ID
`qwen38-prefix-bounded-validation`. The watchdog remains at 96°C.
The historical test scripts target an exclusively owned test server; do not
run their cache-clearing diagnostics on a server now used interactively.

Reproduction uses `start-fast-image.sh`, followed by `select-hc.py` to load
native HC, then the image/smoke and cold-prefix checks in `validate.py`.
Primary receipts: `fast-image-results.json`, `fast-prefill.json`,
`fast-image-hc-switch.json`, `fast-image-native-load.json`,
`fast-image-vision.json`, `fast-image-smoke.json`, `runtime-provenance.json`
and the two thermal records. Code changes in this delivery are the diagnostic
worker RPC, its four cleanup/pressure tests, bilingual runbooks, reproducible
launch/selection scripts and validation records. Existing unrelated experiments
remain outside this delivery. The HF image capability cards published earlier
remain unchanged in this run.
