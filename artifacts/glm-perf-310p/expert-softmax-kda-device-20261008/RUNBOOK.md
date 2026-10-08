# Build, qualify and retain the GLM candidates

Use the existing FP16 scale checkpoint and serving SDK. Reuse the complete
parent build options from the archived build protocol; the flags below are
additions, not substitutes for the existing fused pipeline.

1. Build unique append-only expert versions: v992 adds `--product-pipe-events`
   to decode v984; v993 adds it to prefill v985; v994 adds `--bulk-route-store`
   to v985; v995 adds both. Keep FP16 scales, prepared offsets and existing
   route, activation and specialized W4 options.
2. Build QSA v996 using `--vector-output --vector-accumulate
   --softmax-head-batch`, preserving the shared-cache parent schedule.
3. Stage the qualified KDA row-batch package with `stage_kda_score_matrix.py`.
   Compile with its six previous defines plus `GLM_KDA_SCORE_MATRIX_BATCH`.
   Admit the exact parent SHA listed in SUMMARY.json; do not substitute an
   unqualified package in the same OPP slot.
4. Reproduce the archived free-device gates using the serving interpreter and
   full parent environment. Check exact outputs and changed graph replays,
   then bounded resident admission on all four ranks.
5. Before switching, verify running/waiting request counts are zero. Use the
   localhost resident controller, pause, apply guarded replacements, retire
   graphs and capture all existing sizes. Keep worker and weight digests
   unchanged during the matched comparison. Resume only after all ranks pass.
6. Reproduce `glm-next-three-live-bench-v7.py` from the final receipt archive.
   It selects decode v984, prefill v995 and QSA v991 first in fresh workers,
   retaining the new KDA package. It warms the workers, clears
   prefix state and accounts for complete streams. Compare against the parent
   records in v5; v5 failed its later switch, so this is process-separated.
   The fully combined v6 candidate regressed C1, so keep the v7 prefill selection
   active and save final status. Repeated source transitions
   failed capture allocation at this reservation despite scratch cleanup.
7. For recovery, use the identity-checked launch/controller protocol from the
   archive on the owned idle or failed process tree. Keep the same launch
   command, checkpoint and capacity. Read `/v1/models`, execute a real text
   request, check all ranks and ensure `/is_paused` is false before handing
   the server back.

The captured final launch is [serving-config.json](serving-config.json).
The API is `http://192.168.53.187:8001/v1`; administrative calls remain local.
The final logfile is `/srv/ai/artifacts/glm-prefill-weight-reuse-20261007/
next-three-prefill-server-20261008.log` (one continuous path).

For CPU verification:

```bash
pytest --confcutdir=tests/ut/glm_perf -q tests/ut/glm_perf
```

Current defaults remain off for new experiment flags. Production adoption
requires the explicit qualified build and selection, not an implicit runtime
weight conversion. The transient cleanup helper must run only after graph
retirement while idle; it does not establish complete graph-pool reclamation.
