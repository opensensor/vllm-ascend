# Grouped W2/W3/W4 decode-table reuse candidate

Date: 2026-10-04. This is an **unpromoted** kernel candidate. Isolated control
and candidate OPP packages were compiled and measured on one 310P. There is
still no real-model token/s result for this specific change.

The grouped operator constructs one `W2BlockedDequantMatmulV310Cube` object
and calls `InitGeometry` for each active expert. The tiler fixes N, K, code
width, and canonical versus NZ packing across that entire invocation. Until
this candidate, the default GM path called `Process()` with fresh table
preparation for every active expert. It now passes `decodeTablesReady` after
the first active expert, reusing the already allocated UB pointers and
packing-specific constant tables. Empty experts do not set the flag.

The relevant persistent data are the W2/W4 sign masks and optional canonical
gather offsets, the W3 sign masks, and the canonical W3 byte/gather offsets.
NZ W3 creates field masks inside each tile decode and does not need a
persistent field-mask table. The Cube epilogue's UB reservation ends before
the decode scratch and table area. The existing rejected singleton-L1
candidate remains behind `GLM_W2_GROUPED_CANDIDATE`; this change does not
enable it.

The added hardware regression covers W2, W3, and W4; canonical and NZ
packing; an initially empty expert; an empty expert between active groups;
repeated routes; and peer-owned rows. Two further W3 cases use real GLM gate
geometry: groups above 32 rows and 16 active experts. Every active group
must match the standalone operator bitwise. The standalone OpDef does not
accept NZ-packed `int8` codes, so the NZ-packed grouped result is compared
against the *same signed weights* in canonical `uint8` packing.

The first isolated 310P gate below completed. The remaining promotion gate
would compare 1/4/32-route decode and 512/1280-token prefill against the
baseline, followed by matched real-weight graph serving, Gate-A, and context
checks. NZ-packed serving may gain little: it only avoids a few vector
`Duplicate` operations per active expert, whereas the full dequant/GM/Cube
work remains.

## Isolated package and probe preparation

The October 4 W3-capable control source is
`/srv/ai/src/build-only-glm-w3-nz-csrc-20261004` on Threadripper. Its shared
kernel header SHA-256 is `213cddf5fe011296691e4111648cf6cdebe56bb00d15b4d814e7e09b2fcd7d82`;
its grouped source SHA-256 is
`66f4a10858b5205e778c771784405adb966b1cfcff5cebc87878079dba4a4c5a`.
The candidate is a separate source copy at
`/srv/ai/src/glm-table-reuse-20261004-NkQLIb` with the **same** shared-header
hash and grouped-source hash
`b2b907359ce7d46e8a81894055dece0bfe807462e8f91c96cf6362e32887c3c1`.
Its only kernel-source difference from the control is passing the existing
`decodeTablesReady` flag through the default grouped GM path. The control OPP
was installed only in
`/srv/ai/src/glm-table-clean-control-opp-20261004-KAUf0b/vendors/custom_transformer`.
The candidate OPP target is
`/srv/ai/src/glm-table-candidate-opp-20261004-TjCugB/vendors/custom_transformer`.
These isolated OPP directories do not replace the system package or start a
model server.

The first candidate build command accidentally excluded `build.sh` while
copying source, so it failed immediately and produced no package. That was
repaired. An earlier purported control package had the candidate source due
to a shared staging update; it was discarded as a control and rebuilt from
the exact frozen source above. The standalone probe is
`tools/glm_perf/probe_grouped_table_reuse_310.py` in this checkout, staged on
Threadripper at
`/srv/ai/src/glm-table-reuse-20261004-NkQLIb/probe_grouped_table_reuse_310.py`.
Its `preflight` and `compare` subcommands do not access NPUs; only the
explicit `run` subcommand does. CPU unit tests verify that its canonical and
NZ W2/W3/W4 bytes match the model's packers. The probe hashes actual
`uint8`/`int8` grouped binaries, the unchanged standalone binaries, and the
C++ binding, then checks all active experts bitwise against canonical
standalone calls.

Use this binding library for both packages:

```text
/srv/ai/src/glm-selective-w3-20261004/vllm_ascend/vllm_ascend_C.cpython-312-x86_64-linux-gnu.so
```

The control and candidate were run in separate Python processes with one
`ASCEND_CUSTOM_OPP_PATH` each after confirming the device was idle. All 16
cases per package passed. The binding and standalone binary hashes matched;
grouped source and object hashes differed; paired input/output hashes matched.

Selected synchronized medians (one warmup, three measurements per case;
unpaired-order microchecks, so small differences may be noise):

| Case | Control | Candidate | Change |
| --- | ---: | ---: | ---: |
| W3 2048×4096, 16 active experts, canonical | 41.026 ms | 39.913 ms | 2.7% faster |
| W3 2048×4096, 16 active experts, NZ | 8.869 ms | 8.878 ms | Flat |
| W3 2048×4096, >32 rows/group, NZ | 1.330 ms | 1.316 ms | ~1% faster |
| W3 256×256, middle empty, canonical | 0.578 ms | 0.411 ms | 29% faster |

The serving path uses NZ-packed codes. Its realistic W3 cases do not show a
material gain, so this delta is **not promoted**. It is distinct from the
earlier packed-NZ W3 optimization, whose large gain is documented in the
two-card context study. Raw device JSONs are on the NPU host under
`/home/matteius/experiments/glm-gate-a-20261002/` as
`grouped-table-{clean-control,candidate}-16cases-20261004.json`. A stronger
follow-up reran both packages with canonical standalone parity for the NZ
cases too; its JSONs are `grouped-table-{control,candidate}-standalone-parity-20261004.json`.
The repository's hardware regression then passed **14/14** on device 0 against
the isolated candidate package. The local probe unit tests passed 5/5 and
targeted Ruff lint/format checks passed. These are operator checks only; a
matched full-model graph/quality gate would still be
required before promotion if this candidate is revisited.
