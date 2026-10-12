# Josh / WLTechBlog: external single-card deployment

Source: the user-provided deployment report received October 12, 2026.
This is externally reported evidence; OpenSensor did not run these measurements
or inspect Josh's final filesystem. The runtime had local patches and SDK
backfills, so retain those differences alongside the successful results.

## Reported environment and result

- Ubuntu 26.04, kernel 7.0.0-38-generic, i5-13400, 128 GB host RAM.
- One Atlas 300I Duo: two Ascend 310P3 chips; driver 26.0.rc1.
- Original Qwen3.8 Flash-Next W4 G128 checkpoint, 1,610 shards.
- October 11 runtime archive: all 5,274 recorded files verified before patches.
- Live service on port 8002: TP2, configured 131,072 tokens, MTP2 and
  FULL_DECODE_ONLY graphs.
- Reported effective generation rates approximately 17–26 tokens/s; a creative
  generation completed 630 tokens in 36.1 seconds, about 17.4 tokens/s.
- Six functional checks passed, covering arithmetic, creative text, reasoning,
  Python generation and list recall. These are different checks from the
  original 6/7 smoke; they do not erase its reverse-word failure.
- Reported NPU temperatures 55/56 C. PCIe negotiated x4, despite x16 capability.

A configured 131K limit is not a near-limit recall or generation test. The
original qualification includes those probes; Josh's report does not include
an independent long-window probe, first-replay cache counts, native residual
qualification receipt or equivalent three-run decode methodology. Keep his
rates labeled as effective rates rather than combining them into our median.

## Confirmed handoff defects and offline fixes

| Defect | Change |
| --- | --- |
| Generated `_build_info.py` omitted from the archive | A preparation option validates original hashes, adds only `_310P` metadata and records its SHA-256; it refuses to overwrite another hardware build. |
| Runner selection inherited from the host shell | The single-card launcher explicitly selects the v2 runner for its FULL_DECODE_ONLY graph profile. |
| `--show` displayed flags without runtime environment | Preview includes selected device, runner, SDK, library, HCCL and compiler environment fields. |
| DSpark loaded when Triton happened to be installed | The V4.1 drafter import and registration exclude 310P even with Triton present. No fake model classes are added. |
| Optional `fla_npu` imported by unrelated models | Optional operator imports permit an absent backend; invoking that backend raises an explicit diagnostic. Broken transitive dependencies still propagate. |
| Wrong Python stack and partial SDK discovered after loading weights | A no-NPU preflight compares package versions, inspects critical SDK components and checks selected HCCL exports. |

These source changes have CPU regression coverage. They have not completed a
fresh-container or NPU inference qualification. The original October 11 binary
archive remains immutable; updating repository source does not retrofit its
frozen Python files. Josh's working service was not changed.

## Environment lessons

CANN's version string and a total file count do not establish completeness.
Runtime/development component selection and symlinks can change the count.
Josh reported missing ATB, missing HCCL exports, and missing compiler/TBE
components. The new checker inspects critical paths and exports, not all vendor
files, ELF dependencies or runtime allocator behavior. A passing report still
requires device initialization, operator and inference gates.

Prefer an intact matching SDK installation. Josh's container extraction and
HCCL overlay are recorded recovery workarounds; copying selected files across
SDK trees is not a universal installer. The manylinux image tag alone does not
pin the recovered binaries: retain its immutable digest and copied-file hashes.
The report does not supply those values yet.

Josh found HCCL in `LD_PRELOAD` caused allocator/stream failures while solving a
symbol-resolution error. Use the coherent library search path and inspect the
selected HCCL family. The preflight rejects HCCL preloading for this recipe;
it does not erase arbitrary user preload settings.

Single-host loopback bootstrap was reported working with
`HCCL_SOCKET_IFNAME=lo` and `HCCL_IF_IP=127.0.0.1`. It is not a multi-host
recipe. The printed appendix uses the shortened `127.0.1`; use the complete
address in documentation. Its operator variables and library paths contain
placeholders, so the appendix is not an executable replacement launcher.

Use `<venv>/bin/python -m pip`. Copying a virtual environment retains absolute
shebangs and editable-source paths; plain `pip` may modify the original one.
Keep the Triton-free Flash-Next bundle in its own environment. The package
inventory is an observed baseline; harmless extra packages are warnings, while
core ABI/version drift and Triton in this frozen recipe are blocking checks.

## Memory, loading and operational limits

Josh demonstrated serving with 128 GB RAM, with reported cold-load pressure.
Our original host had 247 GiB. These two observations do not establish a
144 GB minimum, nor a minimum amount of free RAM. File-backed page cache is
reclaimable; record anonymous RSS, cache, swap, peak allocations and loading
latency before choosing a capacity rule or a new streaming-loader design.

Reported weight loading was about 156 seconds cold and six seconds warm,
followed by about 3.5 minutes of engine initialization. That difference alone
cannot establish a purely Python-bound loader: page-in and page-cache effects
also need profiling. Keep engine initialization separate from tensor loading.

Stopping only the API process can leave workers holding NPU memory. Manage the
owned process group and check worker cleanup before relaunching. Probe port
8002 consistently. Reasoning consumes the output budget; disable thinking for
deterministic short-answer probes, and inspect finish reason and usage.

ECC-off is a reported local setting, not a default or a measured performance
recommendation. The reported chip index also differs from the single-card
launch IDs. Do not add an ECC/reset command to the launch workflow.

## Remaining work

- Replace production imports from `tests` with installed runtime modules.
- Resolve the plugin/core FastAPI requirement conflict with compatibility tests;
  do not use an unconstrained install as the permanent recipe.
- Validate device-property initialization for the Triton-free rejection path;
  no guessed core counts or silent fallback should be introduced.
- Run a fresh-environment installation gate and publish the complete dependency
  and SDK provenance, including the container digest and build metadata.
- Document host-module persistence and kernel-update rebuilds in the driver
  repository; DKMS remains a separate unqualified deployment path.
- Obtain comparable long-window, cache and native-activation receipts from the
  external deployment before extending the qualification claims.

## Offline validation of this follow-up

The 55 scoped regression cases pass, including optional-backend failures,
310P with Triton present, metadata preparation, package drift and SDK checks.
The checker also ran with the matched Python environment against a separate
pristine extraction on our host and passed after metadata preparation. It
opened no NPU context. The preparation and host preflight receipts are retained.

The host's HCCL exports `HcclGetRootInfo` from its `DT_NEEDED` companion
`libhccl_compat.so`. Checking only the top-level `libhccl.so` would incorrectly
reject that installation. The bounded inspection follows dependency exports
and accepts the supported compiler entry-point layouts (`asc_opc` or `ccec`).
These checks are diagnostics; they do not guarantee ABI compatibility or
reproduce the dynamic loader's full RPATH/interposition behavior.
