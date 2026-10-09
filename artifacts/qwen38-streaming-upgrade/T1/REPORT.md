# T1: reference and evidence protocol

Offline reference preparation is complete. Live reference identity and hardware
qualification remain incomplete. No server was queried or changed, no SSH command
was run, and no device runtime, library, allocation or kernel was loaded.

The append-only [reference receipt](reference.json) freezes 2,547 committed source
blobs at `2a2a4e4161e5c513b18055c81e9710fc8fad5128`. Dirty and untracked overlays
are excluded. The CATLASS submodule is recorded as a pinned Git commit with its
external bytes explicitly pending, rather than inventing a content SHA256.
Historical partial runtime provenance, launcher defaults, deferred thermal profile,
thermal incident, and native host build receipt are archived and hashed separately.
The historical serving profile is **thermally unqualified**; the live runtime is
**unknown**. The staged `qwen_transfer_v1` binaries are host compiled only.

There are 132 workload definitions: 11 text/tool/image/mixed cold/warm cases,
four submitted-request levels, and MTP0/1/2. C4 includes queuing under the fixed
three-active-slot capacity. Identical token/image/conversation payloads are required
for warm controls. Generation limits and MTP graph shape are fixed per case.
The 65,536-token long case is a planned benchmark size, not a claimed measurement.
Exact token IDs, real tool conversations and fresh image content are unavailable
locally and explicitly pending. Checkpoint content and the coherent serving
OPP/bridge/binary bundle hashes also remain pending. The unrelated archived W8A8
checkpoint manifest was not used to identify the W4 reference.

Schema validators reject source/hash changes, mixed arithmetic, incomplete ranks,
changed namespaces, missing pending-identity keys, changed capacity, mismatched
warm controls, and cached work labeled cold. CPU results cannot be marked passed
hardware gates. Candidate assets, native binaries, bridges, memory contract, native
component gates and every rank are bound to immutable hashes before live admission.
Every new candidate source/build hash invalidates dependent evidence.

Validation: 35 CPU tests passed using `pytest --noconftest`; scoped Ruff passed.
The first test run exposed a Git submodule enumeration issue; the corrected
implementation records Gitlinks separately, and the final suite passes.
The initial corrected 28-case run is retained in `tests.log`; expanded final
validation is in [tests-final.log](tests-final.log).

The validators check evidence structure and immutable identity; they do not
authenticate the truth of an artifact or substitute for actual byte verification,
numerical tests, ABBA timing, sustained thermal measurements, or NPU ordering tests.
Later builders/controllers must verify actual candidate bytes and gate artifacts.
Full live admission is deliberately refused with the present pending identities.

## APIs and later integration

`tools/qwen4exp/streaming_protocol.py` provides data-only canonical SHA256,
file SHA256, source identity, manifest sealing, reference preparation, manifest and
saved evidence validation. `write_append_only` uses exclusive creation to prevent
overwriting a receipt. It does not promise crash-atomic whole-file visibility;
consumers must parse and validate the complete sealed JSON.

Prepare a distinct output with:

```bash
python -m tools.qwen4exp.streaming_protocol --root . \
  --revision 2a2a4e416 --output /path/to/new-reference.json
```

This command only reads committed local Git objects and creates the named receipt.
It does not resolve pending checkpoint, runtime, token or image identities.
