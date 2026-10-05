# October 5 publication validation

This publication contains the Qwen batching changes, trace figures and raw
selected exports, isolated sparse-gather experiments, resident extension,
cast-reuse candidate, and the shared resident-controller dependencies. It does
not enable the experimental sparse gather or cast reuse in the serving default.
The 2,560-token chunk and CANN finalizer remain explicit model overrides.

## Gates completed

- 175 focused host tests passed, both in shared main and in a snapshot of the
  staged publication. Coverage includes native-W4 grouped dispatch, chunk
  bounds, QSA configuration, trace rendering, cast fallback/parity, resident
  transaction ordering/restoration, and workload streaming.
- All configured pre-commit manual hooks passed on the publication files in
  that staged snapshot, including Python/C++ formatting, Markdown lint,
  shell lint, symbolic meta checks and secret scanning.
- Earlier 310P gates passed the three 25,600-route capacity cases and real
  one-layer output parity. The separately named sparse gather passed all 32
  NPU correctness cases after correcting its isolated package metadata.
- TP4/EP4 startup captured both decode graphs and reported 1,068,936 cache
  tokens, with 4.08x planner concurrency at 262,144 tokens. Short generation
  and all four resident status RPCs passed.
- The matched 23,410-token prefill replay reported zero cached tokens and
  identical output. Resident control mutation/recapture on Qwen and cast-reuse
  NPU performance remain unqualified.

## Broader gate limits

The requested repository-wide `bash format.sh ci` was run in a temporary
snapshot of the staged index. It found existing errors in unrelated tracked
files, including undefined Python names, spelling checks of raw exports,
Markdown fencing, and forbidden imports. Formatting was copied back only for
the publication files, leaving concurrent working-tree edits intact.

The full `python -m pytest tests/ --maxfail=1 -q` invocation stopped during
collection because this host does not have `modelscope`. No complete test-suite
pass is claimed. The focused host and recorded NPU gates above cover this
publication; they do not establish broader model quality, sustained thermal
performance, or four full simultaneous context windows.

The secret scanner classified two seeded synthetic QSA K-tensor SHA-256 values
as API keys. The added allowlist is restricted to their exact digests, the
specific experiment JSONL directory, and that detector rule. Measured values
and raw export contents are retained. Published experimental C++ snapshots
were formatted; the installed source/binary hashes in the experiment notes
describe the packages actually measured.
