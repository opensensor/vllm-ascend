# MiniMax M3.1 tool serving and performance evidence

This continues the real-weight six-chip deployment from `2dfaf9626`.
All timings are diagnostics on the actual 310P host, not benchmark-suite scores.

## Client and protocol

- `kilo-minimax.json` is the sanitized configuration fragment merged into the
  existing `~/.config/kilo/kilo.jsonc`. Existing providers and credentials were
  preserved; the full configuration was not copied into this repository.
- `kilo-report.json`, `kilo-smoke.jsonl`, and `kilo-independent-check.json` record
  a successful Kilo 7.8.17 read/read/edit/bash workflow. The model fixed `a - b`
  to `a + b`, ran the unchanged tests, and passed independent negative, zero and
  fractional checks. This client trial used the initial 8K profile.
- The first trial compacted prematurely because `limit.input` had already
  subtracted the output reserve. Kilo subtracts that reserve itself. The fixed
  definition uses the full context capacity and keeps automatic compaction.
- The pinned vLLM Rust tool-parser extension was built with Cargo's locked
  dependency set. Its six upstream tests passed; the binary hash is recorded.
- Named tool choices intentionally use `finish_reason: stop` in this vLLM
  revision. Auto and required choices use `tool_calls`.
- A JSON-schema check exposed historical thinking closers affecting the engine's
  streaming reasoning predicate. The plugin now scopes both initial and
  streaming checks to the current assistant turn. Nine parser regressions passed
  in the serving environment.
- `final-tools/report.json` records all ten real-model protocol checks passing,
  including normal and streamed JSON-schema answers. The original failure is
  retained as `initial-json-schema-failure.json`.

## Qualified optimizations

`native-v6-experts.json` compares full real expert projections with the original
OPP and independent CPU references. All 21 cases passed and were bit-identical
to OPP. Common routing cases were about 10–13% faster. The wider L1 schedule is
built with `--wide-l1`; defaults remain unchanged for other native builds.

`attention-tile64.json` compares real-layer projections on NPU and CPU. The
64-query tile preserves the original sparse scoring, top-k ordering, masking,
and reductions. `attention-promotion.json` confirms that the installed source
has the same AST as the tested candidate.

The combined implementation passed nine coherence checks and the 7943-token
retrieval. Same-output short timings improved as follows:

| Request | Prior baseline | Combined candidate |
| --- | ---: | ---: |
| Greeting | 8.72 s | 7.35 s |
| France capital | 15.22 s | 13.64 s |
| Addition | 10.99 s | 9.72 s |
| 3443-token retrieval | 71.45 s | 61.19 s |

The 7943-token check used a warm prefix from the preceding retrieval. Its
91.27 s is a correctness receipt, not a cold-speedup comparison.
`baseline-before.json` and `combined-after.json` preserve identical rank, PID,
and weight-storage fingerprints for all six workers. The rejected singleton-L1
v5 build was numerically correct but slower; it was never activated in serving.

## Operation and limitations

The final 16K service passed a cold 16043-token retrieval in 299.90 s, returning
`amber cloud`. Both scheduler prefix state and physical caches were cleared
first. `final-before-16k.json` and `final-workers.json` preserve all six worker
identities and weight-storage fingerprints. The planner reports the same
1.75 GiB available KV budget as the initial 8K configuration (58,368 token slots).
Kilo's context/input limits are now 16384, with a 2048-token output reserve.
The provider timeout is 30 minutes, with a 10-minute chunk timeout to accommodate
cold prefill and this host's roughly two-token-per-second decoding.

The persistent `minimax-m31.service` starts the direct `vllm serve` launcher and
uses `activate-native.sh` to restore the sealed native v6 dispatch after workers
are ready. It is installed without enabling boot startup. Python/CANN swaps
preserve weights; API/parser and context-allocation changes require startup.

The profile remains eager, TP2/PP3, EP2 within each stage, one sequence, and one
448-pixel image. ACLGraph, sequence-parallel MoE, speculation and video remain
unsupported in this adapter. These bounded checks do not establish broad model
accuracy or parity with the original FP4 release.

`unit-tests.txt` and `changed-file-lint.txt` record focused validation.
Repository-wide `format.sh ci` still has unrelated failures; a compact hook
summary is retained instead of the multi-megabyte diagnostic log.
