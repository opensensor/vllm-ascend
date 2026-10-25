# Compact decode routes and graph reload ownership

Current retained path: INT8 v15, activation v2, compact four-route decoding,
shared-pool partial graphs. Context remains 163840 with TP2/PP3/EP2.

- v15 median: 3.7780888918 tokens/s; reload-fix median: 3.7769239825.
- Each benchmark uses three greedy 64-output-token counting requests. Decode
  throughput excludes the first token. All response messages match the baseline.
- Twelve isolated real-weight projection comparisons match v14 exactly,
  including unsorted and peer-only routes. Native admission also checks compact
  versus legacy boundaries on each worker.
- All 240 graph pieces captured, with zero disabled layers. Minimum admission
  free memory: 939433984 bytes. Same worker PIDs and weight storage throughout
  these two swaps. Earlier graph experiments required restarts; see the
  piecewise-forward artifact directory for those failures.
- 228 CPU tests passed, 24 hardware tests skipped. The generic parent conftest
  needs unavailable vLLM dependencies locally; use the scoped command below.

```bash
pytest -q --confcutdir=tests/ut/minimax_m3 tests/ut/minimax_m3
```

The saved trial scripts are historical controller snapshots, not general rollback
utilities. The v15 trial handed compatible immutable scale metadata from v14 to
v15 and cleared older copies. Factories run during preparation, before graph
release; this handoff retained the tensors referenced by existing graphs.
Subsequent generations use ordinary side-effect-free preparation.

The installed post-start script converts directly with the v15 manifest and
selects the same module, activation and graph settings. Shell syntax and native
compatibility are checked; a fresh cold startup was not repeated. Sources and
startup configuration are persisted without restarting the working server.
