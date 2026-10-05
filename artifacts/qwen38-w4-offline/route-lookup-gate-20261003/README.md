# Qwen W4A8 route lookup gate, 2026-10-03

The checkpoint has 10 route slots per token and 512 global experts. Under
TP4/EP4 its local bank has 128 experts. The earlier eight-slot, 32-expert
[math sweep](../math-sweep-20261003/README.md) was useful for identifying
expert fan-out as a variable, but did not match these decode shapes.

This candidate changes only the native INT4 routed planner. It preserves
first-seen expert order and original route order, then switches from linear
expert-list search to a 128-entry lookup table after 32 distinct local
experts appear. The C1 and fused down-reduction schedules compile with the
lookup disabled. Expert banks larger than 128 retain linear search.

## Operator gate

The baseline was the coherent r2 OPP. The final candidate was built from an
isolated copy of its source, with the changed planner header and two schedule
instantiations, and installed separately. The GLM service was stopped for the
device gate. The raw files here are from one Ascend 310P3 on the same host:

- [`baseline.jsonl`](baseline.jsonl) and
  [`baseline-repeat.jsonl`](baseline-repeat.jsonl): retained OPP before and
  after the candidate runs.
- [`candidate-v1.jsonl`](candidate-v1.jsonl): lookup initialized at 60 routes.
- [`candidate.jsonl`](candidate.jsonl): lookup initialized after 32 distinct
  experts, still present in every schedule specialization.
- [`candidate-v3.jsonl`](candidate-v3.jsonl): final source, with lookup
  compiled out of C1 and fused down reduction.

Each run used 128 synthetic local experts with varying G128 scale and offset,
10 slots per token, 3/6/12 tokens, and four route patterns. Timings are the
median NPU graph event time per call, from five trials of 12 replays with four
calls captured per graph. Calls within a graph can overlap. Prepared projection
excludes activation packing. The benchmark checked eager/graph parity and
changing-input, changing-ID peer-zero replay, but not a full MoE layer.

| Projection and pattern | Routes | Baseline mean of medians (µs) | Final candidate (µs) | Change |
| --- | ---: | ---: | ---: | ---: |
| Gate/up, TP4 sparse | 30 | 109.9 | 110.3 | +0.4% |
| Gate/up, TP4 sparse | 60 | 349.9 | 342.9 | -2.0% |
| Gate/up, TP4 sparse | 120 | 651.4 | 646.8 | -0.7% |
| Gate/up, distinct | 60 | 1245.2 | 1218.6 | -2.1% |
| Gate/up, distinct | 120 | 2542.5 | 2415.4 | -5.0% |
| Down, distinct | 60 | 522.0 | 506.9 | -2.9% |
| Down, distinct | 120 | 1047.4 | 958.4 | -8.5% |

At 30 distinct routes, gate/up rose from 413.8 to 418.2 µs (+1.1%), despite
the lookup being disabled in C1. The sparse cases stayed within roughly 2% of
baseline across both projections. These small differences need a serving run
before claiming a production effect. The large distinct-expert gain is an
operator result for a deliberately adverse local route distribution.

The baseline and final candidate each passed 62 selected native route tests,
including graph replay, 10 routes per token at 30/60/120 rows, and the
128/129-expert boundary. Four additional prefill-capacity, reused-activation,
and fused down-reduction tests passed for both. The host route planner test
passed for every route count from 1 to 128 with invalid IDs and broadcast
factors 1 and 10.

## Real-checkpoint serving gate

Both operator packages started the same TP4, EP4, MTP2 Qwen server with the real checkpoint,
262,144-token context, four sequences, and full decode graphs captured at
three and six query rows. The candidate passed the established three-request
smoke, a thinking-enabled request, a tool-call request, a unique 23,393-token
cold-prefix prefill, three serial 512-token requests, and four concurrent
256-input/512-output requests. One separate seven-case smoke check failed a
character-reversal prompt: the candidate returned `DNESCA` for `ASCEND`
instead of `DNECSA`. This is a prompt-level model error, not an operator
exception. The paired baseline serving run passed the serial and concurrent
benchmarks; it was not run through the entire smoke set.

| Fixed-prompt serving measure | Baseline | Candidate | Candidate change |
| --- | ---: | ---: | ---: |
| Serial 512 output, request 1 (tok/s) | 32.71 | 29.69 | -9.2% |
| Serial 512 output, request 2 (tok/s) | 26.26 | 25.73 | -2.0% |
| Serial 512 output, request 3 (tok/s) | 27.07 | 25.26 | -6.7% |
| Four concurrent, aggregate end-to-end (tok/s) | 56.48 | 55.14 | -2.4% |

All four concurrent requests completed 512 output tokens without preemption
for both packages. The serial warmup text SHA-256 matched; each 512-token
completion and each concurrent completion differed between packages. The
candidate therefore has no demonstrated serving throughput gain, despite its
synthetic high-fan-out operator gain. Long generations and speculative
acceptance can diverge after small numerical changes, and this single paired
serving pass cannot assign causality to the route lookup. The rebuilt packages
also have different host API and tiler library hashes. A baseline package
with only the three candidate native kernel objects overlaid has been
assembled separately, but it has not been served or benchmarked.

The shared source retains the lookup implementation for explicit kernel
specializations. Its template default is disabled after this serving result.

The serving raw results are [`baseline-serial512.jsonl`](baseline-serial512.jsonl),
[`candidate-serial512.jsonl`](candidate-serial512.jsonl),
[`baseline-c4.jsonl`](baseline-c4.jsonl), and
[`candidate-c4.jsonl`](candidate-c4.jsonl). The candidate
[smoke transcript](candidate-established-smoke.txt) includes a final pass marker
after JSON events; the cold-prefix record is in this directory as well. The
retained baseline is the
live package on port 8001 for manual testing; the candidate was stopped after
its gate.

## Reproduction

[`sweep_route_lookup.py`](sweep_route_lookup.py) and
[`run_gate.sh`](run_gate.sh) are the exact measurement driver and remote
environment setup. On the qualified host, run the baseline and candidate in
separate processes so each imports one coherent OPP package:

```bash
bash run_gate.sh baseline test
bash run_gate.sh baseline test_extra
bash run_gate.sh baseline bench /path/to/baseline.jsonl
bash run_gate.sh candidate test
bash run_gate.sh candidate test_extra
bash run_gate.sh candidate bench /path/to/candidate.jsonl
```

The route patterns and expert weights are synthetic. They omit the router,
shared expert, QSA, GDN, TP collectives, checkpoint weight distribution, and
full MTP scheduling. Do not convert these operator timings into model tokens/s.
