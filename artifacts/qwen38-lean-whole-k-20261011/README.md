# Lean whole-K hardware qualification

Live candidate: `qwen_qsa_owner_lean_int8_math` on six Ascend 310P3 chips.

- `validation-rank-*.json`: exact real-weight comparison at 3, 6, 128, 640,
  and 1,280 tokens, changing-route graph replay and alternating stage timings.
- `real-layer-rank-*.json`: the independent pre-capture qualification.
- `lean-whole-k-*.json`: cold prefill, coding, thinking, and image/text gates.
- `control.json`, `previous-control.json`, `switch-after.json`: cutover source
  and receipt with resident weight storage/PID verification.
- `provenance.json`: frozen assets and SHA-256 digests; binaries stay on host.
- `.cpp.txt` and `.py.txt`: exact immutable compiled/runtime source snapshots.
- `unit-tests.txt`: 73 passing focused CPU tests.

The variant removes unnecessary whole-K vector initialization and accumulation
while preserving scale order, integer accumulation, FP16 output rounding,
partial-tile padding and ownership events. Existing entries default to the old
recipe. Projection peer clearing and CANN finalization remain enabled.

At 1,280 tokens the routed MoE partial improved about 0.5–2.0% across ranks.
Cold TTFT was 18.987 seconds with no cached tokens, and coding decode measured
31.672 tokens/s. End-to-end gains are not established by these single runs.
The 1M request limit is retained but its full-length quality is unqualified.
