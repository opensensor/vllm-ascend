# PRD: System-One Structured-Decision Runtime on Ascend 310P

**Generated**: 2026-09-18
**Companion plan**: `system-one-structured-decision-310p-plan.md` (to be written)
**Precedent / reuse**: the DeepSeek V4.1 and GLM-5.3-Flash W2 ports
(`deepseek_v41_w2_310p_prd.md`, `glm53-flash-w2-310p-plan.md`) — this work reuses
their W2 expert stack, 310P quantization methods, streamed loader, dtype-policy
discipline and observability as the **slow-path (System-Two) fallback tier**. The
genuinely new work is a **single-forward, type-safe, calibrated structured-decision
head** and its serving path.

| | |
| --- | --- |
| Motivation | Autoregressive decode is the 310P's worst workload; a large class of automation tasks needs a *typed decision*, not free-form text |
| Target host | Two Atlas 300I Duo cards = four 48 GB Ascend 310P chips; 122 GB DDR dev host (64 GB until the RAM upgrade lands) |
| Prior art (external, closed) | TypeSafe AI's "Jev" / "System One" — hosted API, no weights, no local-deploy or license info published |
| Initial target | An OSS runtime: `(context, output_schema) -> (typed_value, per-field calibrated confidence)` in **one NPU forward**, with abstain→escalate to the W2 MoE |

---

## 1. Problem statement

The Ascend 310P (Atlas 300I Duo) is a **throughput** part, not a **latency** part.
Our current DeepSeek/GLM W2 paths are autoregressive: each output token is a full
MoE forward with a sequential dependency on the previous token, on a card that (a)
has **no native sub-INT8 GEMM** (we unpack W2→INT8 per active expert every step),
(b) is memory-bandwidth-bound at decode, and (c) grows a KV cache per token. An
N-token structured answer costs N latency-chained passes. For a large fraction of
real automation work — classify, extract, route, validate, fill a schema — the
output is a small **typed value**, not prose, and autoregressive generation is the
wrong tool on this hardware.

A **System-One** runtime answers that class of task in a **single, batched,
compute-bound forward** that plays to the 310P's strength, emits a value that is
**type-safe by construction**, attaches **calibrated per-field confidence**, and
**abstains** (escalating to the big W2 MoE, "System Two") when unsure. This PRD
defines a text-only bring-up of that runtime as an OSS enhancement to vllm-ascend.

This is explicitly **not** a claim to reproduce Jev (closed weights/architecture).
It is an independent design with the same *useful properties*, built on techniques
with public foundations (constrained decoding, calibration, conformal prediction).

## 2. Goals

1. A **single-forward structured-decision path**: a non-autoregressive head that,
   given a context and an output schema, emits all fields in one NPU pass
   (per-field classification / span-extraction / small regression heads), on the
   310P's prefill-style compute-bound path (no KV growth, batchable).
2. **Type-safety by construction**: the emitted value is guaranteed to satisfy the
   requested schema/grammar. Invalid outputs are impossible, not merely unlikely.
3. **Calibrated confidence**: a per-field probability whose calibration is
   *measured* (Expected Calibration Error / reliability curves), starting with
   temperature/Platt scaling and graduating to **conformal prediction** with a
   coverage guarantee and set-valued "unsure" outputs.
4. **Selective prediction (System-One/Two split)**: a principled **abstain**
   threshold that routes low-confidence cases to the existing W2 MoE, so accuracy
   is preserved and only the hard tail pays the slow-path cost.
5. A **measured latency + accuracy + calibration benchmark** on real 310P hardware
   versus the autoregressive-constrained baseline, published with the runtime.

## 3. Non-goals for the first release

- Free-form / long-form generation (that stays the W2 MoE's job — this runtime
  routes to it, it does not replace it).
- Reproducing Jev's model, architecture, training method (RLCD) or numbers.
- Any "cannot hallucinate" claim. We claim **type-safe by construction** and
  **calibrated within a measured ECE**; a valid-but-wrong value is still possible
  and is bounded only probabilistically (that is what calibration + abstain are
  for).
- Multimodal / vision inputs.
- Training a base model from scratch. Phase 1–2 adapt an existing encoder/decoder;
  a bespoke architecture is a later, optional gate.
- A production RL calibration objective ("RLCD analog") — scoped as Phase 3, after
  the cheaper calibration methods are measured.

## 4. Fixed inputs and assumptions

### 4.1 Hardware
- Four independent 48 GB 310P chips (no fast inter-chip fabric assumed beyond what
  the W2 ports already use); host **122 GB DDR, 64 GB until the RAM upgrade** — the
  bring-up must stay within ~45 GB usable host RAM until then (the same page-cache
  discipline, `posix_fadvise(DONTNEED)`, that the W2 converters use).
- 310P: **no sub-INT8 GEMM**; INT8 grouped matmul via
  `npu_quant_grouped_matmul_dequant` is the workhorse (shared with the W2 stack).

### 4.2 Workload assumption (must be validated in Phase 0)
- The target tasks have **bounded, schema-describable outputs** (enums, bounded
  strings/spans, small numerics, nested records) — not open-ended text.
- Output size is small relative to context: the win comes from replacing **N
  sequential decode steps with 1 parallel pass**, so the thesis is strongest when
  the autoregressive baseline would emit many tokens.

### 4.3 The System-Two tier already exists
- The DeepSeek V4.1 and GLM-5.3-Flash W2 runtimes (this repo) are the escalation
  target. No new large-model work is required; this runtime consumes them.

## 5. Why the 310P favors this (latency model — replace with measurements)

Rough, pre-hardware reasoning to be replaced by Phase-0 numbers:

- **Autoregressive constrained baseline**: `T_ar ≈ N_out × t_decode`, where
  `t_decode` is one MoE decode step (bandwidth-bound, W2 unpack per active expert,
  KV append). Latency-chained: no batching across the N steps of one request.
- **Single-forward System-One**: `T_s1 ≈ t_prefill(context) + t_head`, one
  compute-bound pass, batchable across requests, no KV growth. `t_head` is a set of
  small matmuls.
- Expected win scales with `N_out` and with how bandwidth-bound `t_decode` is on
  310P. **If Phase 0 does not show a ≥5× median latency gap on representative
  tasks, the thesis is weak and we stop** — this is the gating measurement, not a
  foregone conclusion.

## 6. The core contract (the critical new work)

### 6.1 Type-safe decoding
- An output schema (JSON-Schema subset / algebraic types) compiles to a **decoding
  constraint**. Two candidate mechanisms, decided in Phase 1:
  - **Constrained AR** (interim): grammar-guided token masking (XGrammar /
    llguidance / outlines-style) on the existing serving path — guarantees validity
    but is still autoregressive (a stepping stone, shippable alone).
  - **Structured heads** (target): per-field heads over a fixed output layout —
    validity is structural (an enum head cannot emit a non-member), no token grammar
    needed, and the pass is single-shot.

### 6.2 Calibrated confidence
- Per-field probability, **post-hoc calibrated** on a held-out set:
  temperature/Platt first; **conformal prediction** for coverage guarantees and
  set-valued outputs. Reported with **ECE + reliability diagrams**, not asserted.

### 6.3 Selective prediction / escalation
- An abstain rule (per-field or per-record confidence below a tuned threshold, or a
  conformal set larger than 1) routes the request to the W2 MoE. The threshold is
  chosen on a validation set to hit a target **risk–coverage** operating point.

### 6.4 Serving path
- A vllm-ascend serving entry that accepts `(context, schema)`, runs the single
  forward on the 310P (reusing the INT8 matmul + dtype-policy plumbing), applies
  calibration, and returns the typed value + confidences + an `escalate` flag. The
  slow path reuses the W2 runtimes unchanged.

## 7. Product requirements

1. **Correctness**: emitted values are 100% schema-valid (property-tested against
   the grammar/heads — a violation is a hard failure, not a metric).
2. **Calibration**: report ECE and reliability curves on a held-out task; the
   release names its measured ECE rather than a slogan.
3. **Latency**: publish median/p95 single-forward latency on 310P vs. the AR
   baseline on the same card and task.
4. **Selective accuracy**: publish the risk–coverage curve (accuracy at each
   abstain rate) and the end-to-end accuracy/cost of the two-tier system.
5. **Host-RAM bound**: bring-up stays within the ~45 GB pre-upgrade ceiling; no
   full-model host materialization.
6. **Reuse, not fork**: the W2 stack is consumed as the fallback tier without
   modification.

## 8. Validation and release gates

- **Phase 0 (gating measurement)**: on ≥1 representative structured task, measure
  the AR-constrained baseline latency on real 310P and the projected single-forward
  cost. **Gate: ≥5× median latency headroom, or the project is re-scoped/stopped.**
- **Phase 1 gate**: type-safe decoding produces 100% valid outputs on a task
  (constrained-AR acceptable here); measured on the card.
- **Phase 2 gate**: single-forward structured head beats the AR baseline on
  measured latency **and** matches its task accuracy within a stated tolerance,
  with a reported ECE after calibration.
- **Phase 3 gate**: selective routing achieves the target risk–coverage point;
  end-to-end two-tier accuracy ≥ the W2 MoE alone at a fraction of the cost.

## 9. Reuse inventory (what carries over)

- **INT8 grouped matmul + 310P quant methods** (`_310p/quantization/methods/*`) —
  the head/encoder matmuls ride the same path.
- **Streamed loader + placement/accounting** (`_310p/sharded_state_loader_310p.py`,
  `observability/*`) — RAM-bounded weight loading and per-chip budgeting.
- **dtype-policy discipline** (per the DeepSeek/GLM `dtype_policy.py`) — no magic
  literals; fp16 IO / fp32 accumulation pinned centrally.
- **Page-cache discipline** (`posix_fadvise(DONTNEED)`) — survives the 45 GB host
  window.
- **The W2 MoE runtimes** (DeepSeek V4.1, GLM-5.3-Flash) — the System-Two tier.

## 10. New work (the port)

1. Schema → decoding-constraint compiler (grammar and/or structured-head layout).
2. Single-forward structured head (per-field classification / span / regression).
3. Post-hoc calibration module (temperature/Platt → conformal) + ECE/reliability
   reporting.
4. Selective-prediction / escalation router to the W2 tier.
5. vllm-ascend serving entry + a reproducible latency/accuracy/calibration
   benchmark harness (the published artifact).
6. A base-model choice for the encoder/head (adapt an existing small model first;
   bespoke architecture optional/later).

## 11. Risks and mitigations

- **The latency win doesn't materialize** → Phase 0 is an explicit stop-gate before
  any model work.
- **Calibration is poor / task-specific** → this is the real research; start with
  well-understood post-hoc methods, report honestly, and lean on abstain+escalate
  so poor calibration degrades to "route to System Two," never to a confident wrong
  answer.
- **Task coverage is narrow** → scope to a concrete task family first; generality is
  earned, not assumed.
- **Overclaiming** → the PRD forbids "can't hallucinate"; every claim maps to a
  measurable (validity %, ECE, risk–coverage, latency).
- **RAM window** → no host full-model materialization until the upgrade; loader
  streams.

## 12. Open decisions

- Base model / encoder for the structured head (adapt existing vs. bespoke).
- Constraint mechanism for Phase 1: constrained-AR interim vs. jump straight to
  structured heads.
- The first concrete **task family** to prove it on (drives the schema, the
  calibration set, and the benchmark).
- Calibration method to ship first (temperature vs. conformal) given the task's
  need for coverage guarantees.
- Whether Phase 3's calibration training (an "RLCD analog") is in scope for the
  first OSS release or a follow-on.
