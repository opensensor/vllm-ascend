# Plan: System-One Structured-Decision Runtime on Ascend 310P

**Generated**: 2026-09-18
**PRD**: `docs/source/developer_guide/Design_Documents/system_one_structured_decision_310p_prd.md`
**Precedent / reuse**: the DeepSeek V4.1 and GLM-5.3-Flash W2 ports — this work
reuses their 310P INT8 matmul, streamed loader, placement/accounting, dtype-policy
discipline, page-cache (`posix_fadvise(DONTNEED)`) discipline, pathspec commits,
`--noconftest` host tests, and per-op tolerances. Those W2 MoE runtimes are the
**System-Two fallback tier** this runtime escalates to.

## Overview

Deliver an OSS **System-One** runtime for vllm-ascend: given `(context, output
schema)`, return `(typed value, per-field calibrated confidence, escalate flag)`
in **one compute-bound NPU forward** — the 310P's strength — instead of latency-
chained autoregressive decode. The value is **type-safe by construction**;
confidence is **post-hoc calibrated** (measured ECE, not asserted); low-confidence
requests **abstain and escalate** to the W2 MoE. Everything except the on-card
measurements is built and tested **host-side, CPU-only, Triton-free**, mirroring
the W2 discipline. **Phase 0 is a hard stop-gate**: no model work begins until a
real 310P measurement shows ≥5× median latency headroom vs. the autoregressive
baseline.

## Guardrails (same as the W2 waves)

- Host RAM ceiling **~45 GB until the RAM upgrade** (64 GB box, ~19 GB in use); no
  full-model host materialization; loader streams; `posix_fadvise(DONTNEED)`.
- Pathspec commits only (`git commit -s -- <files>`); never `git add -A`/`--amend`;
  subagents never push; `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.
- Every claim maps to a measurable (schema-validity %, ECE, risk–coverage, latency).
  **No "can't hallucinate" language anywhere.**
- CPU-testable tasks use `python3 -m pytest -q --noconftest tests/ut/system_one/…`.

## Initial task family (open decision — proposed default)

To make the runtime concrete, the bring-up targets **typed function-call argument
selection / intent routing**: unstructured input + a tool/function schema → a
typed call (enum tool id, typed arguments) with per-field confidence. Proposed
reproducible dataset: a public function-calling / JSON-extraction benchmark (e.g.
BFCL-style, or a JSON-Schema extraction set) — final choice is **P0.3**. The schema
family (enums, bounded strings/spans, small numerics, nested records) is fixed;
the dataset is swappable behind the harness.

---

## Phase 0 — Gating measurement + scaffolding (mostly host; P0.4 needs cards)

### T0.1: Schema → decoding-constraint IR
- **depends_on**: []
- Location: `vllm_ascend/system_one/schema_ir.py` + `tests/ut/system_one/test_schema_ir.py`
- Compile a JSON-Schema subset (enum, bounded string/span, int/float with range,
  bool, nested record, optional) into a normalized **constraint IR** that both the
  constrained-AR path (Phase 1) and the structured-head layout (Phase 2) consume.
  Pure Python, no NPU.
- Acceptance: round-trips the supported schema subset; rejects unsupported
  constructs loudly; enumerates each field's type + domain (for head sizing).
- Validation (CPU): unit tests over a schema corpus; property test that every IR
  field maps back to a validator.

### T0.2: Structured-validity checker + property harness
- **depends_on**: [T0.1]
- Location: `vllm_ascend/system_one/validate.py` + `tests/ut/system_one/test_validate.py`
- A validator that, given the IR and a candidate value, returns valid/invalid with
  the offending field. This is the oracle for the PRD's "100% valid" gate.
- Acceptance: accepts all schema-valid values, rejects every out-of-domain value
  (enum non-member, out-of-range numeric, wrong type, missing required).
- Validation (CPU): exhaustive small-domain enumeration + fuzzed invalid values.

### T0.3: Benchmark harness + task-family loader
- **depends_on**: [T0.1]
- Location: `tools/system_one/bench.py` + `tests/ut/system_one/test_bench.py`
- A reproducible harness that loads the chosen task family into `(context, schema,
  gold_value)` records and scores a runtime on **latency (median/p95), task
  accuracy, schema-validity %, and calibration (ECE + reliability bins)**. Runtime
  is pluggable (baseline vs. System-One vs. two-tier). Finalizes the dataset choice.
- Acceptance: emits a JSON report with all four metric families on a stub runtime;
  ECE/reliability math unit-tested against hand-computed cases.
- Validation (CPU): metric functions tested on synthetic predictions with known ECE.

### T0.4: Baseline latency measurement on 310P (GATE)
- **depends_on**: [T0.3]
- Location: `artifacts/system-one/phase0_baseline.json` + a short write-up appended here.
- **Needs the cards.** Run the autoregressive-constrained baseline (an existing
  GLM/DeepSeek W2 runtime with grammar-guided decoding, or the closest available)
  on the task family; record median/p95 per-request latency and per-token
  `t_decode`. Project the single-forward cost (`t_prefill + t_head`).
- **GATE: ≥5× median latency headroom, or the project is re-scoped/stopped** and the
  rest of the plan does not proceed. Record the decision here regardless of outcome.
- Validation (device): measured numbers, not projections, for the AR baseline.

---

## Phase 1 — Type-safe decoding (shippable alone; interim constrained-AR)

### T1.1: Grammar-guided constrained decoder (interim path)
- **depends_on**: [T0.1, T0.2]
- Location: `vllm_ascend/system_one/constrained_decode.py` + tests
- Compile the constraint IR to a token-mask / grammar (XGrammar / llguidance /
  outlines-style) applied over the existing serving path so autoregressive output
  is **guaranteed schema-valid**. This ships value before the single-forward head
  exists. Host-testable with a tiny tokenizer + a mock logits stream.
- Acceptance: 100% valid outputs against T0.2 on the task family (host mock);
  invalid tokens are masked, not merely penalized.
- Validation (CPU): mock-logits tests prove masking; (device parity in T1.3).

### T1.2: Serving entry (schema in → typed value out)
- **depends_on**: [T1.1]
- Location: `vllm_ascend/system_one/serve.py` + tests
- A serving surface that accepts `(context, schema)`, drives the constrained path,
  parses the result through T0.2, and returns the typed value. This is the seam the
  Phase-2 head swaps into without a call-site change.
- Acceptance: end-to-end host run returns a schema-valid typed value; malformed
  schema rejected at the gate.
- Validation (CPU): host integration test with the mock runtime.

### T1.3: Phase-1 device gate
- **depends_on**: [T1.2, T0.4]
- **Needs the cards.** Run T1.2 on real 310P against the task family; confirm 100%
  schema-validity on the card and record latency (this is still the AR baseline
  shape — the number motivates Phase 2).
- Acceptance: 0 invalid outputs on the card; latency recorded in the artifact.

---

## Phase 2 — Single-forward structured head + calibration (the core win)

### T2.1: Structured-head layout + eager reference
- **depends_on**: [T0.1, T0.4]
- Location: `vllm_ascend/system_one/heads.py` + `tools/system_one/heads_reference.py` + tests
- From the constraint IR, size a **non-autoregressive head set**: per-field
  classification (enum), span-extraction (bounded string), small regression
  (numeric), presence (optional). One forward emits all fields. Provide a pure-torch
  CPU **eager reference** (the parity oracle, mirroring the KDA G-ref pattern).
- Acceptance: heads emit structurally-valid fields by construction (an enum head
  cannot produce a non-member); reference runs on CPU, fp32 accumulation, fp16 IO.
- Validation (CPU): validity is structural (T0.2 always passes); shape/dtype tests.

### T2.2: 310P head execution (INT8 matmul reuse)
- **depends_on**: [T2.1]
- Location: `vllm_ascend/system_one/heads_310p.py` + tests
- Run the head matmuls on the 310P INT8 path (reuse `_310p/quantization/methods/*`),
  Triton-free, torch_npu-gated so the host parity path runs on CPU. Parity vs. the
  T2.1 reference within a stated tolerance.
- Acceptance: host parity allclose vs. reference; no triton / `ops.triton` on the
  import path (grep-gate).
- Validation (CPU): parity test against the eager reference.

### T2.3: Encoder / base-model adapter
- **depends_on**: [T2.1]
- Location: `vllm_ascend/system_one/encoder.py` + tests
- Adapt an existing small model as the single-pass context encoder feeding the
  heads (base-model choice is an **open decision**; start with an off-the-shelf
  encoder or a decoder run non-AR). Streamed loader + dtype-policy reuse; RAM-bounded.
- Acceptance: constructs + forwards on CPU at tiny dims with dummy weights (the
  assembly/boot pattern from the W2 waves); loader stays within the RAM ceiling.
- Validation (CPU): dummy-weight boot test.

### T2.4: Post-hoc calibration module
- **depends_on**: [T0.3]
- Location: `vllm_ascend/system_one/calibrate.py` + tests
- Temperature/Platt scaling first; **conformal prediction** (coverage guarantee,
  set-valued outputs) second. Fit on a held-out split; emit calibrated per-field
  probabilities + ECE/reliability via the T0.3 harness.
- Acceptance: temperature scaling reduces ECE on a synthetic miscalibrated set;
  conformal sets achieve the requested coverage (± tolerance) on held-out data.
- Validation (CPU): calibration math tests with known-miscalibration fixtures.

### T2.5: Single-forward assembly + dummy-weight CPU boot
- **depends_on**: [T2.2, T2.3, T2.4]
- Location: `vllm_ascend/system_one/assembly.py` + tests
- Wire encoder → heads → calibration into one `SystemOneRuntime` that boots on CPU
  with dummy weights, forwards once, and returns `(typed value, calibrated
  confidences)`. Mirrors the W2 eager-assembly + boot-test pattern.
- Acceptance: constructs/forwards/returns a schema-valid calibrated result on CPU;
  Triton-free import (frozen import-delta asserted).
- Validation (CPU): boot test (construct → forward → validate → calibrated output).

### T2.6: Phase-2 device gate
- **depends_on**: [T2.5, T1.3]
- **Needs the cards.** Real-weight (or fine-tuned) single-forward run on 310P vs. the
  T1.3 AR baseline on the same task/card.
- **GATE**: single-forward beats the baseline on **measured latency** AND matches
  task accuracy within a stated tolerance, with a **reported ECE** after calibration.
- Validation (device): the T0.3 report for both runtimes, side by side.

---

## Phase 3 — Selective prediction / escalation (two-tier System-One/Two)

### T3.1: Abstain rule + risk–coverage tuner
- **depends_on**: [T2.4]
- Location: `vllm_ascend/system_one/select.py` + tests
- A per-field / per-record abstain rule (confidence threshold or conformal-set-size
  > 1) with a tuner that picks the operating point on a validation risk–coverage
  curve for a target risk.
- Acceptance: given a validation set, returns a threshold hitting the target risk;
  risk–coverage curve is monotone and reported.
- Validation (CPU): tuner tests on synthetic score/label sets.

### T3.2: Escalation router to the W2 MoE (System-Two)
- **depends_on**: [T3.1, T2.5]
- Location: `vllm_ascend/system_one/router.py` + tests
- On abstain, route `(context, schema)` to the existing DeepSeek/GLM W2 runtime and
  return its (constrained) result; otherwise return System-One's. The W2 tier is
  consumed unchanged.
- Acceptance: abstained requests reach the W2 path (mocked in host tests); non-
  abstained bypass it; the two-tier output is always schema-valid.
- Validation (CPU): host test with a mock System-Two backend.

### T3.3: Two-tier device benchmark (release artifact)
- **depends_on**: [T3.2, T2.6]
- **Needs the cards.** End-to-end two-tier run on 310P: publish
  latency/accuracy/validity/ECE + the risk–coverage curve and the end-to-end
  accuracy/cost vs. the W2 MoE alone.
- **GATE**: two-tier accuracy ≥ the W2 MoE alone at a fraction of the cost; publish
  the artifact as the runtime's benchmark.

### T3.4 (optional, follow-on): calibration-aware training ("RLCD analog")
- **depends_on**: [T3.3]
- Scope only after the post-hoc results are measured: a training objective combining
  a proper scoring rule (Brier/log-loss) with a selective-prediction term. Likely
  needs GPU (rental) for training; out of scope for the first OSS release unless
  T3.3 shows post-hoc calibration is insufficient.

---

## Wave order (for `/parallel-task`)

1. **Wave A (host, no cards)**: T0.1 → then T0.2, T0.3 in parallel.
2. **GATE**: T0.4 (cards) — proceed only on ≥5× headroom.
3. **Wave B (host)**: T1.1 → T1.2; T2.1, T2.4 in parallel (T2.1 after T0.4 gate).
4. **Wave C (host)**: T2.2, T2.3 (after T2.1); T3.1 (after T2.4).
5. **Wave D (host)**: T2.5 (after T2.2/T2.3/T2.4); T3.2 (after T3.1/T2.5).
6. **Device gates**: T1.3, T2.6, T3.3 as the cards + fine-tuned weights allow.

## Open decisions (carried from the PRD §12)

- Base model / encoder for the head (T2.3).
- Constraint mechanism split: how long the constrained-AR interim (T1.x) lives
  alongside the structured heads (T2.x).
- Final task family + dataset (T0.3).
- First calibration method to ship (temperature vs. conformal) given coverage needs.
- Whether T3.4 (calibration-aware training) is in the first release.

---

## Completion log

### T0.1 — Schema → decoding-constraint IR — DONE (2026-09-18)

**Status**: complete. RED→GREEN, ruff clean, committed (pathspec).

**Files**
- `vllm_ascend/system_one/__init__.py` (new subpackage; host-side, CPU-only, Triton-free).
- `vllm_ascend/system_one/schema_ir.py` — the compiler.
- `tests/ut/system_one/__init__.py`, `tests/ut/system_one/test_schema_ir.py` — 28 UTs.

**IR shape (what downstream tasks consume)**
- `compile_schema(schema: dict) -> SchemaIR`. Root must be an `object` with
  `properties`; anything else raises loudly.
- `SchemaIR(fields: tuple[FieldIR, ...], path: str)` — immutable (frozen dataclass);
  `iter_fields()` (top-level, declaration order), `leaf_fields()` (depth-first,
  descends into nested objects, objects themselves are **not** leaves),
  `field_paths()`, `validator_descriptors()`.
- `FieldIR(name, path, kind, domain, required)` — `path` is JSON-pointer-ish, root
  `"$"`, nested `"$.user.role"`. `is_leaf()` = not OBJECT.
  `validator_descriptor()` flattens a field back to a dict carrying
  `path/name/kind/required` + the domain keys (the T0.2 round-trip contract).
- `FieldKind`: `ENUM / STRING / INTEGER / NUMBER / BOOLEAN / OBJECT` (str-mixin enum).
- Domain descriptors (all frozen, `.describe()` → dict):
  `EnumDomain(members)`, `StringDomain(max_length, pattern)`,
  `NumericDomain(minimum, maximum, integral)`, `BooleanDomain()`,
  `ObjectDomain(schema: SchemaIR)`.
- Exceptions: `SchemaCompileError(ValueError)` base with `.path`;
  `UnsupportedSchemaError(SchemaCompileError)` for out-of-subset constructs.

**Supported subset / decisions (downstream must know)**
- `enum`: non-empty list of `str`/`int`. **`bool` members rejected** (bool is an
  int subclass; kept distinct from the BOOLEAN kind).
- `string`: **`maxLength` required** (unbounded string rejected). `pattern` is
  *recorded but not enforced* by the IR (T1.1/T0.2 may enforce).
- `integer`/`number`: **both `minimum` and `maximum` required** (unbounded numeric
  rejected) so Phase-2 heads can size the range. `integral` flag distinguishes int
  vs number.
- `object`: must declare `properties`; `required` is validated against declared
  field names (naming an unknown field → `SchemaCompileError`). Field/property
  **order is preserved** (dict insertion order) — heads and grammars can rely on it.
- Rejected loudly with the offending path: `oneOf/anyOf/allOf/not`, `$ref`,
  list-valued (union) `type`, `array`, unknown/`null` type, field with neither
  `type` nor `enum`, non-object root, object without `properties`.

**Gotchas**
- Tests load `schema_ir.py` **by file path** (importlib) instead of
  `import vllm_ascend...` to avoid the package `__init__` pulling in torch/vLLM;
  keeps the UT truly host-side. The loader must register the module in
  `sys.modules` **before** `exec_module` or Python 3.12+ dataclass forward-ref
  resolution of the nested `SchemaIR` reference fails with an `AttributeError`.
- Import-hygiene gate is `ast`-based (flags real `import`/`from` statements only)
  so the module docstring may legitimately name `torch_npu`/`triton` in prose.
- Ruff (0.15.x) enforces `X | Y` unions and unquoted forward refs under
  `from __future__ import annotations`; the module is written that way.

**RED→GREEN evidence**
- RED (before `schema_ir.py`): collection error `FileNotFoundError: .../schema_ir.py`.
- GREEN: `python3 -m pytest -q --noconftest tests/ut/system_one/test_schema_ir.py`
  → `28 passed`. `ruff check vllm_ascend/system_one/ tests/ut/system_one/` → clean.

### T0.2 — Structured-validity checker + property harness — DONE (2026-09-18)

**Status**: complete. RED→GREEN, ruff clean, committed (pathspec).

**Files**
- `vllm_ascend/system_one/validate.py` — the validity oracle.
- `tests/ut/system_one/test_validate.py` — 30 UTs (host-side, `--noconftest`).

**API shape (what T1.1/T1.3/T2.1 consume)**
- `validate_value(ir: SchemaIR, value, *, strict=True) -> ValidationResult`.
- `is_valid(ir, value, *, strict=True) -> bool` (convenience).
- `ValidationResult(ok: bool, violations: tuple[Violation, ...])` — frozen; also
  `.path` / `.reason` (first offending violation, or `None` when valid) and
  `__bool__` → `ok`. `Violation(path, reason)` — frozen; `path` is the IR's
  JSON-pointer-ish path (`"$.user.role"`).
- Violations are ordered by schema declaration order, depth-first into nested
  objects; missing/invalid declared fields precede unknown-extra-key violations,
  so `.path` is the first offending field in schema order.

**Per-kind checks (driven only by the IR domains)**
- ENUM → membership in `domain.members`; STRING → `str` type, `len ≤ max_length`,
  pattern; INTEGER → `int` (not bool) in `[min, max]`; NUMBER → `int`/`float`
  (not bool) in `[min, max]` (int accepted); BOOLEAN → `bool`; OBJECT → value
  must be `dict`, recurse into the nested `SchemaIR`.
- Required-field presence checked per record; nested records recurse with correct
  nested paths.

**Decisions (downstream must know)**
- **Strict by default**: unknown/extra keys are a violation (`strict=True`). This
  is the safer oracle for the "100% schema-valid" gate. `strict=False` gives a lax
  mode that tolerates unknown keys but still enforces every declared field's domain.
- **`bool` is not int/number/enum-int**: although `bool` subclasses `int`,
  `True`/`False` are rejected for INTEGER/NUMBER fields and for enum int members;
  valid only for BOOLEAN fields. (Mirrors T0.1 rejecting bool enum members.)
- **Patterns ARE enforced** (T0.1 records-not-enforces; the oracle enforces).
  `re.search` semantics (JSON-Schema `pattern`: match somewhere unless anchored);
  an un-compilable pattern is treated as unsatisfiable (value rejected).
- **Consumes ONLY the IR**: dispatch is on the `str`-valued `FieldKind` plus the
  domain descriptors; `validate.py` does **not** import `schema_ir`, so it stays
  pure-Python/host-side and in lockstep with T0.1's contract.

**Gotchas**
- Tests load both `schema_ir.py` and `validate.py` **by file path** (importlib),
  registering each in `sys.modules` before `exec_module`, to avoid the package
  `__init__` pulling in torch/vLLM. `validate.py` importing no `schema_ir` keeps
  that clean (no cross-module resolution needed).
- Import-hygiene gate is `ast`-based; the module docstring may name
  `torch_npu`/`triton` in prose.

**RED→GREEN evidence**
- RED (before `validate.py`): collection error
  `FileNotFoundError: .../validate.py`.
- GREEN: `python3 -m pytest -q --noconftest tests/ut/system_one/test_validate.py`
  → `30 passed`. Regression: `test_schema_ir.py` → `28 passed`.
  `ruff check vllm_ascend/system_one/validate.py tests/ut/system_one/test_validate.py`
  → clean.

### T0.3 — Benchmark harness + task-family loader — DONE (2026-09-18)

**Status**: complete. RED→GREEN, ruff clean (check + format), committed (pathspec).

**Files**
- `tools/system_one/__init__.py`, `tools/system_one/bench.py` — the harness.
- `tests/ut/system_one/test_bench.py` — 16 UTs.

**What it does (four PRD §7/§8 metric families on a PLUGGABLE runtime)**
- `run_benchmark(runtime, records, *, n_bins=10, timer=None) -> dict` times each
  `runtime.predict(context, schema)` call and emits a JSON-serializable report:
  `latency` (median/p95/mean/n), `accuracy` (exact-match + `per_field`),
  `validity` (valid_fraction + which validator ran), `calibration` (ECE +
  reliability bins). This report is the contract T0.4/T1.3/T2.6/T3.3 consume.
- **Latency**: `latency_stats(durations)` — `statistics.median`, nearest-rank p95
  (`rank = ceil(0.95·N)`), mean. `timer` is injectable so latency is deterministic
  under test.
- **Accuracy**: whole-record `exact_match` + `per_field_accuracy` keyed by IR leaf
  path (navigates nested value dicts by `$.a.b` path).
- **Validity**: prefers the T0.2 checker when importable, else an IR-driven
  fallback; report `validity.validator` is `"t0.2"` or `"fallback"`.
- **Calibration**: `expected_calibration_error(confidences, correct, n_bins)` and
  `reliability_bins(...)` implemented from scratch (equal-width bins over [0,1];
  ECE = Σ (|Bₘ|/N)·|acc−conf|). Calibration points are per **leaf field** per
  record (confidence vs field-matched-gold).

**Interfaces (downstream consumes these)**
- `Record(context: str, schema: dict, gold: dict)` — the `(context, schema,
  gold_value)` datum.
- `Prediction(value: dict, confidences: dict[str, float])` — `confidences` maps a
  leaf field **path** (`"$.tool"`) to a probability in [0,1] (scalar runtimes map
  every path to one value).
- `Runtime` protocol: `predict(context, schema) -> Prediction`.
- `StubRuntime(records, *, overrides=..., confidences=..., correct_conf=0.9,
  wrong_conf=0.3)` — deterministic gold-with-noise reference; matched by
  `context`; exercises all four metrics.
- `TaskFamily` protocol + `register_task_family(name, loader)` /
  `load_task_family(name) -> Iterator[Record]` (name validated eagerly).

**Task family / dataset decision (P0.3)**
- Shipped default: the in-repo synthetic **`intent_routing`** family (typed
  function-call / intent routing): unstructured context text + a tool/arg schema
  (enum tool id, bounded-int arg count, boolean confirm flag, bounded-string
  query) + the gold typed call. Fully offline, 6 fixture records; every schema
  compiles via T0.1 and every gold validates.
- A real dataset (BFCL-style function-calling, or a JSON-Schema extraction set)
  swaps in behind `register_task_family` by yielding `Record`s; the loader owns
  any download/caching — **the harness never fetches anything**.

**T0.2 consumption**
- The soft dependency is live: T0.2 `validate.py` landed concurrently and the
  harness auto-detected it (`validate_value(ir, value) -> ValidationResult`,
  adapted via its `__bool__`/`.path`); the report labels `validator: "t0.2"`.
  With `validate.py` absent the IR-driven fallback runs unchanged.

**Gotchas**
- The T0.1 IR compiler is loaded **by file path** (importlib, registered in
  `sys.modules` before `exec_module`), never `import vllm_ascend...`, so the
  harness stays pure-Python/host-side. Tests load `bench.py` the same way.
- Import-hygiene gate is `ast`-based; no `torch`/`torch_npu`/`triton`/`vllm` on
  the import path.
- `load_task_family` validates the name eagerly (before returning the iterator)
  so an unknown family raises `KeyError` at call time, not on first `next()`.

**RED→GREEN evidence**
- RED (before `bench.py`): collection error `FileNotFoundError: .../bench.py`.
- GREEN: `python3 -m pytest -q --noconftest tests/ut/system_one/test_bench.py`
  → `16 passed`. Regression: `test_schema_ir.py` → `28 passed`.
  `ruff check tools/system_one/ tests/ut/system_one/test_bench.py` → clean.

### T1.1 — Grammar-guided constrained decoder (interim path) — DONE (2026-09-18)

**Status**: complete. RED→GREEN, ruff clean, committed (pathspec).

**Files**
- `vllm_ascend/system_one/constrained_decode.py` — the interim constrained-AR
  masking engine (compiles the T0.1 IR to a character NFA + token mask).
- `tests/ut/system_one/test_constrained_decode.py` — 38 UTs (host-side,
  `--noconftest`), incl. the 100%-validity corpus under adversarial + random logits.

**What it does**
- Compiles the T0.1 constraint IR into a **token-level decoding constraint**: an
  incremental character NFA (Thompson-style build over `Lit`/`Alt`/`Seq`/bounded
  repeat) whose language is exactly the canonical-JSON documents that pass the
  T0.2 oracle. Applied step-by-step over a decoder's logits, disallowed tokens are
  **hard-masked to `-inf`** (never merely penalized) and the argmax over what
  remains is taken — so every emitted value is schema-valid **by construction**.

**API shape (what T1.2 / T1.3 consume)**
- `compile_constraint(ir) -> GrammarNFA` — IR → compiled character NFA.
- `ConstraintEngine(nfa)` — stateful matcher: `allowed_token_mask(vocab) ->
  list[bool]`, `advance(vocab, token_id)`, `advance_str(surface)`,
  `is_complete() -> bool`, `allowed_chars() -> frozenset[str]`, `text() -> str`.
- `constrained_decode(ir, logits_stream, tokenizer, *, max_steps=10_000) -> value`
  — argmax-over-allowed greedy driver; returns the parsed typed value.
- `apply_mask(logits, mask) -> list[float]` — pure, non-mutating; disallowed → `-inf`.
- **Token/vocab abstraction** (`SimpleVocab(tokens, eos_id)` + `char_vocab(alphabet)`):
  `tokens[i]` is the surface string token `i` emits; `tokens[eos_id]` is the stop
  token. The engine is charset-/token-agnostic — single-char *or* multi-char
  fragment vocabs both work (tested). This is the seam T1.2 adapts a real
  tokenizer to (and where XGrammar/llguidance/outlines later swap in).
- `logits_stream` is normalized (`_logits_source`) to accept a `callable(step) ->
  vector`, a single constant vector (reused each step), a sequence of per-step
  vectors, or an iterator of vectors.

**Termination / completeness**
- The EOS token is allowed **iff** `is_complete()` (the NFA accept state is live,
  i.e. a full `{...}` document has been emitted). The driver stops only when EOS
  wins the masked argmax, so it can never stop at a partial document, and required
  fields (structurally forced, no "absent" branch) are always present. Optional
  fields have an absent branch and may be skipped.

**Domain enforcement — token-wise vs. value-close (as required by the task)**
- **enum / boolean** — literal alternation; only members reachable, value-exact.
- **integer** — **enumerated** over `[ceil(min), floor(max)]` (bounded by
  `MAX_ENUM_LITERALS = 100_000`; a larger range raises `ConstraintCompileError`);
  range holds **exactly**, not digit-wise.
- **number** — enumerated candidate set: the integers in range plus the exact
  `minimum`/`maximum` literals; range holds exactly. The interim path does **not**
  synthesize arbitrary reals — full real-line coverage is deferred to the
  structured-head path (documented limitation).
- **string length** — enforced **token-wise**: the NFA admits at most `max_length`
  content chars (from a safe, escaping-free alphabet), then only the closing quote.
- **string pattern** — enforced by **enumerating bounded matching witnesses**
  (`re.search`-satisfying, length-bounded); the field becomes an alternation over
  those literals, so each already satisfies the T0.2 pattern + length checks.

**Decisions / gotchas (downstream must know)**
- Output surface is **canonical, whitespace-free JSON**, root fields in declaration
  order; nested objects recurse. `json.loads` on the finished string is the only
  parse and cannot fail (the grammar only accepts valid JSON).
- Does **not** import `schema_ir` / `validate`: duck-types the IR (dispatch on the
  `str`-valued `FieldKind`) so it stays host-side and in lockstep with T0.1.
- Import-hygiene: `ast`-gate test asserts no `torch`/`torch_npu`/`triton`/`vllm`
  imports. Pure Python, stdlib only.
- Numeric/pattern enumeration guarantees **validity** but limits *reachability*
  (e.g. NUMBER cannot emit every real); acceptable for the interim shippable path,
  replaced by structured heads / a digit-wise grammar later.

**RED→GREEN evidence**
- RED (before `constrained_decode.py`): collection error
  `FileNotFoundError: .../constrained_decode.py`.
- GREEN: `python3 -m pytest -q --noconftest tests/ut/system_one/test_constrained_decode.py`
  → `38 passed`. Regression: `test_schema_ir.py` + `test_validate.py` → `58 passed`
  (96 total across the three files). `ruff check
  vllm_ascend/system_one/constrained_decode.py tests/ut/system_one/test_constrained_decode.py`
  → clean.
