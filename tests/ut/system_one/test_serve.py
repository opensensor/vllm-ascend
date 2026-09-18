# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the System-One serving entry (task T1.2).

RED-first contract for ``vllm_ascend/system_one/serve.py``. The serving surface
accepts ``(context, schema)``, compiles the schema (T0.1), drives a pluggable
``DecodeBackend`` (the constrained-AR path today, the Phase-2 structured head
later — *without a call-site change*), validates the result through T0.2, and
returns a stable :class:`Decision` (``value`` / ``confidences`` / ``escalate``).

Properties tested:

* **End-to-end host run** — a schema + context + backend returns a schema-valid
  typed ``Decision.value`` (asserted against the T0.2 oracle ``is_valid``).
* **Gate** — a malformed / unsupported schema raises ``ServingError`` at
  ``decide`` (a clean surfaced gate error, not an uncaught deep decoder error).
* **Backend seam** — swapping ``MockBackend`` for ``ConstrainedARBackend`` yields
  the *same* ``Decision`` shape (proves the Phase-2 head swap point).
* **Decision shape stable** — ``value`` / ``confidences`` (placeholder ok) /
  ``escalate`` defaulting ``False``.
* **Defensive validation** — a backend that returns an invalid value is caught.
* **Import hygiene** — no ``triton`` / ``torch_npu`` / ``torch`` / ``vllm`` on the
  import path (ast-parsed).

Modules are loaded **by file path** (importlib), never ``import vllm_ascend...``,
so the tests stay pure-Python and host-side.

Run: ``python3 -m pytest -q --noconftest tests/ut/system_one/test_serve.py``
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SERVE_PATH = _REPO_ROOT / "vllm_ascend" / "system_one" / "serve.py"
_SIR_PATH = _REPO_ROOT / "vllm_ascend" / "system_one" / "schema_ir.py"
_VAL_PATH = _REPO_ROOT / "vllm_ascend" / "system_one" / "validate.py"
_CD_PATH = _REPO_ROOT / "vllm_ascend" / "system_one" / "constrained_decode.py"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None, f"cannot load {path}"
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


sir = _load("system_one_schema_ir", _SIR_PATH)
val = _load("system_one_validate", _VAL_PATH)
cd = _load("system_one_constrained_decode", _CD_PATH)
srv = _load("system_one_serve", _SERVE_PATH)

compile_schema = sir.compile_schema
is_valid = val.is_valid

SystemOneServer = srv.SystemOneServer
Decision = srv.Decision
ServingError = srv.ServingError
DecodeBackend = srv.DecodeBackend
MockBackend = srv.MockBackend
ConstrainedARBackend = srv.ConstrainedARBackend
serve = srv.serve
char_vocab = srv.char_vocab


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
# A typed function-call / intent-routing schema (the plan's initial task family):
# an enum tool id, a small bounded int, and a boolean confirm flag. All required
# so the constrained-AR walk emits every field deterministically.
SCHEMA = {
    "type": "object",
    "properties": {
        "tool": {"enum": ["search", "email", "calc"]},
        "count": {"type": "integer", "minimum": 0, "maximum": 5},
        "confirm": {"type": "boolean"},
    },
    "required": ["tool", "count", "confirm"],
}

CONTEXT = "please search the archive twice and do not ask me to confirm"

# A broad single-char alphabet covering every structural + payload character the
# canonical JSON for SCHEMA needs, plus a trailing EOS (empty surface).
_ALPHABET = (
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    ' -.{}":,_'
)


def _zero_logits_fn(context, step):
    """A benign mock logits backend: constant logits (the mask forces validity)."""
    return [0.0] * (len(_ALPHABET) + 1)


def _ar_backend():
    vocab = char_vocab(_ALPHABET)
    return ConstrainedARBackend(_zero_logits_fn, vocab)


# --------------------------------------------------------------------------- #
# End-to-end host run
# --------------------------------------------------------------------------- #
def test_decide_returns_schema_valid_typed_value_mock_backend():
    ir = compile_schema(SCHEMA)
    gold = {"tool": "email", "count": 3, "confirm": True}
    assert is_valid(ir, gold)  # sanity: the gold is valid

    server = SystemOneServer(MockBackend(value=gold))
    decision = server.decide(CONTEXT, SCHEMA)

    assert isinstance(decision, Decision)
    assert decision.value == gold
    assert is_valid(compile_schema(SCHEMA), decision.value)
    assert decision.escalate is False


def test_end_to_end_constrained_ar_backend_returns_valid_value():
    server = SystemOneServer(_ar_backend())
    decision = server.decide(CONTEXT, SCHEMA)

    # The constrained-AR path emits a canonical-JSON doc valid by construction.
    assert isinstance(decision.value, dict)
    assert is_valid(compile_schema(SCHEMA), decision.value)
    # every required leaf is present
    assert set(decision.value) == {"tool", "count", "confirm"}
    assert decision.value["tool"] in ("search", "email", "calc")
    assert 0 <= decision.value["count"] <= 5
    assert isinstance(decision.value["confirm"], bool)
    assert decision.escalate is False


def test_serve_convenience_function_matches_server():
    backend = MockBackend(value={"tool": "calc", "count": 1, "confirm": False})
    d1 = serve(CONTEXT, SCHEMA, backend)
    d2 = SystemOneServer(backend).decide(CONTEXT, SCHEMA)
    assert d1 == d2
    assert is_valid(compile_schema(SCHEMA), d1.value)


# --------------------------------------------------------------------------- #
# Gate: malformed / unsupported schema rejected cleanly at decide
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "bad_schema",
    [
        {"type": "array"},  # non-object root
        {"type": "object", "properties": {"x": {"type": "array"}}},  # unsupported field
        {"type": "object", "properties": {"s": {"type": "string"}}},  # unbounded string
        {"type": "object"},  # object without properties
        {"type": "object", "properties": {"n": {"type": "integer"}}},  # unbounded numeric
    ],
)
def test_malformed_schema_rejected_at_gate(bad_schema):
    server = SystemOneServer(MockBackend(value={"anything": 1}))
    with pytest.raises(ServingError):
        server.decide(CONTEXT, bad_schema)


def test_gate_error_is_surfaced_not_deep():
    # The gate error is the serving-level ServingError, raised before the backend
    # is ever driven — a never-called backend proves the gate short-circuits.
    class ExplodingBackend:
        def decode(self, ir, context):  # pragma: no cover - must not run
            raise AssertionError("backend must not be reached on a malformed schema")

    server = SystemOneServer(ExplodingBackend())
    with pytest.raises(ServingError):
        server.decide(CONTEXT, {"type": "array"})


# --------------------------------------------------------------------------- #
# Backend seam: swap MockBackend <-> ConstrainedARBackend, same Decision shape
# --------------------------------------------------------------------------- #
def test_backend_seam_swap_yields_same_decision_shape():
    # Drive the real constrained-AR backend first, capture its value, then feed
    # that exact value through the MockBackend — the two Decisions are identical,
    # proving the Phase-2 head backend can swap in without a call-site change.
    ar_decision = SystemOneServer(_ar_backend()).decide(CONTEXT, SCHEMA)
    mock_decision = SystemOneServer(MockBackend(value=ar_decision.value)).decide(CONTEXT, SCHEMA)

    assert ar_decision == mock_decision
    # Structural shape is stable across the swap.
    assert set(vars(ar_decision)) == set(vars(mock_decision))
    assert ar_decision.confidences.keys() == mock_decision.confidences.keys()
    assert ar_decision.escalate == mock_decision.escalate is False


def test_backends_satisfy_the_decode_backend_protocol():
    assert isinstance(MockBackend(value={}), DecodeBackend)
    assert isinstance(_ar_backend(), DecodeBackend)


# --------------------------------------------------------------------------- #
# Decision shape stability (confidences placeholder + escalate default)
# --------------------------------------------------------------------------- #
def test_decision_shape_is_stable():
    d = Decision(value={"tool": "search", "count": 0, "confirm": True})
    assert d.value == {"tool": "search", "count": 0, "confirm": True}
    assert d.escalate is False
    assert hasattr(d, "confidences")
    # confidences is a mapping (placeholder for T2.5 calibration), default empty.
    assert dict(d.confidences) == {}


def test_confidences_placeholder_keyed_by_leaf_paths():
    decision = SystemOneServer(
        MockBackend(value={"tool": "search", "count": 2, "confirm": False})
    ).decide(CONTEXT, SCHEMA)
    # Placeholder confidences carry the stable shape T2.5 fills: one entry per
    # declared leaf path. Values are placeholders (uncalibrated).
    assert set(decision.confidences) == {"$.tool", "$.count", "$.confirm"}


# --------------------------------------------------------------------------- #
# Defensive validation: an invalid backend value is caught
# --------------------------------------------------------------------------- #
def test_invalid_backend_value_is_caught():
    # enum non-member -> out of domain -> T0.2 rejects -> ServingError.
    bad = MockBackend(value={"tool": "NOPE", "count": 2, "confirm": True})
    server = SystemOneServer(bad)
    with pytest.raises(ServingError):
        server.decide(CONTEXT, SCHEMA)


def test_invalid_backend_value_out_of_range_is_caught():
    bad = MockBackend(value={"tool": "search", "count": 99, "confirm": True})
    with pytest.raises(ServingError):
        SystemOneServer(bad).decide(CONTEXT, SCHEMA)


# --------------------------------------------------------------------------- #
# Import-hygiene ast-gate
# --------------------------------------------------------------------------- #
def test_module_has_no_npu_or_triton_imports():
    """Grep-gate: no NPU/Triton/torch/vLLM on the import path (ast-parsed)."""
    import ast

    forbidden_roots = {"triton", "torch", "torch_npu", "vllm"}
    tree = ast.parse(_SERVE_PATH.read_text(encoding="utf-8"))
    imported_roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported_roots.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imported_roots.add(node.module.split(".")[0])
    offenders = forbidden_roots & imported_roots
    assert not offenders, f"serve.py must not import {sorted(offenders)}"
    for line in _SERVE_PATH.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith(("import ", "from ")):
            assert "triton" not in stripped and "torch_npu" not in stripped, (
                f"forbidden import: {stripped!r}"
            )
