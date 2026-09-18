# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the System-One grammar-guided constrained decoder (task T1.1).

RED-first contract for ``vllm_ascend/system_one/constrained_decode.py``. The
decoder compiles the T0.1 constraint IR into a *token-level decoding constraint*
(an incremental character NFA + token mask) so that any argmax-over-allowed
autoregressive walk emits a canonical JSON document that **parses into a
schema-valid value** — proven against the T0.2 oracle (``is_valid``).

Key properties tested:

* **Masking, not penalizing** — disallowed tokens become ``-inf``, so an
  adversarial logits stream that *wants* an invalid token is forced to a valid one.
* **100% validity** — a corpus of schemas (enum / bool / bounded string /
  ranged int+number / nested object / optional fields) driven by adversarial and
  random logits: every decoded value passes ``is_valid``.
* **Enum**: only members reachable; a non-member is impossible for any logits.
* **Numeric range + string length** bounds respected.
* **Completeness**: ``is_complete()`` is true only at a valid terminal; required
  fields always present, optional fields may be skipped.

Modules are loaded **by file path** (importlib), never ``import vllm_ascend...``,
so the tests stay pure-Python and host-side: no torch, no torch_npu, no triton,
no vLLM package ``__init__`` side effects.

Run: ``python3 -m pytest -q --noconftest tests/ut/system_one/test_constrained_decode.py``
"""

from __future__ import annotations

import importlib.util
import random
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
_CD_PATH = _REPO_ROOT / "vllm_ascend" / "system_one" / "constrained_decode.py"
_SIR_PATH = _REPO_ROOT / "vllm_ascend" / "system_one" / "schema_ir.py"
_VAL_PATH = _REPO_ROOT / "vllm_ascend" / "system_one" / "validate.py"


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

compile_schema = sir.compile_schema
is_valid = val.is_valid

compile_constraint = cd.compile_constraint
ConstraintEngine = cd.ConstraintEngine
SimpleVocab = cd.SimpleVocab
char_vocab = cd.char_vocab
apply_mask = cd.apply_mask
constrained_decode = cd.constrained_decode
NEG_INF = cd.NEG_INF


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
# A broad character alphabet covering every structural + payload character our
# corpus needs, plus a trailing EOS (empty surface).
_ALPHABET = (
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    " -.{}\":,_"
)


def _vocab():
    return char_vocab(_ALPHABET)


def _prefer(vocab, wanted: str, *, high: float = 1000.0):
    """Constant logits vector that maximally prefers the token whose surface is
    ``wanted`` (and, secondarily, any content-ish token) — an adversary."""
    vec = [0.0] * len(vocab.tokens)
    for i, surf in enumerate(vocab.tokens):
        if surf == wanted:
            vec[i] = high
    return lambda step: vec


def _random_logits(vocab, seed: int):
    rng = random.Random(seed)
    return lambda step: [rng.uniform(-5.0, 5.0) for _ in vocab.tokens]


# A corpus of schemas exercising every supported kind.
_CORPUS = {
    "enum_str": {
        "type": "object",
        "properties": {"color": {"enum": ["red", "green", "blue"]}},
        "required": ["color"],
    },
    "enum_int": {
        "type": "object",
        "properties": {"code": {"enum": [1, 7, 42]}},
        "required": ["code"],
    },
    "boolean": {
        "type": "object",
        "properties": {"flag": {"type": "boolean"}},
        "required": ["flag"],
    },
    "int_range": {
        "type": "object",
        "properties": {"n": {"type": "integer", "minimum": 3, "maximum": 9}},
        "required": ["n"],
    },
    "neg_int_range": {
        "type": "object",
        "properties": {"t": {"type": "integer", "minimum": -4, "maximum": 4}},
        "required": ["t"],
    },
    "number_range": {
        "type": "object",
        "properties": {"x": {"type": "number", "minimum": 0.0, "maximum": 1.0}},
        "required": ["x"],
    },
    "bounded_string": {
        "type": "object",
        "properties": {"name": {"type": "string", "maxLength": 4}},
        "required": ["name"],
    },
    "optional_mix": {
        "type": "object",
        "properties": {
            "id": {"type": "integer", "minimum": 0, "maximum": 5},
            "label": {"type": "string", "maxLength": 3},
            "on": {"type": "boolean"},
        },
        "required": ["id"],
    },
    "nested": {
        "type": "object",
        "properties": {
            "tool": {"enum": ["search", "calc"]},
            "args": {
                "type": "object",
                "properties": {
                    "k": {"type": "integer", "minimum": 1, "maximum": 3},
                    "q": {"type": "string", "maxLength": 3},
                },
                "required": ["k"],
            },
        },
        "required": ["tool", "args"],
    },
}


# --------------------------------------------------------------------------- #
# Masking, not penalizing
# --------------------------------------------------------------------------- #
def test_disallowed_tokens_are_masked_to_neg_inf():
    vocab = _vocab()
    ir = compile_schema(_CORPUS["enum_str"])
    eng = ConstraintEngine(compile_constraint(ir))

    mask = eng.allowed_token_mask(vocab)
    # At the very start only '{' may be emitted (canonical object opens with it).
    open_id = vocab.index("{")
    assert mask[open_id] is True
    assert mask[vocab.eos_id] is False  # not complete at the start

    logits = [1.0] * len(vocab.tokens)
    masked = apply_mask(logits, mask)
    for i, allowed in enumerate(mask):
        if allowed:
            assert masked[i] == logits[i]
        else:
            assert masked[i] == NEG_INF
    # It is a hard mask: the disallowed max is -inf, never a mere penalty.
    assert max(masked) == 1.0


def test_apply_mask_returns_new_list_without_mutating():
    vocab = _vocab()
    logits = [1.0] * len(vocab.tokens)
    mask = [False] * len(vocab.tokens)
    mask[0] = True
    masked = apply_mask(logits, mask)
    assert masked is not logits
    assert logits == [1.0] * len(vocab.tokens)  # unchanged
    assert masked[0] == 1.0 and masked[1] == NEG_INF


# --------------------------------------------------------------------------- #
# Enum: only members reachable
# --------------------------------------------------------------------------- #
def test_enum_non_member_is_impossible_for_any_logits():
    vocab = _vocab()
    ir = compile_schema(_CORPUS["enum_str"])
    # Adversary always wants 'p' (as in "purple"), which is not a member.
    value = constrained_decode(ir, _prefer(vocab, "p"), vocab)
    assert value["color"] in ("red", "green", "blue")
    assert is_valid(ir, value)


def test_enum_all_members_are_reachable():
    vocab = _vocab()
    ir = compile_schema(_CORPUS["enum_str"])
    for member in ("red", "green", "blue"):
        # Prefer the member's own first char to steer toward it.
        got = constrained_decode(ir, _prefer(vocab, member[0]), vocab)
        assert got["color"] == member
        assert is_valid(ir, got)


def test_enum_int_members_only():
    vocab = _vocab()
    ir = compile_schema(_CORPUS["enum_int"])
    value = constrained_decode(ir, _prefer(vocab, "9"), vocab)  # 9 is not a member
    assert value["code"] in (1, 7, 42)
    assert is_valid(ir, value)


# --------------------------------------------------------------------------- #
# Boolean
# --------------------------------------------------------------------------- #
def test_boolean_only_true_false():
    vocab = _vocab()
    ir = compile_schema(_CORPUS["boolean"])
    for wanted, expect in (("t", True), ("f", False)):
        value = constrained_decode(ir, _prefer(vocab, wanted), vocab)
        assert value["flag"] is expect
        assert is_valid(ir, value)


# --------------------------------------------------------------------------- #
# Numeric range
# --------------------------------------------------------------------------- #
def test_integer_range_respected_against_adversary():
    vocab = _vocab()
    ir = compile_schema(_CORPUS["int_range"])
    # Adversary wants '9' everywhere; range is [3, 9] so 9 is allowed, but the
    # point is the value must stay in range regardless.
    value = constrained_decode(ir, _prefer(vocab, "8"), vocab)
    assert 3 <= value["n"] <= 9
    assert is_valid(ir, value)


def test_integer_range_upper_bound_not_exceeded():
    vocab = _vocab()
    # range [3, 7]; adversary wants '9' -> must be masked; result stays <= 7.
    ir = compile_schema(
        {
            "type": "object",
            "properties": {"n": {"type": "integer", "minimum": 3, "maximum": 7}},
            "required": ["n"],
        }
    )
    value = constrained_decode(ir, _prefer(vocab, "9"), vocab)
    assert 3 <= value["n"] <= 7
    assert is_valid(ir, value)


def test_negative_integer_range():
    vocab = _vocab()
    ir = compile_schema(_CORPUS["neg_int_range"])
    value = constrained_decode(ir, _prefer(vocab, "-"), vocab)
    assert -4 <= value["t"] <= 4
    assert is_valid(ir, value)


def test_number_range_respected():
    vocab = _vocab()
    ir = compile_schema(_CORPUS["number_range"])
    value = constrained_decode(ir, _prefer(vocab, "9"), vocab)  # 9 out of [0,1]
    assert 0.0 <= float(value["x"]) <= 1.0
    assert is_valid(ir, value)


# --------------------------------------------------------------------------- #
# Bounded string length
# --------------------------------------------------------------------------- #
def test_bounded_string_length_not_exceeded():
    vocab = _vocab()
    ir = compile_schema(_CORPUS["bounded_string"])
    # Adversary always wants a content char 'a' and never the closing quote:
    # the grammar must force the quote at maxLength.
    value = constrained_decode(ir, _prefer(vocab, "a"), vocab)
    assert isinstance(value["name"], str)
    assert len(value["name"]) <= 4
    assert is_valid(ir, value)


def test_empty_string_is_reachable_when_bounded():
    vocab = _vocab()
    ir = compile_schema(_CORPUS["bounded_string"])
    # Adversary wants to close immediately (prefer the quote char).
    value = constrained_decode(ir, _prefer(vocab, '"'), vocab)
    assert value["name"] == ""
    assert is_valid(ir, value)


# --------------------------------------------------------------------------- #
# Completeness / required fields
# --------------------------------------------------------------------------- #
def test_is_complete_only_at_terminal():
    ir = compile_schema(_CORPUS["boolean"])
    eng = ConstraintEngine(compile_constraint(ir))
    assert not eng.is_complete()
    for ch in '{"flag":true':
        assert not eng.is_complete()
        eng.advance_str(ch)
    assert not eng.is_complete()  # still needs the closing brace
    eng.advance_str("}")
    assert eng.is_complete()


def test_eos_masked_until_complete():
    vocab = _vocab()
    ir = compile_schema(_CORPUS["boolean"])
    # Force EOS every step: it must never be selectable until the doc is complete.
    force_eos = lambda step: [
        1000.0 if i == vocab.eos_id else 0.0 for i in range(len(vocab.tokens))
    ]
    value = constrained_decode(ir, force_eos, vocab)
    assert is_valid(ir, value)  # decoder could not stop early


def test_required_fields_always_present():
    vocab = _vocab()
    ir = compile_schema(_CORPUS["optional_mix"])
    # Adversary tries to close the object immediately after '{'.
    value = constrained_decode(ir, _prefer(vocab, "}"), vocab)
    assert "id" in value  # required
    assert is_valid(ir, value)


def test_optional_fields_may_be_skipped():
    vocab = _vocab()
    ir = compile_schema(_CORPUS["optional_mix"])
    value = constrained_decode(ir, _prefer(vocab, "}"), vocab)
    # With an adversary that wants to close ASAP, optionals are skipped.
    assert set(value.keys()) == {"id"}
    assert is_valid(ir, value)


# --------------------------------------------------------------------------- #
# Nested objects
# --------------------------------------------------------------------------- #
def test_nested_object_is_valid():
    vocab = _vocab()
    ir = compile_schema(_CORPUS["nested"])
    value = constrained_decode(ir, _prefer(vocab, "x"), vocab)
    assert isinstance(value["args"], dict)
    assert "k" in value["args"]  # nested required
    assert is_valid(ir, value)


# --------------------------------------------------------------------------- #
# Fragment (multi-char token) vocab: engine is token-agnostic
# --------------------------------------------------------------------------- #
def test_fragment_vocab_multichar_tokens():
    # Tokens are multi-character fragments, not single chars.
    frags = ["{", "}", '"flag"', ":", "true", "false", ","]
    vocab = SimpleVocab(frags + [""], eos_id=len(frags))
    ir = compile_schema(_CORPUS["boolean"])
    prefer_false = lambda step: [
        (1000.0 if vocab.tokens[i] == "false" else 0.0)
        for i in range(len(vocab.tokens))
    ]
    value = constrained_decode(ir, prefer_false, vocab)
    assert value == {"flag": False}
    assert is_valid(ir, value)


# --------------------------------------------------------------------------- #
# HEADLINE: 100% validity over the corpus under adversarial + random logits
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("schema_name", sorted(_CORPUS))
def test_corpus_100pct_valid_under_random_logits(schema_name):
    vocab = _vocab()
    ir = compile_schema(_CORPUS[schema_name])
    for seed in range(40):
        value = constrained_decode(ir, _random_logits(vocab, seed), vocab)
        assert is_valid(ir, value), (schema_name, seed, value)


@pytest.mark.parametrize("schema_name", sorted(_CORPUS))
def test_corpus_100pct_valid_under_adversarial_single_char(schema_name):
    vocab = _vocab()
    ir = compile_schema(_CORPUS[schema_name])
    # Point the adversary at every alphabet character in turn.
    for ch in _ALPHABET:
        if ch == "":
            continue
        value = constrained_decode(ir, _prefer(vocab, ch), vocab)
        assert is_valid(ir, value), (schema_name, ch, value)


# --------------------------------------------------------------------------- #
# Engine-level: allowed_token_mask never strands the decoder
# --------------------------------------------------------------------------- #
def test_mask_always_offers_a_move_until_complete():
    vocab = _vocab()
    for name in _CORPUS:
        ir = compile_schema(_CORPUS[name])
        eng = ConstraintEngine(compile_constraint(ir))
        steps = 0
        while not eng.is_complete() or steps == 0:
            mask = eng.allowed_token_mask(vocab)
            # There is always at least one allowed real token OR eos (if complete).
            assert any(mask), (name, eng.text())
            # Take the first allowed non-eos token deterministically.
            nxt = None
            for i, ok in enumerate(mask):
                if ok and i != vocab.eos_id:
                    nxt = i
                    break
            if nxt is None:
                assert eng.is_complete()
                break
            eng.advance(vocab, nxt)
            steps += 1
            if steps > 200:
                pytest.fail(f"did not terminate for {name}: {eng.text()!r}")
        assert eng.is_complete()
        assert is_valid(ir, __import__("json").loads(eng.text()))


# --------------------------------------------------------------------------- #
# Import-hygiene ast-gate
# --------------------------------------------------------------------------- #
def test_module_has_no_npu_or_triton_imports():
    """Grep-gate: no NPU/Triton/torch on the import path (ast-parsed)."""
    import ast

    forbidden_roots = {"triton", "torch", "torch_npu", "vllm"}
    tree = ast.parse(_CD_PATH.read_text(encoding="utf-8"))
    imported_roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported_roots.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imported_roots.add(node.module.split(".")[0])
    offenders = forbidden_roots & imported_roots
    assert not offenders, f"constrained_decode.py must not import {sorted(offenders)}"
    for line in _CD_PATH.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith(("import ", "from ")):
            assert "triton" not in stripped and "torch_npu" not in stripped, (
                f"forbidden import: {stripped!r}"
            )
