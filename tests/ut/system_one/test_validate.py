# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the System-One structured-validity checker (task T0.2).

These tests are the RED-first contract for ``vllm_ascend/system_one/validate.py``.
The checker is the **oracle** for the PRD's "100% schema-valid" gate: given a
compiled :class:`SchemaIR` (from T0.1) and a candidate value, it returns
valid/invalid with the *first offending field path* (and the full violation list).

The checker consumes **only the IR** (its ``FieldIR`` kinds + domain descriptors),
so it stays in lockstep with T0.1's contract and never re-parses the raw schema.

Both modules are loaded **by file path** (importlib), not via ``import
vllm_ascend...``, so the test stays pure-Python and host-side: no torch, no
torch_npu, no triton, no vLLM package ``__init__`` side effects.

Run: ``python3 -m pytest -q --noconftest tests/ut/system_one/test_validate.py``
"""

from __future__ import annotations

import importlib.util
import random
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SIR_PATH = _REPO_ROOT / "vllm_ascend" / "system_one" / "schema_ir.py"
_VAL_PATH = _REPO_ROOT / "vllm_ascend" / "system_one" / "validate.py"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None, f"cannot load {path}"
    module = importlib.util.module_from_spec(spec)
    # Register before exec so dataclass forward-ref resolution works on 3.12+.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


sir = _load_module("system_one_schema_ir", _SIR_PATH)
val = _load_module("system_one_validate", _VAL_PATH)

compile_schema = sir.compile_schema

validate_value = val.validate_value
is_valid = val.is_valid
ValidationResult = val.ValidationResult


# --------------------------------------------------------------------------- #
# Fixtures: a representative schema covering every supported kind + nesting.
# --------------------------------------------------------------------------- #
FULL_SCHEMA = {
    "type": "object",
    "properties": {
        "tool": {"enum": ["search", "calc", "noop"]},
        "priority": {"enum": [1, 2, 3]},
        "name": {"type": "string", "maxLength": 8},
        "age": {"type": "integer", "minimum": 0, "maximum": 120},
        "score": {"type": "number", "minimum": -1.0, "maximum": 1.0},
        "flag": {"type": "boolean"},
        "user": {
            "type": "object",
            "properties": {
                "id": {"type": "integer", "minimum": 0, "maximum": 1000},
                "role": {"enum": ["admin", "guest"]},
            },
            "required": ["id"],
        },
    },
    "required": ["tool", "age", "user"],
}


def _valid_value():
    return {
        "tool": "calc",
        "priority": 2,
        "name": "abc",
        "age": 30,
        "score": 0.5,
        "flag": True,
        "user": {"id": 7, "role": "admin"},
    }


# --------------------------------------------------------------------------- #
# ValidationResult shape / convenience API
# --------------------------------------------------------------------------- #
def test_valid_value_is_ok_with_no_violations():
    ir = compile_schema(FULL_SCHEMA)
    result = validate_value(ir, _valid_value())
    assert result.ok is True
    assert result.violations == ()
    assert result.path is None
    assert result.reason is None
    assert bool(result) is True


def test_is_valid_convenience_matches_result_ok():
    ir = compile_schema(FULL_SCHEMA)
    assert is_valid(ir, _valid_value()) is True
    bad = _valid_value()
    bad["age"] = 999
    assert is_valid(ir, bad) is False


# --------------------------------------------------------------------------- #
# Acceptance: every schema-valid value passes (small domains exhaustively).
# --------------------------------------------------------------------------- #
def test_all_enum_members_accepted():
    ir = compile_schema(FULL_SCHEMA)
    for tool in ("search", "calc", "noop"):
        for priority in (1, 2, 3):
            v = _valid_value()
            v["tool"] = tool
            v["priority"] = priority
            assert is_valid(ir, v), (tool, priority)


def test_boolean_domain_exhaustive():
    ir = compile_schema(FULL_SCHEMA)
    for flag in (True, False):
        v = _valid_value()
        v["flag"] = flag
        assert is_valid(ir, v)


def test_numeric_range_boundaries_accepted():
    ir = compile_schema(FULL_SCHEMA)
    for age in (0, 1, 60, 119, 120):
        v = _valid_value()
        v["age"] = age
        assert is_valid(ir, v), age
    for score in (-1.0, 0.0, 1.0):
        v = _valid_value()
        v["score"] = score
        assert is_valid(ir, v), score


def test_optional_fields_may_be_absent():
    ir = compile_schema(FULL_SCHEMA)
    # only the required fields present (tool, age, user); role optional in user.
    v = {"tool": "noop", "age": 5, "user": {"id": 0}}
    assert is_valid(ir, v)


def test_integer_is_accepted_for_number_field():
    ir = compile_schema(FULL_SCHEMA)
    v = _valid_value()
    v["score"] = 1  # int is a valid JSON number
    assert is_valid(ir, v)


# --------------------------------------------------------------------------- #
# Rejection: one violation class per test, checking the offending path.
# --------------------------------------------------------------------------- #
def _first(ir, value):
    result = validate_value(ir, value)
    assert result.ok is False
    assert result.violations, "expected at least one violation"
    return result


def test_reject_enum_non_member():
    ir = compile_schema(FULL_SCHEMA)
    v = _valid_value()
    v["tool"] = "delete"
    r = _first(ir, v)
    assert r.path == "$.tool"


def test_reject_out_of_range_integer():
    ir = compile_schema(FULL_SCHEMA)
    v = _valid_value()
    v["age"] = 121
    assert _first(ir, v).path == "$.age"
    v["age"] = -1
    assert _first(ir, v).path == "$.age"


def test_reject_out_of_range_number():
    ir = compile_schema(FULL_SCHEMA)
    v = _valid_value()
    v["score"] = 2.0
    assert _first(ir, v).path == "$.score"


def test_reject_wrong_type_string():
    ir = compile_schema(FULL_SCHEMA)
    v = _valid_value()
    v["name"] = 123
    assert _first(ir, v).path == "$.name"


def test_reject_true_for_integer_field_bool_is_not_int():
    ir = compile_schema(FULL_SCHEMA)
    v = _valid_value()
    v["age"] = True  # bool is a subclass of int but NOT a valid integer value
    assert _first(ir, v).path == "$.age"


def test_reject_true_for_number_field():
    ir = compile_schema(FULL_SCHEMA)
    v = _valid_value()
    v["score"] = False
    assert _first(ir, v).path == "$.score"


def test_reject_bool_for_enum_of_ints():
    ir = compile_schema(FULL_SCHEMA)
    v = _valid_value()
    v["priority"] = True  # True == 1 in Python, but is not an enum int member
    assert _first(ir, v).path == "$.priority"


def test_reject_non_bool_for_boolean_field():
    ir = compile_schema(FULL_SCHEMA)
    v = _valid_value()
    v["flag"] = 1
    assert _first(ir, v).path == "$.flag"


def test_reject_over_length_string():
    ir = compile_schema(FULL_SCHEMA)
    v = _valid_value()
    v["name"] = "waytoolongname"  # > maxLength 8
    assert _first(ir, v).path == "$.name"


def test_reject_missing_required_field():
    ir = compile_schema(FULL_SCHEMA)
    v = _valid_value()
    del v["age"]
    assert _first(ir, v).path == "$.age"


def test_reject_missing_nested_required_field():
    ir = compile_schema(FULL_SCHEMA)
    v = _valid_value()
    del v["user"]["id"]
    assert _first(ir, v).path == "$.user.id"


def test_reject_unknown_extra_key_strict():
    ir = compile_schema(FULL_SCHEMA)
    v = _valid_value()
    v["mystery"] = 1
    r = _first(ir, v)
    assert r.path == "$.mystery"


def test_reject_unknown_nested_extra_key_strict():
    ir = compile_schema(FULL_SCHEMA)
    v = _valid_value()
    v["user"]["ghost"] = 1
    assert _first(ir, v).path == "$.user.ghost"


def test_nested_object_violation_path_points_into_nesting():
    ir = compile_schema(FULL_SCHEMA)
    v = _valid_value()
    v["user"]["id"] = 99999  # out of [0, 1000]
    assert _first(ir, v).path == "$.user.id"
    v = _valid_value()
    v["user"]["role"] = "root"  # enum non-member
    assert _first(ir, v).path == "$.user.role"


def test_reject_non_dict_nested_object():
    ir = compile_schema(FULL_SCHEMA)
    v = _valid_value()
    v["user"] = "not-an-object"
    assert _first(ir, v).path == "$.user"


def test_reject_non_dict_root():
    ir = compile_schema(FULL_SCHEMA)
    r = validate_value(ir, ["not", "a", "dict"])
    assert r.ok is False
    assert r.path == "$"


# --------------------------------------------------------------------------- #
# Lax mode: extra keys tolerated (strict is the default / safer oracle).
# --------------------------------------------------------------------------- #
def test_lax_mode_tolerates_extra_keys():
    ir = compile_schema(FULL_SCHEMA)
    v = _valid_value()
    v["mystery"] = 1
    v["user"]["ghost"] = 2
    assert is_valid(ir, v) is False  # strict default rejects
    assert validate_value(ir, v, strict=False).ok is True  # lax tolerates


def test_lax_mode_still_enforces_domains():
    ir = compile_schema(FULL_SCHEMA)
    v = _valid_value()
    v["extra"] = 1
    v["age"] = 999  # still an out-of-range violation even in lax mode
    r = validate_value(ir, v, strict=False)
    assert r.ok is False
    assert r.path == "$.age"


# --------------------------------------------------------------------------- #
# Pattern enforcement (T0.2 decision: patterns ARE enforced, re.search semantics).
# --------------------------------------------------------------------------- #
def test_pattern_is_enforced():
    schema = {
        "type": "object",
        "properties": {"code": {"type": "string", "maxLength": 8, "pattern": "^[a-z]+$"}},
        "required": ["code"],
    }
    ir = compile_schema(schema)
    assert is_valid(ir, {"code": "abc"})
    r = validate_value(ir, {"code": "ABC"})
    assert r.ok is False
    assert r.path == "$.code"


def test_multiple_violations_collected_first_reported():
    ir = compile_schema(FULL_SCHEMA)
    v = _valid_value()
    v["tool"] = "bogus"  # violation 1
    v["age"] = 999  # violation 2
    r = validate_value(ir, v)
    assert r.ok is False
    assert len(r.violations) >= 2
    # first offending path is the first declared field in schema order.
    assert r.path == "$.tool"


# --------------------------------------------------------------------------- #
# Fuzz / property test: single-constraint mutations of a valid value.
# --------------------------------------------------------------------------- #
def _mutations():
    """Each returns a value that violates exactly one constraint."""
    muts = []

    def m_enum(v):
        v["tool"] = "not_a_tool"
        return v

    def m_int_hi(v):
        v["age"] = 121
        return v

    def m_int_lo(v):
        v["age"] = -5
        return v

    def m_bool_as_int(v):
        v["age"] = True
        return v

    def m_num_hi(v):
        v["score"] = 3.14
        return v

    def m_str_type(v):
        v["name"] = 5
        return v

    def m_str_len(v):
        v["name"] = "x" * 20
        return v

    def m_bool_type(v):
        v["flag"] = "yes"
        return v

    def m_missing_req(v):
        del v["tool"]
        return v

    def m_nested(v):
        v["user"]["id"] = 10**9
        return v

    def m_extra(v):
        v["surprise"] = 1
        return v

    muts.extend(
        [m_enum, m_int_hi, m_int_lo, m_bool_as_int, m_num_hi, m_str_type,
         m_str_len, m_bool_type, m_missing_req, m_nested, m_extra]
    )
    return muts


def test_fuzz_every_single_mutation_is_rejected():
    ir = compile_schema(FULL_SCHEMA)
    rng = random.Random(1234)
    muts = _mutations()
    for _ in range(500):
        mut = rng.choice(muts)
        v = mut(_valid_value())
        result = validate_value(ir, v)
        assert result.ok is False, f"{mut.__name__} should be invalid: {v}"
        assert result.path is not None


def test_fuzz_valid_permutations_all_accepted():
    ir = compile_schema(FULL_SCHEMA)
    rng = random.Random(99)
    for _ in range(500):
        v = {
            "tool": rng.choice(["search", "calc", "noop"]),
            "priority": rng.choice([1, 2, 3]),
            "name": "".join(rng.choice("abcdefgh") for _ in range(rng.randint(0, 8))),
            "age": rng.randint(0, 120),
            "score": rng.uniform(-1.0, 1.0),
            "flag": rng.choice([True, False]),
            "user": {"id": rng.randint(0, 1000)},
        }
        if rng.random() < 0.5:
            v["user"]["role"] = rng.choice(["admin", "guest"])
        assert is_valid(ir, v), v


# --------------------------------------------------------------------------- #
# Import hygiene: pure-Python, no NPU/Triton/torch on the import path.
# --------------------------------------------------------------------------- #
def test_module_has_no_npu_or_triton_imports():
    """AST-gate: no NPU/Triton/torch/vllm on the import path."""
    import ast

    forbidden_roots = {"triton", "torch", "torch_npu", "vllm"}
    tree = ast.parse(_VAL_PATH.read_text(encoding="utf-8"))
    imported_roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported_roots.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imported_roots.add(node.module.split(".")[0])
    offenders = forbidden_roots & imported_roots
    assert not offenders, f"validate.py must not import {sorted(offenders)}"
    for line in _VAL_PATH.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith(("import ", "from ")):
            assert "triton" not in stripped and "torch_npu" not in stripped, (
                f"forbidden import: {stripped!r}"
            )
