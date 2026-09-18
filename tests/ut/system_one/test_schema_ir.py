# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the System-One schema -> constraint IR compiler (task T0.1).

These tests are the RED-first contract for ``vllm_ascend/system_one/schema_ir.py``.
The compiler turns a JSON-Schema *subset* into a normalized, immutable constraint
IR that both the Phase-1 constrained-AR path and the Phase-2 structured heads
consume. The IR must:

* compile each supported kind (enum / bounded string / bounded int / bounded
  number / boolean / nested object / optional) to the right ``FieldIR``,
* capture required vs optional and nested records as nested ``SchemaIR``,
* round-trip every field back to a validator descriptor (the property T0.2 relies
  on), and
* reject unsupported constructs *loudly* with the offending path.

The module is loaded **directly by file path** (not via ``import
vllm_ascend...``) so the test stays pure-Python and host-side: no torch, no
torch_npu, no triton, no vLLM package ``__init__`` side effects.

Run: ``python3 -m pytest -q --noconftest tests/ut/system_one/test_schema_ir.py``
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
_MODULE_PATH = _REPO_ROOT / "vllm_ascend" / "system_one" / "schema_ir.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("system_one_schema_ir", _MODULE_PATH)
    assert spec is not None and spec.loader is not None, f"cannot load {_MODULE_PATH}"
    module = importlib.util.module_from_spec(spec)
    # Register before exec so dataclass forward-ref resolution (e.g. the nested
    # ``SchemaIR`` reference) can find the module in ``sys.modules`` on 3.12+.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


sir = _load_module()

FieldKind = sir.FieldKind
compile_schema = sir.compile_schema
SchemaIR = sir.SchemaIR
FieldIR = sir.FieldIR
UnsupportedSchemaError = sir.UnsupportedSchemaError
SchemaCompileError = sir.SchemaCompileError


# --------------------------------------------------------------------------- #
# Per-kind compilation
# --------------------------------------------------------------------------- #
def _field_by_name(ir, name):
    for field in ir.fields:
        if field.name == name:
            return field
    raise AssertionError(f"field {name!r} not found in IR")


def test_enum_field_compiles_to_enum_domain():
    schema = {
        "type": "object",
        "properties": {"tool": {"enum": ["search", "calc", "noop"]}},
        "required": ["tool"],
    }
    ir = compile_schema(schema)
    field = _field_by_name(ir, "tool")
    assert field.kind is FieldKind.ENUM
    assert tuple(field.domain.members) == ("search", "calc", "noop")
    assert field.required is True


def test_enum_of_ints_is_supported():
    schema = {"type": "object", "properties": {"k": {"enum": [1, 2, 3]}}}
    field = _field_by_name(compile_schema(schema), "k")
    assert field.kind is FieldKind.ENUM
    assert tuple(field.domain.members) == (1, 2, 3)


def test_bounded_string_records_length_and_optional_pattern():
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string", "maxLength": 32, "pattern": "^[a-z]+$"}
        },
    }
    field = _field_by_name(compile_schema(schema), "name")
    assert field.kind is FieldKind.STRING
    assert field.domain.max_length == 32
    assert field.domain.pattern == "^[a-z]+$"
    assert field.required is False  # not in a required list


def test_integer_field_exposes_numeric_range():
    schema = {
        "type": "object",
        "properties": {"age": {"type": "integer", "minimum": 0, "maximum": 120}},
        "required": ["age"],
    }
    field = _field_by_name(compile_schema(schema), "age")
    assert field.kind is FieldKind.INTEGER
    assert field.domain.minimum == 0
    assert field.domain.maximum == 120
    assert field.domain.integral is True


def test_number_field_exposes_numeric_range():
    schema = {
        "type": "object",
        "properties": {"score": {"type": "number", "minimum": -1.0, "maximum": 1.0}},
    }
    field = _field_by_name(compile_schema(schema), "score")
    assert field.kind is FieldKind.NUMBER
    assert field.domain.minimum == -1.0
    assert field.domain.maximum == 1.0
    assert field.domain.integral is False


def test_boolean_field():
    schema = {"type": "object", "properties": {"flag": {"type": "boolean"}}}
    field = _field_by_name(compile_schema(schema), "flag")
    assert field.kind is FieldKind.BOOLEAN


# --------------------------------------------------------------------------- #
# Nested records + required/optional
# --------------------------------------------------------------------------- #
def test_nested_object_becomes_nested_schema_ir():
    schema = {
        "type": "object",
        "properties": {
            "user": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer", "minimum": 0, "maximum": 1_000_000},
                    "role": {"enum": ["admin", "guest"]},
                },
                "required": ["id"],
            }
        },
        "required": ["user"],
    }
    ir = compile_schema(schema)
    user = _field_by_name(ir, "user")
    assert user.kind is FieldKind.OBJECT
    assert user.required is True
    nested = user.domain.schema
    assert isinstance(nested, SchemaIR)
    nid = _field_by_name(nested, "id")
    nrole = _field_by_name(nested, "role")
    assert nid.kind is FieldKind.INTEGER and nid.required is True
    assert nrole.kind is FieldKind.ENUM and nrole.required is False


def test_field_order_is_preserved():
    schema = {
        "type": "object",
        "properties": {
            "a": {"type": "boolean"},
            "b": {"type": "boolean"},
            "c": {"type": "boolean"},
        },
    }
    ir = compile_schema(schema)
    assert [f.name for f in ir.fields] == ["a", "b", "c"]


def test_paths_are_recorded_for_nested_fields():
    schema = {
        "type": "object",
        "properties": {
            "user": {
                "type": "object",
                "properties": {"role": {"enum": ["a", "b"]}},
            }
        },
    }
    ir = compile_schema(schema)
    user = _field_by_name(ir, "user")
    role = _field_by_name(user.domain.schema, "role")
    assert role.path == "$.user.role"


# --------------------------------------------------------------------------- #
# Leaf iteration (head sizing) + round-trip to validator descriptor (T0.2)
# --------------------------------------------------------------------------- #
def test_leaf_fields_flatten_nested_objects():
    schema = {
        "type": "object",
        "properties": {
            "tool": {"enum": ["x", "y"]},
            "args": {
                "type": "object",
                "properties": {
                    "n": {"type": "integer", "minimum": 0, "maximum": 9},
                    "s": {"type": "string", "maxLength": 8},
                },
            },
        },
    }
    ir = compile_schema(schema)
    leaf_paths = [f.path for f in ir.leaf_fields()]
    # the object field itself is NOT a leaf; its children are.
    assert leaf_paths == ["$.tool", "$.args.n", "$.args.s"]
    assert all(f.is_leaf() for f in ir.leaf_fields())


def test_round_trip_validator_descriptor_enum():
    schema = {"type": "object", "properties": {"t": {"enum": ["a", "b", "c"]}}}
    field = _field_by_name(compile_schema(schema), "t")
    desc = field.validator_descriptor()
    assert desc["kind"] == "enum"
    assert desc["members"] == ["a", "b", "c"]
    assert desc["path"] == "$.t"


def test_round_trip_validator_descriptor_numeric():
    schema = {
        "type": "object",
        "properties": {"n": {"type": "integer", "minimum": 3, "maximum": 7}},
    }
    field = _field_by_name(compile_schema(schema), "n")
    desc = field.validator_descriptor()
    assert desc["kind"] == "integer"
    assert desc["minimum"] == 3
    assert desc["maximum"] == 7


def test_every_leaf_maps_back_to_a_validator_descriptor():
    schema = {
        "type": "object",
        "properties": {
            "tool": {"enum": ["x", "y"]},
            "conf": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "meta": {
                "type": "object",
                "properties": {
                    "flag": {"type": "boolean"},
                    "label": {"type": "string", "maxLength": 4},
                },
            },
        },
    }
    ir = compile_schema(schema)
    descriptors = ir.validator_descriptors()
    # one descriptor per leaf, each carries kind + path + a domain key.
    assert len(descriptors) == len(list(ir.leaf_fields()))
    kinds = {d["path"]: d["kind"] for d in descriptors}
    assert kinds == {
        "$.tool": "enum",
        "$.conf": "number",
        "$.meta.flag": "boolean",
        "$.meta.label": "string",
    }
    for desc in descriptors:
        assert "kind" in desc and "path" in desc and "required" in desc


def test_ir_is_immutable():
    schema = {"type": "object", "properties": {"flag": {"type": "boolean"}}}
    ir = compile_schema(schema)
    with pytest.raises((AttributeError, Exception)):
        ir.fields = ()  # type: ignore[misc]
    with pytest.raises((AttributeError, Exception)):
        ir.fields[0].name = "nope"  # type: ignore[misc]


# --------------------------------------------------------------------------- #
# Loud rejection of unsupported constructs (with offending path)
# --------------------------------------------------------------------------- #
def test_reject_oneof():
    schema = {
        "type": "object",
        "properties": {"x": {"oneOf": [{"type": "boolean"}]}},
    }
    with pytest.raises(UnsupportedSchemaError) as exc:
        compile_schema(schema)
    assert "$.x" in str(exc.value)
    assert exc.value.path == "$.x"


def test_reject_anyof_and_allof_and_not():
    for combinator in ("anyOf", "allOf", "not"):
        schema = {
            "type": "object",
            "properties": {"x": {combinator: [{"type": "boolean"}]}},
        }
        with pytest.raises(UnsupportedSchemaError):
            compile_schema(schema)


def test_reject_ref():
    schema = {
        "type": "object",
        "properties": {"x": {"$ref": "#/definitions/Foo"}},
    }
    with pytest.raises(UnsupportedSchemaError) as exc:
        compile_schema(schema)
    assert "$.x" in str(exc.value)


def test_reject_unbounded_string():
    schema = {"type": "object", "properties": {"s": {"type": "string"}}}
    with pytest.raises(UnsupportedSchemaError) as exc:
        compile_schema(schema)
    assert "$.s" in str(exc.value)


def test_reject_unbounded_numeric():
    # missing maximum
    schema = {
        "type": "object",
        "properties": {"n": {"type": "integer", "minimum": 0}},
    }
    with pytest.raises(UnsupportedSchemaError) as exc:
        compile_schema(schema)
    assert "$.n" in str(exc.value)


def test_reject_array_of_arbitrary():
    schema = {
        "type": "object",
        "properties": {"xs": {"type": "array", "items": {"type": "string"}}},
    }
    with pytest.raises(UnsupportedSchemaError) as exc:
        compile_schema(schema)
    assert "$.xs" in str(exc.value)


def test_reject_unknown_type():
    schema = {"type": "object", "properties": {"x": {"type": "null"}}}
    with pytest.raises(UnsupportedSchemaError):
        compile_schema(schema)


def test_reject_enum_with_non_scalar_member():
    schema = {"type": "object", "properties": {"x": {"enum": ["ok", {"nested": 1}]}}}
    with pytest.raises(UnsupportedSchemaError) as exc:
        compile_schema(schema)
    assert "$.x" in str(exc.value)


def test_reject_field_without_type_or_enum():
    schema = {"type": "object", "properties": {"x": {"description": "no type"}}}
    with pytest.raises(SchemaCompileError) as exc:
        compile_schema(schema)
    assert "$.x" in str(exc.value)


def test_reject_non_object_root():
    with pytest.raises(SchemaCompileError):
        compile_schema({"type": "string", "maxLength": 4})


def test_reject_root_without_properties():
    with pytest.raises(SchemaCompileError):
        compile_schema({"type": "object"})


def test_reject_required_naming_unknown_field():
    schema = {
        "type": "object",
        "properties": {"a": {"type": "boolean"}},
        "required": ["a", "ghost"],
    }
    with pytest.raises(SchemaCompileError) as exc:
        compile_schema(schema)
    assert "ghost" in str(exc.value)


def test_unsupported_error_is_a_schema_compile_error():
    # T0.2 can catch the base class for either failure mode.
    assert issubclass(UnsupportedSchemaError, SchemaCompileError)


# --------------------------------------------------------------------------- #
# Import hygiene: pure-Python, no NPU/Triton/torch on the import path
# --------------------------------------------------------------------------- #
def test_module_has_no_npu_or_triton_imports():
    """Grep-gate: no NPU/Triton/torch on the import path.

    Parsed via ``ast`` so it flags real ``import``/``from`` statements only, not
    the module names appearing in docstrings/comments.
    """
    import ast

    forbidden_roots = {"triton", "torch", "torch_npu", "vllm"}
    tree = ast.parse(_MODULE_PATH.read_text(encoding="utf-8"))
    imported_roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported_roots.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imported_roots.add(node.module.split(".")[0])
    offenders = forbidden_roots & imported_roots
    assert not offenders, f"schema_ir.py must not import {sorted(offenders)}"
    # Belt-and-suspenders textual gate on actual import statements.
    for line in _MODULE_PATH.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith(("import ", "from ")):
            assert "triton" not in stripped and "torch_npu" not in stripped, (
                f"forbidden import: {stripped!r}"
            )
