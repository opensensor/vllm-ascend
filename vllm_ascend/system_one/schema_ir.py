# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Schema -> decoding-constraint IR compiler (System-One task T0.1).

This module compiles a **JSON-Schema subset** into a normalized, immutable
**constraint IR** that downstream System-One tasks consume:

* the Phase-1 constrained-AR path (T1.1) turns the IR into a token grammar/mask,
* the Phase-2 structured heads (T2.1) size a non-autoregressive head set from it,
* the validity checker (T0.2) uses each field's validator descriptor as its oracle.

The IR is a small, introspectable tree of frozen dataclasses. A top-level
:class:`SchemaIR` holds an ordered tuple of :class:`FieldIR`; each field carries a
name, a JSON-pointer-ish ``path`` (root is ``"$"``), a :class:`FieldKind`, an
immutable *domain descriptor* (enum members / string length bound / numeric
``[min, max]`` / nested :class:`SchemaIR`), and a ``required`` flag.

Supported subset (anything else is rejected LOUDLY with the offending path):

* ``enum`` of finite scalars (``str`` or ``int``; ``bool`` members are rejected),
* bounded ``string`` (requires ``maxLength``; ``pattern`` recorded, not enforced),
* ``integer`` / ``number`` with **both** ``minimum`` and ``maximum``,
* ``boolean``,
* nested ``object`` records with named ``required`` / optional fields.

Design constraints: **pure Python, stdlib only.** No ``torch``, no ``torch_npu``,
no ``triton`` — the module imports and tests host-side, CPU-only.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from enum import Enum
from typing import Any

__all__ = [
    "FieldKind",
    "EnumDomain",
    "StringDomain",
    "NumericDomain",
    "BooleanDomain",
    "ObjectDomain",
    "FieldIR",
    "SchemaIR",
    "SchemaCompileError",
    "UnsupportedSchemaError",
    "compile_schema",
]

# Root path token for the top-level record. Nested fields extend it with ``.name``.
ROOT_PATH = "$"

# JSON-Schema combinator / reference keys we explicitly do not support.
_UNSUPPORTED_KEYS = ("oneOf", "anyOf", "allOf", "not", "$ref")


class SchemaCompileError(ValueError):
    """A schema cannot be compiled into the constraint IR.

    Carries the JSON-pointer-ish ``path`` of the offending location so callers
    (and downstream tasks) can point at exactly what failed.
    """

    def __init__(self, message: str, path: str) -> None:
        self.path = path
        super().__init__(f"{message} (at {path})")


class UnsupportedSchemaError(SchemaCompileError):
    """The schema uses a construct outside the supported subset.

    A subclass of :class:`SchemaCompileError` so callers may catch either the
    specific ``unsupported construct`` case or the general ``cannot compile`` one.
    """


class FieldKind(str, Enum):
    """The IR field kinds. ``str`` mixin makes the value JSON/log friendly."""

    ENUM = "enum"
    STRING = "string"
    INTEGER = "integer"
    NUMBER = "number"
    BOOLEAN = "boolean"
    OBJECT = "object"


# --------------------------------------------------------------------------- #
# Domain descriptors (one per kind). All immutable + hashable.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class EnumDomain:
    """A finite set of scalar members (strings or ints), order preserved."""

    members: tuple[str | int, ...]

    def describe(self) -> dict[str, Any]:
        return {"members": list(self.members)}


@dataclass(frozen=True)
class StringDomain:
    """A length-bounded string. ``pattern`` is recorded, not enforced here."""

    max_length: int
    pattern: str | None = None

    def describe(self) -> dict[str, Any]:
        return {"max_length": self.max_length, "pattern": self.pattern}


@dataclass(frozen=True)
class NumericDomain:
    """A closed numeric range ``[minimum, maximum]``.

    ``integral`` is ``True`` for an ``integer`` field, ``False`` for a ``number``.
    """

    minimum: int | float
    maximum: int | float
    integral: bool

    def describe(self) -> dict[str, Any]:
        return {
            "minimum": self.minimum,
            "maximum": self.maximum,
            "integral": self.integral,
        }


@dataclass(frozen=True)
class BooleanDomain:
    """The two-valued boolean domain."""

    def describe(self) -> dict[str, Any]:
        return {"values": [False, True]}


@dataclass(frozen=True)
class ObjectDomain:
    """A nested record, wrapping a nested :class:`SchemaIR`."""

    schema: SchemaIR

    def describe(self) -> dict[str, Any]:
        return {"fields": [f.name for f in self.schema.fields]}


Domain = EnumDomain | StringDomain | NumericDomain | BooleanDomain | ObjectDomain


# --------------------------------------------------------------------------- #
# Field + top-level IR
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class FieldIR:
    """One field of a record: name, path, kind, domain descriptor, required flag."""

    name: str
    path: str
    kind: FieldKind
    domain: Domain
    required: bool

    def is_leaf(self) -> bool:
        """True unless this field is a nested record (which has leaf children)."""
        return self.kind is not FieldKind.OBJECT

    def validator_descriptor(self) -> dict[str, Any]:
        """Round-trip this field back to a flat validator descriptor.

        This is the contract T0.2 (validity checker) relies on: an enum field
        enumerates its members, a numeric field exposes ``[minimum, maximum]``,
        a string field its ``max_length``, etc. Every descriptor carries at least
        ``path``, ``kind`` and ``required``.
        """
        descriptor: dict[str, Any] = {
            "path": self.path,
            "name": self.name,
            "kind": self.kind.value,
            "required": self.required,
        }
        descriptor.update(self.domain.describe())
        return descriptor


@dataclass(frozen=True)
class SchemaIR:
    """A record: an ordered tuple of :class:`FieldIR`, plus its own ``path``."""

    fields: tuple[FieldIR, ...]
    path: str = ROOT_PATH

    def __iter__(self) -> Iterator[FieldIR]:
        return iter(self.fields)

    def iter_fields(self) -> Iterator[FieldIR]:
        """Iterate the top-level fields in declaration order (round-trip)."""
        return iter(self.fields)

    def leaf_fields(self) -> Iterator[FieldIR]:
        """Iterate leaf fields depth-first, descending into nested records.

        Nested ``object`` fields are *not* yielded themselves; their children are.
        This is the flat view Phase-2 head sizing consumes.
        """
        for field in self.fields:
            if field.kind is FieldKind.OBJECT:
                assert isinstance(field.domain, ObjectDomain)
                yield from field.domain.schema.leaf_fields()
            else:
                yield field

    def field_paths(self) -> list[str]:
        """All leaf-field paths, in depth-first declaration order."""
        return [f.path for f in self.leaf_fields()]

    def validator_descriptors(self) -> list[dict[str, Any]]:
        """A validator descriptor per leaf field (the T0.2 oracle input)."""
        return [f.validator_descriptor() for f in self.leaf_fields()]


# --------------------------------------------------------------------------- #
# Compiler
# --------------------------------------------------------------------------- #
def _child_path(parent: str, name: str) -> str:
    return f"{parent}.{name}"


def _reject_combinators(node: dict[str, Any], path: str) -> None:
    for key in _UNSUPPORTED_KEYS:
        if key in node:
            raise UnsupportedSchemaError(
                f"unsupported schema construct {key!r}", path
            )


def _is_scalar_enum_member(value: Any) -> bool:
    # bool is a subclass of int; enum members are strings or ints, not booleans.
    if isinstance(value, bool):
        return False
    return isinstance(value, (str, int))


def _compile_enum(node: dict[str, Any], path: str) -> EnumDomain:
    members = node["enum"]
    if not isinstance(members, (list, tuple)) or len(members) == 0:
        raise UnsupportedSchemaError(
            "enum must be a non-empty list of scalar members", path
        )
    for member in members:
        if not _is_scalar_enum_member(member):
            raise UnsupportedSchemaError(
                f"enum members must be strings or ints, got {type(member).__name__}",
                path,
            )
    return EnumDomain(tuple(members))


def _compile_string(node: dict[str, Any], path: str) -> StringDomain:
    if "maxLength" not in node:
        raise UnsupportedSchemaError(
            "string fields must be bounded with 'maxLength'", path
        )
    max_length = node["maxLength"]
    if not isinstance(max_length, int) or isinstance(max_length, bool) or max_length < 0:
        raise UnsupportedSchemaError(
            "'maxLength' must be a non-negative integer", path
        )
    pattern = node.get("pattern")
    if pattern is not None and not isinstance(pattern, str):
        raise UnsupportedSchemaError("'pattern' must be a string when present", path)
    return StringDomain(max_length=max_length, pattern=pattern)


def _compile_numeric(node: dict[str, Any], path: str, integral: bool) -> NumericDomain:
    if "minimum" not in node or "maximum" not in node:
        raise UnsupportedSchemaError(
            "numeric fields must be bounded with both 'minimum' and 'maximum'",
            path,
        )
    minimum = node["minimum"]
    maximum = node["maximum"]
    for bound in (minimum, maximum):
        if isinstance(bound, bool) or not isinstance(bound, (int, float)):
            raise UnsupportedSchemaError(
                "'minimum'/'maximum' must be numbers", path
            )
    if minimum > maximum:
        raise SchemaCompileError(
            f"'minimum' ({minimum}) exceeds 'maximum' ({maximum})", path
        )
    return NumericDomain(minimum=minimum, maximum=maximum, integral=integral)


def _compile_field(name: str, path: str, node: Any, required: bool) -> FieldIR:
    if not isinstance(node, dict):
        raise SchemaCompileError("field schema must be an object", path)

    _reject_combinators(node, path)

    if "enum" in node:
        return FieldIR(name, path, FieldKind.ENUM, _compile_enum(node, path), required)

    if "type" not in node:
        raise SchemaCompileError(
            "field schema must declare a 'type' or an 'enum'", path
        )

    type_value = node["type"]
    if not isinstance(type_value, str):
        # union types (list-valued 'type') are outside the supported subset.
        raise UnsupportedSchemaError(
            f"'type' must be a single string, got {type(type_value).__name__}",
            path,
        )

    if type_value == "string":
        return FieldIR(name, path, FieldKind.STRING, _compile_string(node, path), required)
    if type_value == "integer":
        domain = _compile_numeric(node, path, integral=True)
        return FieldIR(name, path, FieldKind.INTEGER, domain, required)
    if type_value == "number":
        domain = _compile_numeric(node, path, integral=False)
        return FieldIR(name, path, FieldKind.NUMBER, domain, required)
    if type_value == "boolean":
        return FieldIR(name, path, FieldKind.BOOLEAN, BooleanDomain(), required)
    if type_value == "object":
        nested = _compile_object(node, path)
        return FieldIR(name, path, FieldKind.OBJECT, ObjectDomain(nested), required)

    raise UnsupportedSchemaError(f"unsupported field type {type_value!r}", path)


def _compile_object(node: dict[str, Any], path: str) -> SchemaIR:
    _reject_combinators(node, path)

    declared_type = node.get("type")
    if declared_type is not None and declared_type != "object":
        raise SchemaCompileError(
            f"expected an 'object' schema, got type {declared_type!r}", path
        )

    if "properties" not in node:
        raise SchemaCompileError(
            "object schema must declare 'properties'", path
        )
    properties = node["properties"]
    if not isinstance(properties, dict):
        raise SchemaCompileError("'properties' must be an object", path)

    required_names = node.get("required", [])
    if not isinstance(required_names, (list, tuple)):
        raise SchemaCompileError("'required' must be a list of field names", path)
    required_set = set(required_names)
    unknown = required_set - set(properties.keys())
    if unknown:
        raise SchemaCompileError(
            f"'required' names unknown field(s): {sorted(unknown)}", path
        )

    fields: list[FieldIR] = []
    for prop_name, prop_schema in properties.items():
        child_path = _child_path(path, prop_name)
        fields.append(
            _compile_field(
                prop_name,
                child_path,
                prop_schema,
                required=prop_name in required_set,
            )
        )
    return SchemaIR(fields=tuple(fields), path=path)


def compile_schema(schema: dict[str, Any]) -> SchemaIR:
    """Compile a JSON-Schema-subset ``dict`` into a :class:`SchemaIR`.

    The root must be an ``object`` record with a ``properties`` map. Every field
    is compiled into a :class:`FieldIR`; nested objects recurse into nested
    :class:`SchemaIR`. Unsupported constructs raise :class:`UnsupportedSchemaError`
    (a :class:`SchemaCompileError`) carrying the offending ``path``.
    """
    if not isinstance(schema, dict):
        raise SchemaCompileError("top-level schema must be an object", ROOT_PATH)
    return _compile_object(schema, ROOT_PATH)
