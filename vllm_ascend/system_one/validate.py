# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Structured-validity checker (System-One task T0.2).

Given a compiled :class:`~vllm_ascend.system_one.schema_ir.SchemaIR` (from T0.1)
and a candidate value, decide whether the value is **schema-valid** and, if not,
point at the *first offending field path* with a human-readable reason (the full
violation list is also carried). This module is the **oracle** for the PRD's
"100% schema-valid" gate that the Phase-1/Phase-2 runtimes (T1.1 / T1.3 / T2.1)
are measured against.

Design contract: the checker consumes **only the IR** — each ``FieldIR``'s
``kind`` and its immutable *domain descriptor* (enum members / string bound /
numeric ``[min, max]`` / nested ``SchemaIR``). It never re-parses the raw JSON
schema, so it stays in lockstep with T0.1's compiler and cannot drift from it.
It does **not** import ``schema_ir`` (dispatch is on the ``str``-valued
``FieldKind``), so it imports and unit-tests host-side with no NPU, no Triton,
no vLLM package ``__init__`` side effects.

Decisions T1.1 must know
------------------------
* **Strict by default.** Any key present in the value that the schema does not
  declare is a violation (``strict=True``). This is the safer oracle: an "extra"
  key means the decoder emitted something outside the grammar. Pass
  ``strict=False`` for a lax mode that tolerates unknown keys but still enforces
  every declared field's domain.
* **``bool`` is not an integer / number / enum-int.** Although ``bool`` subclasses
  ``int`` in Python, ``True`` / ``False`` are rejected for ``integer`` / ``number``
  fields and for ``enum`` int members. They are only valid for ``boolean`` fields.
* **Patterns ARE enforced.** T0.1 records ``pattern`` but does not enforce it; the
  oracle *does*, using ``re.search`` (JSON-Schema "pattern" semantics: the regex
  must match somewhere in the string unless it is anchored). An un-compilable
  pattern is treated as an unsatisfiable constraint (the value is rejected).
* **First offending path.** Violations are reported in schema declaration order,
  depth-first into nested objects; missing/invalid declared fields are reported
  before unknown extra keys.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from typing import Any

__all__ = [
    "Violation",
    "ValidationResult",
    "validate_value",
    "is_valid",
]

# ``FieldKind`` string values (T0.1's ``FieldKind`` is a ``str``-mixin Enum, so we
# dispatch on the plain string and never need to import ``schema_ir``).
_KIND_ENUM = "enum"
_KIND_STRING = "string"
_KIND_INTEGER = "integer"
_KIND_NUMBER = "number"
_KIND_BOOLEAN = "boolean"
_KIND_OBJECT = "object"


@dataclass(frozen=True)
class Violation:
    """A single reason a value fails the schema, with the offending field path."""

    path: str
    reason: str


@dataclass(frozen=True)
class ValidationResult:
    """The verdict for a candidate value.

    ``ok`` is the boolean gate. On failure, ``violations`` lists every problem
    found (declaration order, depth-first); ``path`` / ``reason`` are convenience
    accessors for the *first* offending violation.
    """

    ok: bool
    violations: tuple[Violation, ...] = dataclass_field(default_factory=tuple)

    @property
    def path(self) -> str | None:
        """The first offending field path, or ``None`` when valid."""
        return self.violations[0].path if self.violations else None

    @property
    def reason(self) -> str | None:
        """The first offending reason, or ``None`` when valid."""
        return self.violations[0].reason if self.violations else None

    def __bool__(self) -> bool:
        return self.ok


def _type_name(value: Any) -> str:
    return type(value).__name__


def _pattern_matches(pattern: str, value: str) -> bool:
    try:
        return re.search(pattern, value) is not None
    except re.error:
        # An un-compilable pattern is an unsatisfiable constraint: reject.
        return False


def _check_field(field: Any, value: Any, strict: bool, out: list[Violation]) -> None:
    """Validate one leaf/record field's value, appending any violations."""
    kind = field.kind
    domain = field.domain
    path = field.path

    if kind == _KIND_ENUM:
        # bool is an int subclass; True/False must not slip in as an int member.
        if isinstance(value, bool) or value not in domain.members:
            out.append(
                Violation(path, f"value {value!r} is not an enum member of "
                                f"{list(domain.members)!r}")
            )

    elif kind == _KIND_STRING:
        if not isinstance(value, str):
            out.append(Violation(path, f"expected string, got {_type_name(value)}"))
        elif len(value) > domain.max_length:
            out.append(
                Violation(path, f"string length {len(value)} exceeds maxLength "
                                f"{domain.max_length}")
            )
        elif domain.pattern is not None and not _pattern_matches(domain.pattern, value):
            out.append(
                Violation(path, f"string {value!r} does not match pattern "
                                f"{domain.pattern!r}")
            )

    elif kind == _KIND_INTEGER:
        if isinstance(value, bool) or not isinstance(value, int):
            out.append(Violation(path, f"expected integer, got {_type_name(value)}"))
        elif not (domain.minimum <= value <= domain.maximum):
            out.append(
                Violation(path, f"integer {value} out of range "
                                f"[{domain.minimum}, {domain.maximum}]")
            )

    elif kind == _KIND_NUMBER:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            out.append(Violation(path, f"expected number, got {_type_name(value)}"))
        elif not (domain.minimum <= value <= domain.maximum):
            out.append(
                Violation(path, f"number {value} out of range "
                                f"[{domain.minimum}, {domain.maximum}]")
            )

    elif kind == _KIND_BOOLEAN:
        if not isinstance(value, bool):
            out.append(Violation(path, f"expected boolean, got {_type_name(value)}"))

    elif kind == _KIND_OBJECT:
        if not isinstance(value, dict):
            out.append(Violation(path, f"expected object, got {_type_name(value)}"))
        else:
            _check_object(domain.schema, value, strict, out)

    else:  # pragma: no cover - defensive: the IR only produces the kinds above.
        out.append(Violation(path, f"unknown field kind {kind!r}"))


def _check_object(ir: Any, value: dict[str, Any], strict: bool,
                  out: list[Violation]) -> None:
    """Validate a record (a ``SchemaIR``) against a dict value."""
    declared: set[str] = set()
    for field in ir.fields:
        declared.add(field.name)
        if field.name in value:
            _check_field(field, value[field.name], strict, out)
        elif field.required:
            out.append(Violation(field.path, "missing required field"))

    if strict:
        for key in value:
            if key not in declared:
                out.append(Violation(f"{ir.path}.{key}", f"unknown key {key!r}"))


def validate_value(ir: Any, value: Any, *, strict: bool = True) -> ValidationResult:
    """Check ``value`` against a compiled ``SchemaIR`` ``ir``.

    Returns a :class:`ValidationResult`: ``ok=True`` with no violations when the
    value satisfies every declared field's domain (and, when ``strict``, carries
    no undeclared keys); otherwise ``ok=False`` with the ordered violation list.

    ``strict`` (default ``True``) rejects unknown/extra keys; ``strict=False``
    tolerates them while still enforcing every declared field.
    """
    out: list[Violation] = []
    if not isinstance(value, dict):
        out.append(Violation(ir.path, f"expected object, got {_type_name(value)}"))
    else:
        _check_object(ir, value, strict, out)
    return ValidationResult(ok=not out, violations=tuple(out))


def is_valid(ir: Any, value: Any, *, strict: bool = True) -> bool:
    """Convenience boolean form of :func:`validate_value`."""
    return validate_value(ir, value, strict=strict).ok
