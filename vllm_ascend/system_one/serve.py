# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Serving entry: schema in -> typed value out (System-One task T1.2).

This module is the **serving seam** for the System-One runtime (PRD §6.4). It
accepts ``(context, schema)``, compiles the schema into the T0.1 constraint IR,
drives a pluggable :class:`DecodeBackend` to produce a value, validates that
value through the T0.2 oracle, and returns a stable :class:`Decision` carrying
``value`` / ``confidences`` / ``escalate``.

The whole point of this file is the **backend seam**. Today the value is produced
by the interim constrained-AR path (T1.1) via :class:`ConstrainedARBackend`.
Tomorrow the Phase-2 single-forward structured head (T2.x) implements the *same*
:class:`DecodeBackend` protocol and swaps in **without a call-site change** — the
serving surface, the gate, the defensive validation, and the ``Decision`` shape
all stay identical. A :class:`MockBackend` stands in for tests.

Decision shape (stable across the whole project)
------------------------------------------------
* ``value`` — the typed value, guaranteed to pass the T0.2 oracle (asserted
  defensively; a backend that violates it raises :class:`ServingError`).
* ``confidences`` — a per-leaf-path mapping. For now this is a **placeholder /
  pass-through** (uncalibrated); the T2.4 calibrator is wired in at T2.5
  assembly. The field exists now so the shape never changes under consumers.
* ``escalate`` — defaults ``False``; the T3.2 selective-prediction router fills
  it when the request should fall through to the W2 MoE (System-Two).

Gate discipline
---------------
A malformed / unsupported schema is rejected **at the gate** (``decide`` surfaces
T0.1's :class:`SchemaCompileError` as a clean :class:`ServingError`) *before* the
backend is ever driven — never as an uncaught error deep inside the decoder.

Design constraints: **pure Python, stdlib only.** No ``torch``, ``torch_npu`` or
``triton`` on the import path — this module imports and unit-tests host-side,
CPU-only. To stay host-side it loads its sibling System-One modules
(``schema_ir`` / ``validate`` / ``constrained_decode``) **by file path** rather
than ``import vllm_ascend...`` (which would drag in the package ``__init__`` and,
through it, ``torch``/vLLM). The loaded modules are the exact same source files
that ship in the package.
"""

from __future__ import annotations

import importlib.util
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

__all__ = [
    "ServingError",
    "Decision",
    "DecodeBackend",
    "ConstrainedARBackend",
    "MockBackend",
    "SystemOneServer",
    "serve",
    "SimpleVocab",
    "char_vocab",
    "DEFAULT_MAX_STEPS",
    "PLACEHOLDER_CONFIDENCE",
]

# Default step budget handed to the constrained-AR driver (matches T1.1).
DEFAULT_MAX_STEPS = 10_000

# Placeholder per-field confidence until the T2.4 calibrator is wired in (T2.5).
# ``None`` explicitly marks "not yet calibrated" so a consumer cannot mistake a
# placeholder for a real probability.
PLACEHOLDER_CONFIDENCE: float | None = None


# --------------------------------------------------------------------------- #
# Host-side sibling loading (keeps this module free of ``import vllm_ascend``)
# --------------------------------------------------------------------------- #
_SIBLING_DIR = Path(__file__).resolve().parent


def _load_sibling(stem: str) -> Any:
    """Load a System-One sibling module by file path, host-side and cached.

    Prefers a copy already imported as ``vllm_ascend.system_one.<stem>`` (the
    normal in-package case), otherwise loads the sibling source file directly so
    the module never triggers the heavyweight package ``__init__`` (torch/vLLM).
    """
    canonical = f"vllm_ascend.system_one.{stem}"
    existing = sys.modules.get(canonical)
    if existing is not None:
        return existing
    cache_name = f"_system_one_serve__{stem}"
    cached = sys.modules.get(cache_name)
    if cached is not None:
        return cached
    path = _SIBLING_DIR / f"{stem}.py"
    spec = importlib.util.spec_from_file_location(cache_name, path)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise ImportError(f"cannot load System-One sibling module {stem!r} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[cache_name] = module
    spec.loader.exec_module(module)
    return module


_schema_ir = _load_sibling("schema_ir")
_validate = _load_sibling("validate")
_constrained_decode = _load_sibling("constrained_decode")

# The pieces the serving path drives. Re-exported (below) where consumers need
# them so a caller need not reach past ``serve`` for the vocab seam.
compile_schema = _schema_ir.compile_schema
SchemaCompileError = _schema_ir.SchemaCompileError
validate_value = _validate.validate_value
constrained_decode = _constrained_decode.constrained_decode
SimpleVocab = _constrained_decode.SimpleVocab
char_vocab = _constrained_decode.char_vocab


class ServingError(RuntimeError):
    """A serving-level failure surfaced cleanly at the ``decide`` boundary.

    Raised at the gate for a malformed / unsupported schema (wrapping T0.1's
    :class:`SchemaCompileError`), and defensively when a backend produces a value
    that fails the T0.2 oracle. Carries the offending ``path`` when one is known.
    """

    def __init__(self, message: str, *, path: str | None = None) -> None:
        self.path = path
        super().__init__(message)


# --------------------------------------------------------------------------- #
# Decision (the stable output shape every downstream tier consumes)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Decision:
    """The serving result: a typed value plus its confidence + escalate shape.

    * ``value`` — the typed value (T0.2-valid by contract).
    * ``confidences`` — per-leaf-path mapping; placeholder until T2.5 calibration.
    * ``escalate`` — ``False`` until the T3.2 router routes to the W2 MoE.
    """

    value: Any
    confidences: Mapping[str, float | None] = field(default_factory=dict)
    escalate: bool = False


# --------------------------------------------------------------------------- #
# Backend seam
# --------------------------------------------------------------------------- #
@runtime_checkable
class DecodeBackend(Protocol):
    """The abstraction over *how a value is produced* from ``(ir, context)``.

    A backend receives the compiled constraint IR (T0.1) and the request context
    and returns a typed value. This is the single seam the Phase-2 single-forward
    structured head swaps into: the constrained-AR backend drives an
    autoregressive masked decode today; the head backend will emit all fields in
    one forward — both behind the identical ``decode`` signature, so the serving
    call site never changes.
    """

    def decode(self, ir: Any, context: Any) -> Any:
        """Produce a typed value for ``context`` under the constraints in ``ir``."""
        ...


class ConstrainedARBackend:
    """Backend wrapping the T1.1 constrained-AR decoder over an injected logits fn.

    ``logits_fn(context, step) -> vector`` supplies the per-step logits (a real
    tokenizer/model in production, a mock in tests); ``tokenizer`` is the
    :class:`SimpleVocab` mapping token ids to surfaces. The decode is masked so
    the emitted value is schema-valid by construction (it passes T0.2).
    """

    def __init__(
        self,
        logits_fn: Callable[[Any, int], Sequence[float]],
        tokenizer: Any,
        *,
        max_steps: int = DEFAULT_MAX_STEPS,
    ) -> None:
        self._logits_fn = logits_fn
        self._tokenizer = tokenizer
        self._max_steps = max_steps

    def decode(self, ir: Any, context: Any) -> Any:
        # Adapt ``(context, step)`` to the driver's ``step -> vector`` stream.
        def _stream(step: int) -> Sequence[float]:
            return self._logits_fn(context, step)

        return constrained_decode(ir, _stream, self._tokenizer, max_steps=self._max_steps)


class MockBackend:
    """A test/stub backend that returns a fixed value or one computed per request.

    Either pass a constant ``value``, or a ``value_fn(ir, context) -> value``.
    Used to exercise the serving seam (and its defensive validation) without a
    real model — and to stand in for the Phase-2 head backend in shape tests.
    """

    def __init__(
        self,
        *,
        value: Any = None,
        value_fn: Callable[[Any, Any], Any] | None = None,
    ) -> None:
        if value_fn is None and value is None:
            raise ValueError("MockBackend needs either 'value' or 'value_fn'")
        self._value = value
        self._value_fn = value_fn

    def decode(self, ir: Any, context: Any) -> Any:
        if self._value_fn is not None:
            return self._value_fn(ir, context)
        return self._value


# --------------------------------------------------------------------------- #
# Serving surface
# --------------------------------------------------------------------------- #
def _placeholder_confidences(ir: Any) -> dict[str, float | None]:
    """One placeholder confidence per declared leaf-field path (stable shape).

    T2.5 replaces these placeholders with T2.4-calibrated per-field probabilities;
    keeping the keys here means that swap does not change the ``Decision`` shape.
    """
    return {path: PLACEHOLDER_CONFIDENCE for path in ir.field_paths()}


class SystemOneServer:
    """The serving entry: ``(context, schema) -> Decision`` over a pluggable backend.

    ``decide`` compiles the schema at the gate (surfacing malformed schemas as
    :class:`ServingError`), drives the backend, defensively validates the result
    through T0.2, and returns a stable :class:`Decision`. The backend is the only
    thing that changes between the interim constrained-AR path and the Phase-2
    structured head.
    """

    def __init__(self, backend: DecodeBackend, *, strict: bool = True) -> None:
        self._backend = backend
        self._strict = strict

    def decide(self, context: Any, schema: Any) -> Decision:
        ir = self._compile_gate(schema)
        value = self._backend.decode(ir, context)
        self._assert_valid(ir, value)
        return Decision(
            value=value,
            confidences=_placeholder_confidences(ir),
            escalate=False,
        )

    # -- internals ---------------------------------------------------------- #
    def _compile_gate(self, schema: Any) -> Any:
        """Compile the schema, surfacing any T0.1 error as a clean ServingError."""
        try:
            return compile_schema(schema)
        except SchemaCompileError as exc:
            path = getattr(exc, "path", None)
            raise ServingError(f"invalid output schema: {exc}", path=path) from exc

    def _assert_valid(self, ir: Any, value: Any) -> None:
        """Defensive T0.2 gate: a backend must never emit a schema-invalid value."""
        result = validate_value(ir, value, strict=self._strict)
        if not result.ok:
            raise ServingError(
                f"backend produced a schema-invalid value: {result.reason}",
                path=result.path,
            )


def serve(context: Any, schema: Any, backend: DecodeBackend, *, strict: bool = True) -> Decision:
    """Convenience one-shot: build a :class:`SystemOneServer` and ``decide`` once."""
    return SystemOneServer(backend, strict=strict).decide(context, schema)
