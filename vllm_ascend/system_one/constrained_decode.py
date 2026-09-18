# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Grammar-guided constrained decoder — interim constrained-AR path (task T1.1).

This module compiles the T0.1 constraint IR
(:class:`~vllm_ascend.system_one.schema_ir.SchemaIR`) into a **token-level
decoding constraint**: an incremental character NFA plus a per-step *allowed
token mask*. Applied over any autoregressive decoder's logits — masking
disallowed tokens to ``-inf`` (a hard mask, never a soft penalty) and taking the
argmax over what remains — the emitted string is a canonical JSON document that
is **guaranteed to parse into a schema-valid value** (it passes the T0.2 oracle
``is_valid`` by construction).

This is the *interim* path the PRD (§6.1) calls "constrained AR": a self-contained
masking engine with no external grammar library. The real serving path later
swaps in XGrammar / llguidance / outlines (T1.2 consumes the token/vocab
abstraction defined here), or the Phase-2 structured heads make validity
structural.

Output surface (canonical JSON)
-------------------------------
The root object is serialized with **no whitespace**, its fields in declaration
order::

    {"field_a":<value>,"field_b":<value>}

Present optional fields keep declaration order; absent optionals (and their
commas) simply do not appear. Per-field value surfaces:

* **enum**    — exactly one member literal (``json.dumps`` of the member).
* **boolean** — ``true`` or ``false``.
* **integer** — the finite set of integer literals in ``[minimum, maximum]``.
* **number**  — a finite candidate set: the integers in range plus the exact
  ``minimum`` / ``maximum`` literals (see "Domain enforcement" below).
* **string**  — ``"`` + up to ``max_length`` characters from a safe alphabet +
  ``"`` (no escaping needed). A ``pattern`` restricts the field to a bounded set
  of enumerated *matching* witness literals.
* **object**  — a nested ``{...}`` built the same way.

Domain enforcement — token-wise vs. value-close
------------------------------------------------
Every enforcement here is **value-exact by construction**: each leaf domain is
compiled to a finite set of *complete valid literals* (an ``Alt`` of ``Lit``\\s),
so masking can never walk off a valid prefix. Concretely:

* **enum / boolean** — literal alternation; only members reachable.
* **integer** — *enumerated* over ``[ceil(min), floor(max)]`` (bounded by
  :data:`MAX_ENUM_LITERALS`); range holds exactly, not digit-wise.
* **number** — enumerated candidate set (integers in range + exact bounds); the
  interim path does not synthesize arbitrary reals — full real-line coverage is
  deferred to the structured-head path. Range holds exactly.
* **string length** — enforced **token-wise**: the NFA allows at most
  ``max_length`` content characters, then only the closing quote.
* **string pattern** — enforced by *enumerating* bounded matching witnesses; the
  field becomes an alternation over those literals (each already satisfies the
  T0.2 pattern + length checks).

Design constraints: **pure Python, stdlib only.** No ``torch``, ``torch_npu`` or
``triton`` on the import path — the module imports and unit-tests host-side,
CPU-only. It does **not** import ``schema_ir`` / ``validate``: it duck-types the
IR (dispatch on the ``str``-valued ``FieldKind``) so it stays in lockstep with
T0.1 without pulling in the package ``__init__``.
"""

from __future__ import annotations

import json
import math
import string
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any

__all__ = [
    "NEG_INF",
    "MAX_ENUM_LITERALS",
    "SAFE_STRING_ALPHABET",
    "ConstraintError",
    "ConstraintCompileError",
    "GrammarNFA",
    "ConstraintEngine",
    "SimpleVocab",
    "char_vocab",
    "compile_constraint",
    "apply_mask",
    "constrained_decode",
]

# Hard-mask sentinel: disallowed logits are set to this, never merely penalized.
NEG_INF = float("-inf")

# Upper bound on how many literals a single numeric domain may enumerate. Keeps
# the interim path bounded; a range larger than this raises loudly (the real
# serving path uses a digit-wise grammar instead of enumeration).
MAX_ENUM_LITERALS = 100_000

# Safe characters allowed inside a bounded string body. Excludes ``"`` and ``\``
# so the canonical JSON literal needs no escaping and always re-parses cleanly.
SAFE_STRING_ALPHABET = "".join(
    ch for ch in (string.ascii_letters + string.digits + " _-.")
    if ch not in ('"', "\\")
)

# When a string has a ``pattern`` we enumerate matching witnesses by bounded
# search; these cap the search so compilation stays fast and finite.
_PATTERN_MAX_WITNESSES = 32
_PATTERN_MAX_LEN = 6
_PATTERN_SEARCH_ALPHABET = string.ascii_lowercase + string.digits + "_-"

# ``FieldKind`` string values (T0.1's ``FieldKind`` is a ``str``-mixin Enum, so we
# dispatch on the plain string and never import ``schema_ir``).
_KIND_ENUM = "enum"
_KIND_STRING = "string"
_KIND_INTEGER = "integer"
_KIND_NUMBER = "number"
_KIND_BOOLEAN = "boolean"
_KIND_OBJECT = "object"


class ConstraintError(RuntimeError):
    """A constrained-decoding runtime error (e.g. no allowed token)."""


class ConstraintCompileError(ValueError):
    """The IR cannot be compiled into a bounded interim decoding constraint."""

    def __init__(self, message: str, path: str) -> None:
        self.path = path
        super().__init__(f"{message} (at {path})")


# --------------------------------------------------------------------------- #
# NFA representation + Thompson-style builder
# --------------------------------------------------------------------------- #
# A state is an ``int``. ``_char_trans[s]`` is a list of ``(charset, target)`` —
# consuming a character ``c`` in ``charset`` moves ``s -> target``. ``_eps[s]``
# is a list of epsilon (no-input) targets. Charsets are *concrete* frozensets so
# the engine can enumerate the allowed next characters.
_Fragment = tuple[int, int]  # (start_state, out_state)


@dataclass
class GrammarNFA:
    """A compiled character NFA for a schema's canonical-JSON language."""

    char_trans: dict[int, list[tuple[frozenset[str], int]]]
    eps: dict[int, list[int]]
    start: int
    accept: int
    alphabet: frozenset[str]

    def epsilon_closure(self, states: Iterable[int]) -> frozenset[int]:
        stack = list(states)
        seen: set[int] = set(stack)
        while stack:
            s = stack.pop()
            for t in self.eps.get(s, ()):
                if t not in seen:
                    seen.add(t)
                    stack.append(t)
        return frozenset(seen)

    def step(self, states: frozenset[int], ch: str) -> frozenset[int]:
        """Advance a state set by a single character; returns the new closure."""
        moved: set[int] = set()
        for s in states:
            for charset, target in self.char_trans.get(s, ()):
                if ch in charset:
                    moved.add(target)
        if not moved:
            return frozenset()
        return self.epsilon_closure(moved)

    def allowed_chars(self, states: frozenset[int]) -> frozenset[str]:
        out: set[str] = set()
        for s in states:
            for charset, _target in self.char_trans.get(s, ()):
                out |= charset
        return frozenset(out)


class _NFABuilder:
    """Accumulates states/transitions and returns ``(start, out)`` fragments."""

    def __init__(self) -> None:
        self.char_trans: dict[int, list[tuple[frozenset[str], int]]] = {}
        self.eps: dict[int, list[int]] = {}
        self._n = 0
        self._alphabet: set[str] = set()

    def _new(self) -> int:
        s = self._n
        self._n += 1
        return s

    def _add_char(self, src: int, charset: frozenset[str], dst: int) -> None:
        self.char_trans.setdefault(src, []).append((charset, dst))
        self._alphabet |= charset

    def _add_eps(self, src: int, dst: int) -> None:
        self.eps.setdefault(src, []).append(dst)

    # -- combinators -------------------------------------------------------- #
    def empty(self) -> _Fragment:
        s = self._new()
        return (s, s)

    def lit(self, text: str) -> _Fragment:
        start = self._new()
        cur = start
        for ch in text:
            nxt = self._new()
            self._add_char(cur, frozenset({ch}), nxt)
            cur = nxt
        return (start, cur)

    def char_class(self, charset: frozenset[str]) -> _Fragment:
        a = self._new()
        b = self._new()
        self._add_char(a, charset, b)
        return (a, b)

    def seq(self, frags: Sequence[_Fragment]) -> _Fragment:
        frags = [f for f in frags]
        if not frags:
            return self.empty()
        for left, right in zip(frags, frags[1:]):
            self._add_eps(left[1], right[0])
        return (frags[0][0], frags[-1][1])

    def alt(self, frags: Sequence[_Fragment]) -> _Fragment:
        start = self._new()
        out = self._new()
        for f in frags:
            self._add_eps(start, f[0])
            self._add_eps(f[1], out)
        return (start, out)

    def bounded_repeat_class(self, charset: frozenset[str], max_count: int) -> _Fragment:
        """0..``max_count`` characters from ``charset``; every count is a valid stop."""
        start = self._new()
        out = self._new()
        self._add_eps(start, out)  # zero characters
        cur = start
        for _ in range(max_count):
            nxt = self._new()
            self._add_char(cur, charset, nxt)
            self._add_eps(nxt, out)
            cur = nxt
        return (start, out)

    def finish(self, frag: _Fragment) -> GrammarNFA:
        return GrammarNFA(
            char_trans=self.char_trans,
            eps=self.eps,
            start=frag[0],
            accept=frag[1],
            alphabet=frozenset(self._alphabet),
        )


# --------------------------------------------------------------------------- #
# IR -> grammar compilation
# --------------------------------------------------------------------------- #
def _literal_alt(builder: _NFABuilder, literals: Sequence[str]) -> _Fragment:
    return builder.alt([builder.lit(lit) for lit in literals])


def _enum_literals(domain: Any) -> list[str]:
    # Members are strings or ints (T0.1 rejects bool members). json.dumps gives
    # the canonical literal ("red" -> '"red"', 42 -> '42').
    return [json.dumps(member) for member in domain.members]


def _integer_literals(domain: Any, path: str) -> list[str]:
    lo = math.ceil(domain.minimum)
    hi = math.floor(domain.maximum)
    if hi < lo:
        raise ConstraintCompileError(
            f"integer range [{domain.minimum}, {domain.maximum}] contains no integer",
            path,
        )
    if hi - lo + 1 > MAX_ENUM_LITERALS:
        raise ConstraintCompileError(
            f"integer range [{lo}, {hi}] too large to enumerate "
            f"(> {MAX_ENUM_LITERALS}); interim path enumerates literals",
            path,
        )
    return [str(k) for k in range(lo, hi + 1)]


def _number_literals(domain: Any, path: str) -> list[str]:
    lo_v = domain.minimum
    hi_v = domain.maximum
    literals: list[str] = []
    seen: set[str] = set()

    def _push(text: str) -> None:
        if text not in seen:
            seen.add(text)
            literals.append(text)

    # Exact bounds are always valid, always included.
    _push(json.dumps(float(lo_v)))
    _push(json.dumps(float(hi_v)))
    # Plus every integer strictly inside the range (bounded).
    lo_i = math.ceil(lo_v)
    hi_i = math.floor(hi_v)
    if hi_i >= lo_i:
        count = hi_i - lo_i + 1
        if count > MAX_ENUM_LITERALS:
            # Sample evenly rather than raising: numbers do not need every point.
            step = math.ceil(count / MAX_ENUM_LITERALS)
        else:
            step = 1
        for k in range(lo_i, hi_i + 1, step):
            _push(str(k))
    # Belt: keep only literals whose parsed value is truly in range.
    safe = [t for t in literals if lo_v <= json.loads(t) <= hi_v]
    if not safe:  # pragma: no cover - bounds are always in range
        raise ConstraintCompileError(
            f"number range [{lo_v}, {hi_v}] produced no valid literal", path
        )
    return safe


def _string_value(builder: _NFABuilder, field: Any) -> _Fragment:
    domain = field.domain
    if domain.pattern is not None:
        witnesses = _pattern_witnesses(domain, field.path)
        return _literal_alt(builder, [json.dumps(w) for w in witnesses])
    charset = frozenset(SAFE_STRING_ALPHABET)
    body = builder.bounded_repeat_class(charset, domain.max_length)
    return builder.seq([builder.lit('"'), body, builder.lit('"')])


def _pattern_witnesses(domain: Any, path: str) -> list[str]:
    """Bounded search for strings that satisfy ``pattern`` and ``max_length``.

    Any witness found is valid by construction (it passes the same length +
    ``re.search`` checks the T0.2 oracle applies), so we only need a non-empty
    set. The search is capped in both length and count.
    """
    import re

    try:
        rx = re.compile(domain.pattern)
    except re.error as exc:
        raise ConstraintCompileError(
            f"string pattern {domain.pattern!r} is not a compilable regex: {exc}",
            path,
        )

    max_len = min(domain.max_length, _PATTERN_MAX_LEN)
    found: list[str] = []
    alphabet = _PATTERN_SEARCH_ALPHABET

    # Breadth-first over lengths 0..max_len; short witnesses first.
    def _extend(prefix: str) -> None:
        if len(found) >= _PATTERN_MAX_WITNESSES:
            return
        if rx.search(prefix) is not None and len(prefix) <= domain.max_length:
            found.append(prefix)
        if len(prefix) >= max_len:
            return
        for ch in alphabet:
            if len(found) >= _PATTERN_MAX_WITNESSES:
                return
            _extend(prefix + ch)

    _extend("")
    if not found:
        raise ConstraintCompileError(
            f"could not synthesize any string matching pattern {domain.pattern!r} "
            f"within length {max_len} and alphabet {alphabet!r}",
            path,
        )
    # Deduplicate while preserving discovery order.
    seen: set[str] = set()
    unique = [w for w in found if not (w in seen or seen.add(w))]
    return unique


def _value_fragment(builder: _NFABuilder, field: Any) -> _Fragment:
    kind = field.kind
    domain = field.domain
    if kind == _KIND_ENUM:
        return _literal_alt(builder, _enum_literals(domain))
    if kind == _KIND_BOOLEAN:
        return _literal_alt(builder, ["true", "false"])
    if kind == _KIND_INTEGER:
        return _literal_alt(builder, _integer_literals(domain, field.path))
    if kind == _KIND_NUMBER:
        return _literal_alt(builder, _number_literals(domain, field.path))
    if kind == _KIND_STRING:
        return _string_value(builder, field)
    if kind == _KIND_OBJECT:
        return _object_fragment(builder, domain.schema)
    raise ConstraintCompileError(f"unsupported field kind {kind!r}", field.path)


def _fields_fragment(
    builder: _NFABuilder, fields: Sequence[Any], index: int, first: bool
) -> _Fragment:
    """Render ``fields[index:]``; ``first`` means no field emitted yet (no comma)."""
    if index >= len(fields):
        return builder.empty()
    field = fields[index]
    comma = builder.empty() if first else builder.lit(",")
    key = builder.lit(json.dumps(field.name) + ":")
    value = _value_fragment(builder, field)
    tail_present = _fields_fragment(builder, fields, index + 1, first=False)
    present = builder.seq([comma, key, value, tail_present])
    if field.required:
        return present
    absent = _fields_fragment(builder, fields, index + 1, first=first)
    return builder.alt([present, absent])


def _object_fragment(builder: _NFABuilder, schema_ir: Any) -> _Fragment:
    open_b = builder.lit("{")
    inner = _fields_fragment(builder, list(schema_ir.fields), 0, first=True)
    close_b = builder.lit("}")
    return builder.seq([open_b, inner, close_b])


def compile_constraint(ir: Any) -> GrammarNFA:
    """Compile a compiled ``SchemaIR`` into a :class:`GrammarNFA`.

    The resulting NFA's language is exactly the set of canonical-JSON documents
    whose parsed value passes the T0.2 oracle for ``ir``.
    """
    builder = _NFABuilder()
    frag = _object_fragment(builder, ir)
    return builder.finish(frag)


# --------------------------------------------------------------------------- #
# Stateful decoding engine
# --------------------------------------------------------------------------- #
class ConstraintEngine:
    """A stateful incremental matcher over a compiled :class:`GrammarNFA`.

    Tracks the set of live NFA states given the characters emitted so far and
    exposes the mask / advance / completeness primitives the driver (and T1.2's
    serving entry) consume.
    """

    def __init__(self, nfa: GrammarNFA) -> None:
        self._nfa = nfa
        self._states = nfa.epsilon_closure([nfa.start])
        self._text: list[str] = []

    # -- introspection ------------------------------------------------------ #
    def is_complete(self) -> bool:
        """True iff the emitted string is a complete valid document (accept)."""
        return self._nfa.accept in self._states

    def text(self) -> str:
        """The characters emitted so far (excludes the EOS token)."""
        return "".join(self._text)

    def allowed_chars(self) -> frozenset[str]:
        """The set of characters that may legally follow the current prefix."""
        return self._nfa.allowed_chars(self._states)

    def _can_consume(self, surface: str) -> bool:
        states = self._states
        for ch in surface:
            states = self._nfa.step(states, ch)
            if not states:
                return False
        return True

    # -- token / character advance ------------------------------------------ #
    def allowed_token_mask(self, vocab: SimpleVocab) -> list[bool]:
        """A boolean mask over ``vocab``: ``True`` where the token may be emitted.

        The EOS token is allowed **iff** the document is complete. A non-EOS
        token is allowed iff feeding its surface keeps at least one NFA state
        live (i.e. the resulting prefix can still complete to a valid value).
        """
        complete = self.is_complete()
        mask = [False] * len(vocab.tokens)
        for i, surface in enumerate(vocab.tokens):
            if i == vocab.eos_id:
                mask[i] = complete
            elif surface == "":
                mask[i] = False  # non-EOS empty surfaces never advance
            else:
                mask[i] = self._can_consume(surface)
        return mask

    def advance(self, vocab: SimpleVocab, token_id: int) -> None:
        """Consume ``token_id``. EOS is a no-op terminal (only valid if complete)."""
        if token_id == vocab.eos_id:
            if not self.is_complete():
                raise ConstraintError("EOS emitted before the document is complete")
            return
        self.advance_str(vocab.tokens[token_id])

    def advance_str(self, surface: str) -> None:
        """Consume a raw character/fragment surface, updating the live states."""
        states = self._states
        for ch in surface:
            states = self._nfa.step(states, ch)
            if not states:
                raise ConstraintError(
                    f"surface {surface!r} is not allowed at prefix {self.text()!r}"
                )
        self._states = states
        self._text.append(surface)


# --------------------------------------------------------------------------- #
# Vocab abstraction
# --------------------------------------------------------------------------- #
@dataclass
class SimpleVocab:
    """A minimal token/vocab abstraction: token id -> surface string.

    ``tokens[i]`` is the string fragment token ``i`` emits; ``tokens[eos_id]``
    is the stop token (surface ignored, conventionally ``""``). This is the seam
    T1.2's serving entry adapts a real tokenizer to — the engine is otherwise
    charset-/token-agnostic.
    """

    tokens: list[str]
    eos_id: int

    def __post_init__(self) -> None:
        if not (0 <= self.eos_id < len(self.tokens)):
            raise ValueError(f"eos_id {self.eos_id} out of range for {len(self.tokens)}")

    def __len__(self) -> int:
        return len(self.tokens)

    def index(self, surface: str) -> int:
        """The id of the token with this exact surface (first match)."""
        return self.tokens.index(surface)


def char_vocab(alphabet: str, *, eos_surface: str = "") -> SimpleVocab:
    """Build a single-character vocab from ``alphabet`` plus a trailing EOS token."""
    tokens = list(alphabet) + [eos_surface]
    return SimpleVocab(tokens=tokens, eos_id=len(tokens) - 1)


# --------------------------------------------------------------------------- #
# Masking + driver
# --------------------------------------------------------------------------- #
def apply_mask(logits: Sequence[float], mask: Sequence[bool]) -> list[float]:
    """Return a new logits list with disallowed positions hard-set to ``-inf``.

    This is a **mask, not a penalty**: disallowed tokens become exactly
    ``-inf`` so they can never win an argmax, regardless of their raw score.
    """
    if len(logits) != len(mask):
        raise ValueError(f"logits/mask length mismatch: {len(logits)} vs {len(mask)}")
    return [logits[i] if mask[i] else NEG_INF for i in range(len(logits))]


def _argmax_allowed(masked: Sequence[float]) -> int:
    best_i = -1
    best_v = NEG_INF
    for i, v in enumerate(masked):
        if v > best_v:
            best_v = v
            best_i = i
    if best_i < 0 or best_v == NEG_INF:
        raise ConstraintError("no allowed token at this decoding step")
    return best_i


_LogitsStream = Any  # callable(step)->vector | vector | sequence-of-vectors


def _logits_source(stream: _LogitsStream, vocab_size: int) -> Callable[[int], Sequence[float]]:
    """Normalize the ``logits_stream`` argument into a ``step -> vector`` callable.

    Accepts: a callable ``f(step) -> vector``; a single constant vector (reused
    every step); a sequence of per-step vectors; or an iterator of vectors.
    """
    if callable(stream):
        return stream  # type: ignore[return-value]

    # A single constant vector: first element is a number.
    if isinstance(stream, Sequence) and stream and isinstance(stream[0], (int, float)):
        vec = list(stream)
        return lambda step: vec

    if isinstance(stream, Sequence):
        return lambda step: stream[step]

    # Fall back to an iterator of vectors.
    iterator = iter(stream)

    def _from_iter(step: int) -> Sequence[float]:
        return next(iterator)

    return _from_iter


def constrained_decode(
    ir: Any,
    logits_stream: _LogitsStream,
    tokenizer: SimpleVocab,
    *,
    max_steps: int = 10_000,
) -> Any:
    """Greedily decode a schema-valid value under the constraint compiled from ``ir``.

    At each step the engine's allowed-token mask is applied to the step's logits
    (disallowed -> ``-inf``); the argmax over what remains is chosen. Choosing
    the EOS token (allowed only when the document is complete) ends decoding. The
    emitted canonical-JSON string is parsed and returned as the typed value.

    The result passes the T0.2 oracle by construction; a final ``json.loads`` is
    the only parse and it cannot fail (the grammar only accepts valid JSON docs).
    """
    nfa = compile_constraint(ir)
    engine = ConstraintEngine(nfa)
    next_logits = _logits_source(logits_stream, len(tokenizer.tokens))

    step = 0
    while True:
        mask = engine.allowed_token_mask(tokenizer)
        logits = next_logits(step)
        if len(logits) != len(tokenizer.tokens):
            raise ValueError(
                f"logits length {len(logits)} != vocab size {len(tokenizer.tokens)}"
            )
        masked = apply_mask(logits, mask)
        choice = _argmax_allowed(masked)
        if choice == tokenizer.eos_id:
            break
        engine.advance(tokenizer, choice)
        step += 1
        if step > max_steps:
            raise ConstraintError(
                f"decoding did not terminate within {max_steps} steps: "
                f"{engine.text()!r}"
            )

    if not engine.is_complete():  # pragma: no cover - EOS mask guarantees this
        raise ConstraintError(f"decoding stopped before completion: {engine.text()!r}")
    return json.loads(engine.text())
