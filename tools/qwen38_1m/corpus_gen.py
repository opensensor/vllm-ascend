#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Long-context corpus & retrieval-probe generator for Qwen4Exp 1M on 310P (plan T0.4).

Builds deterministic (seeded) "needle in a haystack" prompts at the total-token
sizes the 1M-context validation wave exercises: 8K, 128K, 262144, 524288, and
``1048576 - 516`` (516 tokens reserved for the chat template / generation
headroom so the request still fits the 1M window). Each prompt embeds eight
distributed retrieval records near the beginning, quarter, middle, three-quarter
and end of the document, plus one adversarial *no-answer* record whose fact is
deliberately omitted so a faithful model must decline to answer.

Tokenizer injection
-------------------
The real Qwen4Exp checkpoint tokenizer is not available on the authoring host, so
the tokenizer is an *injected* dependency: any object exposing ``encode(text) ->
list[int]`` and an ``eos_id`` attribute (see :class:`TokenizerLike`). Unit tests
inject a deterministic whitespace tokenizer, which makes token counts exact and
reproducible. Exact-count validation against the real checkpoint tokenizer is
deferred to the device wave (plan D1); the builder is tokenizer-agnostic because
it converges on the exact target by re-encoding after each padding adjustment.

Padding scheme
--------------
Fixed content (header, rendered records, footer questions) is measured with the
injected tokenizer. Filler prose is then distributed across the gaps so each
record lands near its target document fraction, and a trailing filler region is
tuned by a re-encode convergence loop until the emitted text encodes to *exactly*
the requested token count.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Protocol, runtime_checkable

SCHEMA_VERSION = 1
DEFAULT_SEED = 1234

# 516 tokens reserved from the 1M window for the chat template + generation.
_MAX_CONTEXT_TOKENS = 1_048_576
_ONE_M_HEADROOM_TOKENS = 516

TARGET_TOKEN_SIZES = (
    8 * 1024,  # 8192
    128 * 1024,  # 131072
    262144,
    524288,
    _MAX_CONTEXT_TOKENS - _ONE_M_HEADROOM_TOKENS,  # 1048060
)

# Sentinel marking a record whose answer is intentionally absent (adversarial).
NO_ANSWER = "__NO_ANSWER__"

# Bound on the padding convergence loop; a whitespace tokenizer converges in <=2.
_MAX_PADDING_ITERATIONS = 128


@runtime_checkable
class TokenizerLike(Protocol):
    """Minimal tokenizer contract the generator depends on (injectable)."""

    eos_id: int

    def encode(self, text: str) -> list[int]: ...


# Positional layout is deterministic and seed-independent: eight answerable
# records spread near the five canonical zones. Record *content* is seeded.
_NEEDLE_LAYOUT: tuple[tuple[str, float], ...] = (
    ("beginning", 0.02),
    ("beginning", 0.06),
    ("quarter", 0.24),
    ("quarter", 0.28),
    ("middle", 0.49),
    ("middle", 0.52),
    ("three_quarter", 0.75),
    ("end", 0.97),
)
_ADVERSARIAL_LAYOUT: tuple[str, float] = ("adversarial", 0.40)

_SUBJECTS = (
    "access code",
    "vault key",
    "dispatch id",
    "ledger seal",
    "transit token",
    "relay cipher",
    "beacon hash",
    "archive stamp",
    "escrow tag",
)
_CITIES = (
    "Harbin",
    "Kunming",
    "Lhasa",
    "Urumqi",
    "Nanning",
    "Guiyang",
    "Yinchuan",
    "Xining",
    "Haikou",
)
_CODE_ALPHABET = "ACDEFHJKLMNPRTUVWXY3479"
_CODE_LENGTH = 8

# Small cyclic prose vocab (each word is a single whitespace token). Reused so a
# whitespace tokenizer keeps a tiny vocab even for million-token documents.
_FILLER_VOCAB = (
    "the",
    "archive",
    "records",
    "many",
    "routine",
    "status",
    "updates",
    "across",
    "distributed",
    "ledger",
    "during",
    "quarterly",
    "review",
    "cycles",
    "without",
    "notable",
    "changes",
    "to",
    "operational",
    "baselines",
    "and",
    "logged",
    "throughput",
    "metrics",
    "for",
    "downstream",
    "audit",
    "consumers",
)
_FILLER_PROBE = "context"

_HEADER = (
    "SYSTEM CONTEXT DOCUMENT. The following log is a long operational archive. "
    "Some lines contain FACT markers you must retain. Read the whole document, "
    "then answer the retrieval questions at the end using only stated facts."
)


@dataclass
class NeedleRecord:
    """A single retrieval record embedded in the corpus.

    ``expected_answer`` is :data:`NO_ANSWER` for the adversarial case whose fact is
    deliberately omitted.
    """

    needle_id: str
    position_label: str
    fraction: float
    subject: str
    city: str
    expected_answer: str
    is_adversarial: bool = False

    @property
    def question(self) -> str:
        return f"Q ({self.needle_id}): state the registered {self.subject} of {self.city}."

    def render(self) -> str:
        """The line embedded in the document body for this record."""
        if self.is_adversarial:
            return (
                f"[FACT {self.needle_id}] The registered {self.subject} of {self.city} "
                "is deliberately omitted from this document."
            )
        return (
            f"[FACT {self.needle_id}] The registered {self.subject} of {self.city} "
            f"is {self.expected_answer}. Retain this value verbatim."
        )


@dataclass
class BuiltPrompt:
    """A fully built prompt with exact token accounting and answer keys."""

    target_tokens: int
    token_count: int
    seed: int
    text: str
    needles: list[NeedleRecord] = field(default_factory=list)

    def expected_answers(self) -> dict[str, str]:
        """Return the ``{needle_id: expected_answer}`` map (adversarial marked)."""
        return expected_answer_map(self.needles)

    def to_metadata(self) -> dict:
        """JSON-serialisable summary (excludes the large ``text`` body)."""
        return {
            "schema_version": SCHEMA_VERSION,
            "target_tokens": self.target_tokens,
            "token_count": self.token_count,
            "seed": self.seed,
            "needles": [asdict(n) for n in self.needles],
            "expected_answers": self.expected_answers(),
        }


def _make_code(rng: random.Random) -> str:
    return "".join(rng.choice(_CODE_ALPHABET) for _ in range(_CODE_LENGTH))


def make_records(seed: int = DEFAULT_SEED) -> list[NeedleRecord]:
    """Build the deterministic set of 8 answerable records + 1 adversarial record.

    Placement is seed-independent; only the record content (codes) is seeded.
    """
    rng = random.Random(seed)
    records: list[NeedleRecord] = []
    for index, (label, fraction) in enumerate(_NEEDLE_LAYOUT):
        records.append(
            NeedleRecord(
                needle_id=f"needle_{index:02d}",
                position_label=label,
                fraction=fraction,
                subject=_SUBJECTS[index],
                city=_CITIES[index],
                expected_answer=_make_code(rng),
                is_adversarial=False,
            )
        )
    adv_label, adv_fraction = _ADVERSARIAL_LAYOUT
    records.append(
        NeedleRecord(
            needle_id="needle_adv",
            position_label=adv_label,
            fraction=adv_fraction,
            subject=_SUBJECTS[len(_NEEDLE_LAYOUT)],
            city=_CITIES[len(_NEEDLE_LAYOUT)],
            expected_answer=NO_ANSWER,
            is_adversarial=True,
        )
    )
    return records


def expected_answer_map(records: Sequence[NeedleRecord]) -> dict[str, str]:
    """Extract ``{needle_id: expected_answer}`` including the marked adversarial case."""
    return {r.needle_id: r.expected_answer for r in records}


def _render_footer(records: Sequence[NeedleRecord]) -> str:
    lines = ["RETRIEVAL QUESTIONS. Answer each using only the document above."]
    lines.extend(r.question for r in records)
    lines.append("If a requested value is not stated in the document, reply exactly: NO ANSWER.")
    return " ".join(lines)


def _filler_text(word_count: int, start_index: int) -> tuple[str, int]:
    """Return ``word_count`` cyclic filler words joined by spaces, plus next index."""
    if word_count <= 0:
        return "", start_index
    words = [_FILLER_VOCAB[(start_index + i) % len(_FILLER_VOCAB)] for i in range(word_count)]
    return " ".join(words), start_index + word_count


def build_prompt(
    target_tokens: int,
    tokenizer: TokenizerLike,
    *,
    seed: int = DEFAULT_SEED,
) -> BuiltPrompt:
    """Build a prompt that encodes to *exactly* ``target_tokens`` under ``tokenizer``.

    Records are placed near their target document fractions with cyclic filler
    prose, and a trailing filler region is tuned by re-encoding until the exact
    token count is reached.
    """
    records = make_records(seed)
    ordered = sorted(records, key=lambda r: r.fraction)

    header_tokens = len(tokenizer.encode(_HEADER))
    footer = _render_footer(records)
    footer_tokens = len(tokenizer.encode(footer))
    needle_texts = [r.render() for r in ordered]
    needle_tokens = [len(tokenizer.encode(t)) for t in needle_texts]

    fixed_tokens = header_tokens + footer_tokens + sum(needle_tokens)
    if fixed_tokens > target_tokens:
        raise ValueError(
            f"target {target_tokens} tokens is smaller than the fixed content "
            f"({fixed_tokens} tokens); choose a larger target"
        )

    tokens_per_word = max(1, len(tokenizer.encode(_FILLER_PROBE)))

    # Distribute filler across the gaps so each record lands near its fraction.
    gap_word_counts: list[int] = []
    cumulative = header_tokens
    for index, record in enumerate(ordered):
        desired_offset = round(record.fraction * target_tokens)
        filler_tokens = max(0, desired_offset - cumulative)
        words = filler_tokens // tokens_per_word
        gap_word_counts.append(words)
        cumulative += words * tokens_per_word + needle_tokens[index]

    trailing_tokens = target_tokens - cumulative - footer_tokens
    trailing_words = max(0, trailing_tokens // tokens_per_word)

    def assemble(trailing: int) -> str:
        parts = [_HEADER]
        filler_index = 0
        for index in range(len(ordered)):
            gap_text, filler_index = _filler_text(gap_word_counts[index], filler_index)
            if gap_text:
                parts.append(gap_text)
            parts.append(needle_texts[index])
        tail_text, _ = _filler_text(trailing, filler_index)
        if tail_text:
            parts.append(tail_text)
        parts.append(footer)
        return " ".join(parts)

    text = assemble(trailing_words)
    count = len(tokenizer.encode(text))

    iterations = 0
    while count != target_tokens and iterations < _MAX_PADDING_ITERATIONS:
        diff = target_tokens - count
        if abs(diff) >= tokens_per_word:
            step = diff // tokens_per_word
        else:
            step = 1 if diff > 0 else -1
        trailing_words = max(0, trailing_words + step)
        text = assemble(trailing_words)
        count = len(tokenizer.encode(text))
        iterations += 1

    if count != target_tokens:
        raise RuntimeError(
            f"failed to converge to exact token target {target_tokens} "
            f"(reached {count}); the injected tokenizer may be non-additive"
        )

    return BuiltPrompt(
        target_tokens=target_tokens,
        token_count=count,
        seed=seed,
        text=text,
        needles=records,
    )


def build_all(
    tokenizer: TokenizerLike,
    *,
    seed: int = DEFAULT_SEED,
    sizes: Sequence[int] = TARGET_TOKEN_SIZES,
) -> dict[int, BuiltPrompt]:
    """Build one prompt per target size, keyed by target token count."""
    return {size: build_prompt(size, tokenizer, seed=seed) for size in sizes}


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI wrapper
    parser = argparse.ArgumentParser(description="Long-context corpus / retrieval-probe generator (plan T0.4)")
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="*",
        default=list(TARGET_TOKEN_SIZES),
        help="target token sizes to build metadata for",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--json-out", help="write per-size metadata (no bodies) here")
    args = parser.parse_args(argv)

    print(
        "This generator requires an injected tokenizer (encode + eos_id); the real "
        "Qwen4Exp tokenizer is not bundled. Import build_prompt/build_all from your "
        "harness and pass the checkpoint tokenizer. Requested sizes: "
        f"{args.sizes} (seed {args.seed})."
    )
    if args.json_out:
        payload = {
            "schema_version": SCHEMA_VERSION,
            "seed": args.seed,
            "sizes": args.sizes,
            "records": [asdict(r) for r in make_records(args.seed)],
        }
        with open(args.json_out, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
