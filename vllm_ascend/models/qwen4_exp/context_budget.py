# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Context-window budgeting + admission for the Qwen4Exp 1M path (plan T7.2).

This is the single, host-safe admission/validation surface that enforces the
authoritative 1,048,576-token context budget for the Ascend 310P Qwen4Exp
(Qwen3.8-Flash-Next) 1M deployment. The ceiling is imported from the T7.1 RoPE
config so the budget and the validated RoPE extension window can never drift.

Budget split (must sum to the ceiling)::

    1,048,060 input + 512 output + 4 draft = 1,048,576

Design rules (from the PRD / plan):

* **No truncation (R10).** An over-budget prompt is REJECTED with an explicit
  :class:`ContextBudgetExceededError`; it is never silently trimmed to fit. The
  accepted-token count reported back is exactly the submitted prompt length.
* **Concurrency = 1 for the 1M path.** Both admission and execution are capped
  at one in-flight 1M request. A second concurrent 1M request is REJECTED with
  :class:`ConcurrencyLimitExceededError` (fail-closed; see
  :class:`Qwen4Exp1MAdmissionController`).

No ``torch_npu`` / Triton import lives here, so it materializes and unit-tests on
the 310P host lane.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

from vllm_ascend._310p.worker.v2.rope import MAX_EXTENDED_POSITION_EMBEDDINGS

# ---------------------------------------------------------------------------
# Authoritative 1M budget constants
# ---------------------------------------------------------------------------
#
# The ceiling is the T7.1 extended-position ceiling (1,048,576); the split below
# is defined to sum EXACTLY to it, so an at-limit request accepts and any excess
# rejects.
QWEN4EXP_1M_CONTEXT_CEILING = MAX_EXTENDED_POSITION_EMBEDDINGS  # 1,048,576
QWEN4EXP_1M_MAX_INPUT_TOKENS = 1_048_060
QWEN4EXP_1M_MAX_OUTPUT_TOKENS = 512
QWEN4EXP_1M_MAX_DRAFT_TOKENS = 4

# Admission + execution concurrency for the 1M path (fail-closed at one).
QWEN4EXP_1M_MAX_CONCURRENCY = 1


class ContextBudgetError(ValueError):
    """Base class for 1M context-budget admission failures."""


class ContextBudgetExceededError(ContextBudgetError):
    """Raised when a request cannot fit the 1M budget (no truncation)."""


class ConcurrencyLimitExceededError(ContextBudgetError):
    """Raised when a second concurrent 1M request is submitted (limit = 1)."""


@dataclass(frozen=True)
class ContextBudgetPolicy:
    """The token budget + concurrency policy for a 1M deployment.

    The three token components must sum to no more than ``context_ceiling``; the
    authoritative :meth:`default` policy makes them sum to it exactly.
    """

    max_input_tokens: int
    max_output_tokens: int
    max_draft_tokens: int
    context_ceiling: int
    max_concurrency: int

    def __post_init__(self) -> None:
        for name in (
            "max_input_tokens",
            "max_output_tokens",
            "max_draft_tokens",
            "context_ceiling",
            "max_concurrency",
        ):
            value = getattr(self, name)
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        # Each per-component cap is an upper bound and must fit inside the
        # ceiling on its own; the total (input + output + draft) is checked at
        # admission time and is the binding constraint.
        for name in ("max_input_tokens", "max_output_tokens", "max_draft_tokens"):
            value = getattr(self, name)
            if value > self.context_ceiling:
                raise ValueError(f"{name} ({value}) exceeds context ceiling {self.context_ceiling}")

    @classmethod
    def default(cls) -> ContextBudgetPolicy:
        """The authoritative Qwen4Exp 1,048,576-token policy."""
        return cls(
            max_input_tokens=QWEN4EXP_1M_MAX_INPUT_TOKENS,
            max_output_tokens=QWEN4EXP_1M_MAX_OUTPUT_TOKENS,
            max_draft_tokens=QWEN4EXP_1M_MAX_DRAFT_TOKENS,
            context_ceiling=QWEN4EXP_1M_CONTEXT_CEILING,
            max_concurrency=QWEN4EXP_1M_MAX_CONCURRENCY,
        )


@dataclass(frozen=True)
class ContextBudgetDecision:
    """A successful admission decision.

    ``accepted_prompt_tokens`` is exactly the submitted prompt length -- the
    prompt is never truncated -- and is meant to be surfaced in response
    metadata via :meth:`to_metadata`.
    """

    accepted_prompt_tokens: int
    output_tokens: int
    draft_tokens: int
    total_tokens: int
    context_ceiling: int
    accepted: bool = True

    def to_metadata(self) -> dict[str, int | bool]:
        """Response-metadata view (accepted token count included)."""
        return {
            "accepted_prompt_tokens": self.accepted_prompt_tokens,
            "output_tokens": self.output_tokens,
            "draft_tokens": self.draft_tokens,
            "total_tokens": self.total_tokens,
            "context_ceiling": self.context_ceiling,
            "truncated": False,
        }


def check_context_budget(
    input_tokens: int,
    *,
    output_tokens: int | None = None,
    draft_tokens: int | None = None,
    policy: ContextBudgetPolicy | None = None,
) -> ContextBudgetDecision:
    """Validate a request against the 1M budget, without truncating.

    ``output_tokens`` / ``draft_tokens`` default to the policy maxima (a request
    reserving its full output + draft budget). An at-limit request
    (``input + output + draft == ceiling``) is accepted; any excess raises
    :class:`ContextBudgetExceededError` with an explicit message. The returned
    :class:`ContextBudgetDecision` reports the accepted prompt length verbatim.
    """
    policy = policy or ContextBudgetPolicy.default()
    if output_tokens is None:
        output_tokens = policy.max_output_tokens
    if draft_tokens is None:
        draft_tokens = policy.max_draft_tokens

    if input_tokens < 0:
        raise ContextBudgetExceededError(f"input_tokens must be non-negative, got {input_tokens}")
    if output_tokens < 0:
        raise ContextBudgetExceededError(f"output_tokens must be non-negative, got {output_tokens}")
    if draft_tokens < 0:
        raise ContextBudgetExceededError(f"draft_tokens must be non-negative, got {draft_tokens}")

    if input_tokens > policy.max_input_tokens:
        raise ContextBudgetExceededError(
            f"Input length ({input_tokens}) exceeds the 1M input budget "
            f"({policy.max_input_tokens}). The prompt is rejected, not truncated "
            "(R10): shorten the prompt and resubmit."
        )
    if output_tokens > policy.max_output_tokens:
        raise ContextBudgetExceededError(
            f"Requested output tokens ({output_tokens}) exceed the 1M output budget ({policy.max_output_tokens})."
        )
    if draft_tokens > policy.max_draft_tokens:
        raise ContextBudgetExceededError(
            f"Requested draft tokens ({draft_tokens}) exceed the 1M draft budget ({policy.max_draft_tokens})."
        )

    total_tokens = input_tokens + output_tokens + draft_tokens
    if total_tokens > policy.context_ceiling:
        raise ContextBudgetExceededError(
            f"Total context ({input_tokens} input + {output_tokens} output + "
            f"{draft_tokens} draft = {total_tokens}) exceeds the 1M context "
            f"ceiling ({policy.context_ceiling}). The prompt is rejected, not "
            "truncated (R10)."
        )

    return ContextBudgetDecision(
        accepted_prompt_tokens=input_tokens,
        output_tokens=output_tokens,
        draft_tokens=draft_tokens,
        total_tokens=total_tokens,
        context_ceiling=policy.context_ceiling,
    )


class Qwen4Exp1MAdmissionController:
    """Fail-closed admission gate for the 1M path (concurrency = 1).

    Admission first validates the token budget (an over-budget request is
    rejected with :class:`ContextBudgetExceededError` and never reserves a
    slot), then reserves one of ``policy.max_concurrency`` in-flight slots. With
    the default policy that limit is one: while a 1M request is in flight, a
    second :meth:`admit` raises :class:`ConcurrencyLimitExceededError`. The slot
    is held through execution and freed by :meth:`release` (or automatically by
    the :meth:`admission_slot` context manager), so admission and execution
    concurrency are both capped at one. Thread-safe.
    """

    def __init__(self, policy: ContextBudgetPolicy | None = None) -> None:
        self._policy = policy or ContextBudgetPolicy.default()
        self._lock = threading.Lock()
        self._in_flight: set[str] = set()

    @property
    def policy(self) -> ContextBudgetPolicy:
        return self._policy

    @property
    def in_flight(self) -> int:
        with self._lock:
            return len(self._in_flight)

    def admit(
        self,
        request_id: str,
        input_tokens: int,
        *,
        output_tokens: int | None = None,
        draft_tokens: int | None = None,
    ) -> ContextBudgetDecision:
        """Validate the budget and reserve a concurrency slot.

        Raises :class:`ContextBudgetExceededError` for an over-budget request
        (no slot consumed) or :class:`ConcurrencyLimitExceededError` when the
        1M path is already at capacity.
        """
        decision = check_context_budget(
            input_tokens,
            output_tokens=output_tokens,
            draft_tokens=draft_tokens,
            policy=self._policy,
        )
        with self._lock:
            if request_id in self._in_flight:
                raise ConcurrencyLimitExceededError(f"request {request_id!r} is already admitted on the 1M path")
            if len(self._in_flight) >= self._policy.max_concurrency:
                raise ConcurrencyLimitExceededError(
                    f"1M path concurrency limit ({self._policy.max_concurrency}) "
                    f"reached; request {request_id!r} rejected while another 1M "
                    "request is in flight."
                )
            self._in_flight.add(request_id)
        return decision

    def release(self, request_id: str) -> None:
        """Free the slot held by ``request_id`` (idempotent)."""
        with self._lock:
            self._in_flight.discard(request_id)

    @contextmanager
    def admission_slot(
        self,
        request_id: str,
        input_tokens: int,
        *,
        output_tokens: int | None = None,
        draft_tokens: int | None = None,
    ) -> Iterator[ContextBudgetDecision]:
        """Admit for the duration of a ``with`` block, releasing on exit."""
        decision = self.admit(
            request_id,
            input_tokens,
            output_tokens=output_tokens,
            draft_tokens=draft_tokens,
        )
        try:
            yield decision
        finally:
            self.release(request_id)


__all__ = [
    "QWEN4EXP_1M_CONTEXT_CEILING",
    "QWEN4EXP_1M_MAX_INPUT_TOKENS",
    "QWEN4EXP_1M_MAX_OUTPUT_TOKENS",
    "QWEN4EXP_1M_MAX_DRAFT_TOKENS",
    "QWEN4EXP_1M_MAX_CONCURRENCY",
    "ContextBudgetError",
    "ContextBudgetExceededError",
    "ConcurrencyLimitExceededError",
    "ContextBudgetPolicy",
    "ContextBudgetDecision",
    "Qwen4Exp1MAdmissionController",
    "check_context_budget",
]
