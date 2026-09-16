# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Chunked-prefill configuration for Qwen4Exp QSA on Ascend 310P (plan T6.3).

The QSA side caches (raw-key ring + compressed history) are advanced one
scheduler chunk at a time during prefill. This module is the single, host-safe
config surface for that behaviour:

* :data:`QSA_DEFAULT_PREFILL_CHUNK_SIZE` -- the chunk-size knob (default 4096).
* :class:`QSAChunkPrefillPolicy` -- the chunk size plus a *preemption-aware
  recompute policy*.

Preemption policy (fail-closed)
-------------------------------
The QSA side-cache blocks are dropped when a request is preempted (mirroring the
GDN state preempt in :mod:`vllm_ascend.models.qwen4_exp.qwen4exp_gdn`). Because
the ring is a fixed circular buffer and the compressed history is addressed by
absolute logical position, a resumed request cannot inherit a partial side
cache; it must recompute from logical position 0. :class:`QSAChunkPrefillPolicy`
therefore defaults to ``recompute_from_scratch=True`` and its
:meth:`~QSAChunkPrefillPolicy.resume_offset` returns 0 for a preempted request.
This keeps a preempted-then-resumed run bit-identical to an unpreempted one, the
same guarantee the chunk-boundary math already gives for a single pass.

No ``torch_npu`` / Triton import lives here, so it materializes and unit-tests on
the 310P host lane.
"""

from __future__ import annotations

from dataclasses import dataclass

# Default chunked-prefill chunk size in tokens (the T6.3 knob). Sized well above
# the QSA raw-ring capacity so every full chunk fully refreshes the ring (the
# chunk-boundary identity requires ``chunk_size >= ring_capacity``).
QSA_DEFAULT_PREFILL_CHUNK_SIZE = 4096


@dataclass(frozen=True)
class QSAChunkPrefillPolicy:
    """Chunk size + preemption-aware recompute policy for QSA prefill.

    Attributes:
        chunk_size: prefill chunk length in tokens (default
            :data:`QSA_DEFAULT_PREFILL_CHUNK_SIZE`).
        recompute_from_scratch: when ``True`` (fail-closed default) a preempted
            request recomputes its whole side cache from logical position 0 on
            resume; when ``False`` the caller may resume from the number of
            already-committed tokens (only valid if the side-cache blocks were
            retained across preemption).
    """

    chunk_size: int = QSA_DEFAULT_PREFILL_CHUNK_SIZE
    recompute_from_scratch: bool = True

    def __post_init__(self) -> None:
        if self.chunk_size <= 0:
            raise ValueError("QSA prefill chunk_size must be positive")

    def validate_for_ring(self, ring_capacity: int) -> None:
        """Ensure the chunk size preserves ring chunk-boundary identity.

        A chunk shorter than the ring capacity cannot fully refresh the circular
        buffer, so the final ring state could depend on chunk boundaries. The
        default chunk size (4096) clears any realistic ring capacity.
        """
        if ring_capacity <= 0:
            raise ValueError("ring capacity must be positive")
        if self.chunk_size < ring_capacity:
            raise ValueError(
                f"QSA prefill chunk_size {self.chunk_size} must be >= ring capacity "
                f"{ring_capacity} to keep the ring chunk-boundary invariant"
            )

    def resume_offset(self, committed_tokens: int) -> int:
        """Logical position a resumed request restarts prefill from.

        Returns 0 under the fail-closed default (recompute the whole side cache);
        otherwise the already-committed token count.
        """
        if committed_tokens < 0:
            raise ValueError("committed_tokens must be non-negative")
        return 0 if self.recompute_from_scratch else committed_tokens


__all__ = [
    "QSA_DEFAULT_PREFILL_CHUNK_SIZE",
    "QSAChunkPrefillPolicy",
]
