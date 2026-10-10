# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Host-only pacing policy; elapsed step time is not a device profiler."""

import math
from dataclasses import dataclass

MAX_DECODE_ONLY_STEPS = 64


@dataclass(frozen=True)
class PrefillPacingConfig:
    target_step_ms: float = 500.0
    min_tokens: int = 128
    initial_tokens: int = 256
    max_tokens: int = 2560
    alignment: int = 128
    smoothing: float = 0.25
    # Opt-in decode cadence between mixed steps; zero retains prior behavior.
    decode_only_steps: int = 0

    def __post_init__(self):
        for name in ("min_tokens", "initial_tokens", "max_tokens", "alignment"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if not self.min_tokens <= self.initial_tokens <= self.max_tokens:
            raise ValueError("require min_tokens <= initial_tokens <= max_tokens")
        if any(value % self.alignment for value in (self.min_tokens, self.initial_tokens, self.max_tokens)):
            raise ValueError("token bounds must be multiples of alignment")
        if not math.isfinite(self.target_step_ms) or self.target_step_ms <= 0:
            raise ValueError("target_step_ms must be finite and positive")
        if not math.isfinite(self.smoothing) or not 0 < self.smoothing <= 1:
            raise ValueError("smoothing must be in (0, 1]")
        if type(self.decode_only_steps) is not int or not 0 <= self.decode_only_steps <= MAX_DECODE_ONLY_STEPS:
            raise ValueError(f"decode_only_steps must be an integer in [0, {MAX_DECODE_ONLY_STEPS}]")


class PrefillPacingPolicy:
    def __init__(self, config: PrefillPacingConfig):
        self.config = config
        self.ms_per_token = None

    def observe(self, prefill_tokens: int, elapsed_ms: float):
        if prefill_tokens <= 0 or not math.isfinite(elapsed_ms) or elapsed_ms <= 0:
            return
        observed = elapsed_ms / prefill_tokens
        if self.ms_per_token is None:
            self.ms_per_token = observed
        else:
            alpha = self.config.smoothing
            self.ms_per_token = alpha * observed + (1 - alpha) * self.ms_per_token

    def budget(self, maximum: int) -> int:
        c = self.config
        tokens = c.initial_tokens if self.ms_per_token is None else int(c.target_step_ms / self.ms_per_token)
        tokens = max(c.min_tokens, min(c.max_tokens, tokens))
        return min(maximum, tokens // c.alignment * c.alignment)
