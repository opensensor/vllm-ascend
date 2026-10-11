# SPDX-License-Identifier: Apache-2.0
"""Expert-only execution seam; accelerator backends can implement the callable."""

import math
from dataclasses import dataclass
from typing import Protocol

from .placement import ExpertPlacement
from .reference import ExpertReply, ExpertRequest, fp32
from .wire import PacketLimits, validate_packet


class ExpertBackend(Protocol):
    def prepare(self) -> tuple[tuple[int, int], ...]:
        """Load/verify local banks and return resident (layer, global expert) keys."""
        ...

    def compute(self, request: ExpertRequest) -> tuple[tuple[float, ...], ...]:
        """Compute all local routes together, returning raw outputs in route order."""
        ...


@dataclass(frozen=True)
class ExpertWorker:
    owner: str
    placement: ExpertPlacement
    limits: PacketLimits
    backend: ExpertBackend

    def prepare(self):
        resident = self.backend.prepare()
        expected = {
            (layer, expert)
            for layer in range(
                self.placement.first_moe_layer, self.placement.first_moe_layer + len(self.placement.layers)
            )
            for expert, owner in enumerate(self.placement.layer(layer))
            if owner == self.owner
        }
        if len(resident) != len(set(resident)) or set(resident) != expected:
            raise ValueError("backend resident expert coverage differs from placement")

    def execute(self, request: ExpertRequest) -> ExpertReply:
        validate_packet(request, self.limits)
        if not isinstance(request, ExpertRequest) or request.owner != self.owner:
            raise ValueError("worker received foreign request")
        layer = self.placement.layer(request.layer_id)
        if any(route.expert >= len(layer) or layer[route.expert] != self.owner for route in request.routes):
            raise ValueError("worker received unowned expert")
        outputs = self.backend.compute(request)
        reply = ExpertReply(request.execution_id, request.layer_id, request.owner, request.routes, outputs)
        validate_packet(reply, self.limits)
        return reply


def _matvec(matrix, vector):
    result = []
    for row in matrix:
        total = 0.0
        for weight, value in zip(row, vector, strict=True):
            total = fp32(total + fp32(fp32(weight) * fp32(value)))
        result.append(total)
    return tuple(result)


@dataclass(frozen=True)
class DenseExpert:
    """Tiny FP32 SwiGLU oracle; never a substitute for packed GLM kernels."""

    gate: tuple[tuple[float, ...], ...]
    up: tuple[tuple[float, ...], ...]
    down: tuple[tuple[float, ...], ...]

    def __post_init__(self):
        if any(
            not isinstance(matrix, tuple) or any(not isinstance(row, tuple) for row in matrix)
            for matrix in (self.gate, self.up, self.down)
        ):
            raise ValueError("expert weights require immutable tuples")
        if not self.gate or not self.gate[0] or len(self.up) != len(self.gate) or len(self.down) != len(self.gate[0]):
            raise ValueError("invalid expert matrix dimensions")
        hidden, intermediate = len(self.gate[0]), len(self.gate)
        for matrix, width in ((self.gate, hidden), (self.up, hidden), (self.down, intermediate)):
            if any(len(row) != width or any(not math.isfinite(value) for value in row) for row in matrix):
                raise ValueError("invalid expert matrix shape or value")

    def __call__(self, hidden):
        gate, up = _matvec(self.gate, hidden), _matvec(self.up, hidden)
        activated = []
        for g, u in zip(gate, up):
            sigmoid = 1 / (1 + math.exp(-g)) if g >= 0 else math.exp(g) / (1 + math.exp(g))
            activated.append(fp32(fp32(g * sigmoid) * u))
        return _matvec(self.down, activated)


@dataclass(frozen=True)
class DenseExpertBank:
    """Explicit resident local indices, including noncontiguous global experts."""

    experts: tuple[tuple[int, int, DenseExpert], ...]

    def __post_init__(self):
        if not isinstance(self.experts, tuple) or any(not isinstance(entry, tuple) for entry in self.experts):
            raise ValueError("expert bank requires immutable tuples")
        keys = [(layer, expert) for layer, expert, _ in self.experts]
        if len(keys) != len(set(keys)):
            raise ValueError("duplicate expert bank entry")

    def __call__(self, layer_id, expert_id, hidden):
        for layer, expert, weights in self.experts:
            if (layer, expert) == (layer_id, expert_id):
                return weights(hidden)
        raise ValueError("expert not resident in bank")

    def compute(self, request):
        hidden = dict(zip(request.token_ids, request.hidden))
        return tuple(self(request.layer_id, route.expert, hidden[route.token]) for route in request.routes)

    def prepare(self):
        return tuple((layer, expert) for layer, expert, _ in self.experts)
