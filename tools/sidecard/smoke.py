# SPDX-License-Identifier: Apache-2.0
"""Five isolated CPU workers with resident tiny SwiGLU experts and binary IPC."""

import hashlib
import json
import struct

from .placement import ASCEND_OWNERS, CUDA_OWNER, ExpertPlacement
from .reference import ExpertReply, combine, dispatch
from .runtime import LocalWorkerPool
from .wire import PacketLimits, WireIdentity
from .worker import DenseExpert, DenseExpertBank

SMOKE_HIDDEN = 4
SMOKE_INTERMEDIATE = 3
SMOKE_EXPERTS = 10
SMOKE_TOP_K = 4
SMOKE_TOKENS = 3
SMOKE_STEPS = 3


def smoke_fixture():
    owners = (CUDA_OWNER, *ASCEND_OWNERS)
    placement = ExpertPlacement(owners, (tuple(owners[expert % len(owners)] for expert in range(SMOKE_EXPERTS)),))
    experts = []
    for expert in range(SMOKE_EXPERTS):
        factor = (expert + 1) / 32
        gate = tuple(tuple(factor * (i + j + 1) for j in range(SMOKE_HIDDEN)) for i in range(SMOKE_INTERMEDIATE))
        up = tuple(tuple(factor * (i - j + 1) for j in range(SMOKE_HIDDEN)) for i in range(SMOKE_INTERMEDIATE))
        down = tuple(tuple(factor * (i + j - 1) for j in range(SMOKE_INTERMEDIATE)) for i in range(SMOKE_HIDDEN))
        experts.append((3, expert, DenseExpert(gate, up, down)))
    checkpoint = json.dumps(
        [(layer, expert, weights.__dict__) for layer, expert, weights in experts], sort_keys=True
    ).encode()
    identity = WireIdentity.for_placement(placement, hashlib.sha256(checkpoint).hexdigest())
    backends = {
        owner: DenseExpertBank(tuple(row for row in experts if placement.layer(row[0])[row[1]] == owner))
        for owner in owners
    }
    limits = PacketLimits(SMOKE_HIDDEN, SMOKE_TOKENS, SMOKE_TOKENS * SMOKE_TOP_K, SMOKE_EXPERTS, SMOKE_TOP_K)
    return placement, limits, identity, backends, DenseExpertBank(tuple(experts))


def run_smoke():
    placement, limits, identity, backends, serial_bank = smoke_fixture()
    hidden = ((0.1, -0.2, 0.3, 0.4), (-0.5, 0.6, -0.7, 0.8), (0.9, 1.0, -1.1, 1.2))
    # FP16 wire rounding is part of the candidate and must also reach the control.
    hidden = tuple(tuple(struct.unpack("<e", struct.pack("<e", value))[0] for value in row) for row in hidden)
    expert_ids = ((0, 1, 2, 3), (4, 5, 6, 7), (8, 9, 0, 1))
    weights = ((0.1, 0.2, 0.3, 0.4),) * SMOKE_TOKENS
    shared = ((0.01, 0.02, 0.03, 0.04),) * SMOKE_TOKENS
    replies = []
    for request in dispatch(placement, "serial", 3, hidden, expert_ids, weights):
        rows = tuple(serial_bank(3, route.expert, hidden[route.token]) for route in request.routes)
        replies.append(ExpertReply("serial", 3, request.owner, request.routes, rows))
    expected = combine(placement, "serial", 3, hidden, expert_ids, weights, replies, shared, 2.5)
    with LocalWorkerPool(placement, limits, identity, backends) as pool:
        pids = {owner: process.pid for owner, process in pool.processes.items()}
        outputs = [pool.execute(3, hidden, expert_ids, weights, shared, 2.5) for _ in range(SMOKE_STEPS)]
    if any(output != expected for output in outputs):
        raise AssertionError("partitioned IPC result differs from serial control")
    return {
        "status": "offline_process_smoke_passed",
        "hardware_validated": False,
        "real_glm_weights": False,
        "worker_pids": pids,
        "steps": SMOKE_STEPS,
        "hidden_width": SMOKE_HIDDEN,
        "placement_sha256": identity.placement_sha256,
        "checkpoint_sha256": identity.checkpoint_sha256,
        "matches_serial_fp32": True,
        "outputs": expected,
    }
