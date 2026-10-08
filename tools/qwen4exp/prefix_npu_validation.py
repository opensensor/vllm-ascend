# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Diagnostic checkpoint-pressure RPC on an exclusively owned test server.

The controller must drain and clear scheduler caches before and after this RPC.
It uses the loaded model's real state tensors, not synthetic state geometry.
It does not evaluate language-model accuracy or token throughput.
"""

import time

import numpy as np
import torch

from tools.qwen4exp.resident_worker import QwenResidentExtension
from vllm_ascend._310p.prefix_mamba_state import retain_prefix_mamba_blocks

TEST_REQUESTS = 3
PRESSURE_ROUNDS = 96
FIRST_DIAGNOSTIC_BLOCK_ID = 1_000_000
BOUNDED_CHECKPOINTS = 27


class QwenPrefixValidationExtension(QwenResidentExtension):
    def resident_prefix_pressure(self, policy):
        if policy not in {"baseline", "bounded"}:
            raise ValueError("Unknown diagnostic prefix policy")
        tiers = self.model_runner._prefix_mamba_tiers
        before = self.resident_status()
        if any(tier._resident or tier._host or tier._device_archive_resident for tier in tiers.values()):
            raise RuntimeError("Drain and reset the exclusively owned test server before the diagnostic")
        pointers = {
            group: [
                tensor.data_ptr()
                for tensor in (
                    *(tensor for layer in tier.layer_states for tensor in layer),
                    *tier._device_archive,
                    *tier._swap_tensors,
                )
            ]
            for group, tier in tiers.items()
        }
        torch.npu.synchronize()
        started = time.perf_counter()
        history = []
        previous = []
        try:
            for step in range(PRESSURE_ROUNDS):
                current = [FIRST_DIAGNOSTIC_BLOCK_ID + step * TEST_REQUESTS + row for row in range(TEST_REQUESTS)]
                rows = [[previous[row], current[row]] if previous else [current[row]] for row in range(TEST_REQUESTS)]
                if policy == "bounded":
                    retained = sorted(set(history[-BOUNDED_CHECKPOINTS:] + previous + current))
                    retain_prefix_mamba_blocks(tiers, {group: retained for group in tiers})
                table = np.asarray(rows, dtype=np.int32)
                for tier in tiers.values():
                    mapped = tier.remap_rows(
                        table, [table.shape[1]] * TEST_REQUESTS, [tuple(range(table.shape[1]))] * TEST_REQUESTS
                    )
                    for row in range(TEST_REQUESTS):
                        for tensor in tier._slot_tensors(int(mapped[row, -1])):
                            tensor.fill_(step * TEST_REQUESTS + row + 1)
                history.extend(current)
                previous = current
            torch.npu.synchronize()
            pressure_elapsed = time.perf_counter() - started
            # Correctness checks are outside the timed pressure loop.
            probes = [
                (block_id, PRESSURE_ROUNDS * TEST_REQUESTS - TEST_REQUESTS + row + 1)
                for row, block_id in enumerate(previous)
            ]
            if policy == "baseline":
                probes.append((FIRST_DIAGNOSTIC_BLOCK_ID, 1))
            for tier in tiers.values():
                for block_id, expected in probes:
                    tier.remap_table(np.asarray([[block_id]], dtype=np.int32), 1)
                    torch.npu.synchronize()
                    if not all(
                        bool(torch.all(tensor == expected)) for tensor in tier._slot_tensors(tier.slot_for(block_id))
                    ):
                        raise RuntimeError(f"Checkpoint value changed for block {block_id}")
            after = self.resident_status()
            if pointers != {
                group: [
                    tensor.data_ptr()
                    for tensor in (
                        *(tensor for layer in tier.layer_states for tensor in layer),
                        *tier._device_archive,
                        *tier._swap_tensors,
                    )
                ]
                for group, tier in tiers.items()
            }:
                raise RuntimeError("Graph-visible checkpoint storage changed")
            return {
                "policy": policy,
                "rank": before["rank"],
                "rounds": PRESSURE_ROUNDS,
                "checkpoint_bytes_per_group": {str(group): tier._bytes_per_slot for group, tier in tiers.items()},
                "elapsed_s": pressure_elapsed,
                "before": before,
                "after": after,
                "value_checks_passed": True,
                "storage_preserved": True,
            }
        finally:
            torch.npu.synchronize()
            for tier in tiers.values():
                tier.reset()
