# SPDX-License-Identifier: Apache-2.0
"""Explicit one-card expert placements, with unchanged native fallback banks."""

from dataclasses import dataclass

from tools.glm_perf.deployment_profiles import W4_LAYERS

SEED_PARTITION = "seed144"
PRECISION_PARTITION = "precision-resident"
PARTITIONS = (SEED_PARTITION, PRECISION_PARTITION)
MAIN_LAYERS = range(3, 45)
NATIVE_BANK_RANGES = ((144, 216), (216, 288))


@dataclass(frozen=True)
class OneCardPartition:
    name: str
    w4_cuda_ranges: tuple[tuple[int, int], ...]
    lowbit_cuda_ranges: tuple[tuple[int, int], ...]

    def ranges(self, layer):
        if type(layer) is not int or layer not in MAIN_LAYERS:
            raise ValueError("one-card partition requires a main MoE layer")
        return self.w4_cuda_ranges if layer in W4_LAYERS else self.lowbit_cuda_ranges

    def cuda_ids(self, layer):
        return tuple(expert for start, stop in self.ranges(layer) for expert in range(start, stop))

    def cuda_mask(self, ids, layer):
        intervals = self.ranges(layer)
        start, stop = intervals[0]
        mask = (ids >= start) & (ids < stop)
        for start, stop in intervals[1:]:
            mask = mask | ((ids >= start) & (ids < stop))
        return mask

    def native_enabled(self, layer):
        return len(self.cuda_ids(layer)) != 288

    def native_ranges(self, layer):
        self.ranges(layer)
        if self.name == SEED_PARTITION:
            return NATIVE_BANK_RANGES
        if layer in W4_LAYERS:
            return ((160, 216), (232, 288))
        return (None, None)

    def validate_layer_bits(self, bits):
        if self.name == SEED_PARTITION:
            return
        if any(
            type(bits.get(str(layer))) is not int or bits[str(layer)] not in ((4,) if layer in W4_LAYERS else (2, 3))
            for layer in MAIN_LAYERS
        ):
            raise ValueError("precision partition differs from the checkpoint's W4 layers")

    @property
    def resident_experts(self):
        return sum(len(self.cuda_ids(layer)) for layer in MAIN_LAYERS)


def one_card_partition(name=SEED_PARTITION):
    if name == SEED_PARTITION:
        return OneCardPartition(name, ((0, 144),), ((0, 144),))
    if name == PRECISION_PARTITION:
        # W4 keeps 56 active experts on each NPU; slower W2/W3 experts all
        # run on CUDA. Native banks remain loaded for placement qualification.
        return OneCardPartition(name, ((0, 160), (216, 232)), ((0, 288),))
    raise ValueError("unsupported one-card CUDA partition")
