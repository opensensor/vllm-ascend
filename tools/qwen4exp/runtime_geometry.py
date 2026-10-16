# SPDX-License-Identifier: Apache-2.0
"""Admission for QFN TP4/TP6 resident optimization owners."""

MAX_REQUESTS = 6
STATE_GEOMETRIES = frozenset(((2560, 12), (1920, 9), (1280, 6)))


def request_capacity(runner):
    count = runner.scheduler_config.max_num_seqs
    if type(count) is not int or not 1 <= count <= MAX_REQUESTS:
        raise ValueError("resident QFN owners support one through six request slots")
    return count


def state_geometry(convolution, recurrent):
    """Return actual slot capacity after checking the paired shard geometry.

    TP6 partitions value heads as 9/9/9/9/6/6. Slot count is a separate
    scheduler allocation, and must not be inferred from tensor parallel size.
    """
    conv, state = tuple(convolution.shape), tuple(recurrent.shape)
    if (
        len(conv) != 3
        or len(state) != 4
        or conv[0] <= 0
        or conv[0] != state[0]
        or conv[1] != 5
        or state[2:] != (128, 128)
        or (conv[2], state[1]) not in STATE_GEOMETRIES
    ):
        raise ValueError("live state differs from supported paired QFN shard geometry")
    return conv[0]
