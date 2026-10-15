# SPDX-License-Identifier: Apache-2.0
"""Private native state-commit prototype; not installed in serving.

Supports contiguous temporal states and SD convolution states with 32-byte row
alignment. The kernel follows request order, copying overlapping SD suffixes
forward. Unsupported layouts must retain the current fallback. Error receipts
are diagnostic outputs, not an automatic serving recovery mechanism.
"""

import torch

PADDED_ROWS = 8
WORD_BYTES = 4
COPY_ALIGNMENT = 32
MAX_STATE_BANKS = 256


def group_geometry(banks, tables, convolution, *, block_size):
    """Cold admission: native cores must never write overlapping state banks."""
    if not 1 <= len(banks) <= MAX_STATE_BANKS or not len(banks) == len(tables) == len(convolution):
        raise ValueError("state group requires 1..256 matching banks, tables and layouts")
    configs = [
        geometry(bank, table, conv=conv, block_size=block_size) for bank, table, conv in zip(banks, tables, convolution)
    ]
    if len({bank.device for bank in banks}) != 1 or len({cfg[0] for cfg in configs}) != 1:
        raise ValueError("state group requires one device and one request count")
    intervals = sorted((bank.data_ptr(), bank.data_ptr() + bank.numel() * bank.element_size()) for bank in banks)
    if any(left[1] > right[0] for left, right in zip(intervals, intervals[1:])):
        raise ValueError("parallel state banks must not overlap")
    for table in tables:
        start, end = table.data_ptr(), table.data_ptr() + table.numel() * table.element_size()
        if any(start < bank_end and bank_start < end for bank_start, bank_end in intervals):
            raise ValueError("block tables must not alias writable state banks")
    return configs


def geometry(state, table, *, conv, block_size):
    if type(conv) is not bool or type(block_size) is not int or not 0 < block_size < (1 << 31):
        raise ValueError("explicit convolution flag and positive INT32 block size required")
    if state.ndim < 2 or state.dtype not in (torch.float16, torch.float32, torch.int32) or not state.is_contiguous():
        raise ValueError("state must be a contiguous FP16/FP32 or raw INT32 bank")
    if table.ndim != 2 or not 1 <= table.shape[0] <= PADDED_ROWS or table.shape[1] < 1:
        raise ValueError("block table must have 1..8 rows and nonempty capacity")
    if table.dtype != torch.int32 or not table.is_contiguous() or table.device != state.device:
        raise ValueError("block table must be contiguous INT32 on the state device")
    if state.shape[0] < 1:
        raise ValueError("empty state bank")
    block_bytes = state[0].numel() * state.element_size()
    if block_bytes < COPY_ALIGNMENT or block_bytes % COPY_ALIGNMENT:
        raise ValueError("state block size must be a positive multiple of 32 bytes")
    if conv and state.ndim != 3:
        raise ValueError("convolution requires SD [blocks, length, width] layout")
    width_bytes = state.shape[-1] * state.element_size() if conv else 0
    if width_bytes % COPY_ALIGNMENT:
        raise ValueError("SD row must be a multiple of 32 bytes")
    return (
        table.shape[0],
        table.shape[1],
        state.shape[0],
        block_bytes // WORD_BYTES,
        width_bytes // WORD_BYTES,
        block_size,
    )


class MambaStateCommit:
    """Keep bank, descriptor, receipt and native code alive across graph replay."""

    def __init__(self, kernel, launch, state, table, *, conv, block_size):
        config = geometry(state, table, conv=conv, block_size=block_size)
        if state.device.type != "npu":
            raise ValueError("native state commit requires an NPU")
        if torch.npu.is_current_stream_capturing():
            raise RuntimeError("initialize commit storage before capture")
        self.kernel, self.launch, self.state, self.table = kernel, launch, state, table
        self.config = torch.tensor(config, dtype=torch.int64, device=state.device)
        self.receipt = torch.empty((2, PADDED_ROWS), dtype=torch.int32, device=state.device)

    def __call__(self, inputs):
        if (
            inputs.shape != (5, PADDED_ROWS)
            or inputs.dtype != torch.int32
            or inputs.device != self.state.device
            or not inputs.is_contiguous()
        ):
            raise ValueError("commit inputs must be contiguous INT32 [5,8] on the state device")
        self.launch(self.kernel, [self.state, inputs, self.table, self.receipt, self.config], 1)
        return self.receipt


class MambaCommitGroup:
    """One native launch for all banks; retains every device pointer target.

    Construction is cold-only. Descriptor addresses and geometry are immutable;
    only input values and block-table contents may change between replays.
    """

    def __init__(self, kernel, launch, banks, tables, convolution, *, block_size):
        configs = group_geometry(banks, tables, convolution, block_size=block_size)
        device = banks[0].device
        if device.type != "npu" or torch.npu.is_current_stream_capturing():
            raise ValueError("construct group on one NPU before capture")
        self.kernel, self.launch = kernel, launch
        self.banks, self.tables = tuple(banks), tuple(tables)
        records = [[bank.data_ptr(), table.data_ptr(), *cfg] for bank, table, cfg in zip(banks, tables, configs)]
        self.descriptors = torch.tensor(records, dtype=torch.int64, device=device)
        self.config = torch.tensor([len(banks)], dtype=torch.int64, device=device)
        self.receipt = torch.empty((len(banks), 2, PADDED_ROWS), dtype=torch.int32, device=device)

    def __call__(self, inputs):
        if (
            inputs.shape != (5, PADDED_ROWS)
            or inputs.dtype != torch.int32
            or inputs.device != self.banks[0].device
            or not inputs.is_contiguous()
        ):
            raise ValueError("commit inputs must be contiguous INT32 [5,8] on the state device")
        self.launch(self.kernel, [inputs, self.descriptors, self.receipt, self.config], 8)
        return self.receipt


def reference(state, inputs, table, *, conv, block_size):
    """CPU-only independent oracle for the bounded native copy contract."""
    config = geometry(state, table, conv=conv, block_size=block_size)
    if any(t.device.type != "cpu" for t in (state, inputs, table)):
        raise ValueError("oracle is CPU only")
    rows, columns, blocks, words, width, _ = config
    bank = state.clone()
    raw = bank.view(torch.int32).reshape(blocks, words)
    receipt = torch.zeros((2, PADDED_ROWS), dtype=torch.int32)
    for row in range(rows):
        source, computed, scheduled, drafts, accepted = inputs[:, row].tolist()
        receipt[0, row] = accepted
        if min(source, computed, drafts, accepted) < 0 or scheduled <= drafts or accepted > drafts + 1:
            receipt[1, row] = 1
            continue
        running = computed + scheduled - drafts
        aligned = (running + accepted - 1) // block_size * block_size
        destination, bias = aligned // block_size - 1, aligned - running
        if aligned < running:
            continue
        if source == destination:
            receipt[0, row] = 1
            if bias == 0:
                continue
        source_index = source if conv else source + bias
        if not 0 <= source_index < columns or not 0 <= destination < columns or not 0 <= bias <= drafts:
            receipt[1, row] = 2
            continue
        src, dst = int(table[row, source_index]), int(table[row, destination])
        offset = width * bias
        if not 0 <= src < blocks or not 0 <= dst < blocks or offset >= words:
            receipt[1, row] = 3
            continue
        raw[dst, : words - offset].copy_(raw[src, offset:].clone())
    return bank, receipt
