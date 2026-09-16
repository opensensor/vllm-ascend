# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Selected-row transfer accounting for the QSA-aware DCP prototype (T8.2).

The defining property of Candidate B is that the cross-rank exchange moves ONLY
the indexer-selected K/V rows, never an all-gather of the (up to 1M-row) main
cache. This ledger records exactly how many cache rows crossed the exchange so a
host unit test can prove ``bytes_moved == selected_rows * row_bytes`` and that it
is a tiny fraction of the whole-cache all-gather it replaces.

Byte figures feed the T0.5 memory-accounting harness
(``observability/qwen38_mem_accounting``); this module only counts rows and
multiplies by the row stride, so it stays a pure-Python, importable helper.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

# The exchange carries a Key row and a Value row for every selected position.
_KV_TENSORS_PER_ROW = 2


def row_bytes(num_kv_heads: int, head_dim: int, dtype: torch.dtype) -> int:
    """Byte stride of one K (or V) cache row: ``num_kv_heads * head_dim`` elements."""
    element_size = torch.empty(0, dtype=dtype).element_size()
    return int(num_kv_heads) * int(head_dim) * element_size


@dataclass
class TransferLedger:
    """Accumulates the selected K/V rows moved across the DCP exchange.

    Attributes:
        row_bytes: byte stride of a single K (or V) cache row.
        key_rows_moved: total selected Key rows placed on the exchange.
        value_rows_moved: total selected Value rows placed on the exchange.
    """

    row_bytes: int
    key_rows_moved: int = 0
    value_rows_moved: int = 0

    def record_selected_rows(self, num_rows: int) -> None:
        """Record ``num_rows`` selected positions exchanged (one K + one V each)."""
        if num_rows < 0:
            raise ValueError("num_rows must be >= 0")
        self.key_rows_moved += int(num_rows)
        self.value_rows_moved += int(num_rows)

    @property
    def selected_rows(self) -> int:
        """Selected positions exchanged (K and V move together, so K == V count)."""
        if self.key_rows_moved != self.value_rows_moved:
            raise AssertionError("K and V row counts diverged; the exchange is asymmetric")
        return self.key_rows_moved

    @property
    def bytes_moved(self) -> int:
        """Total bytes crossing the exchange (K rows + V rows)."""
        return (self.key_rows_moved + self.value_rows_moved) * self.row_bytes

    def whole_cache_bytes(self, context_len: int) -> int:
        """Bytes an all-gather of the whole main K/V cache would move instead."""
        return int(context_len) * _KV_TENSORS_PER_ROW * self.row_bytes

    def to_dict(self) -> dict:
        return {
            "row_bytes": self.row_bytes,
            "selected_rows": self.selected_rows,
            "key_rows_moved": self.key_rows_moved,
            "value_rows_moved": self.value_rows_moved,
            "bytes_moved": self.bytes_moved,
        }


__all__ = ["TransferLedger", "row_bytes"]
