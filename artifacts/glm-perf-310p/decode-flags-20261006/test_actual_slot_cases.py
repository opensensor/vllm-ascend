# SPDX-License-Identifier: Apache-2.0
# Extracted unchanged definitions; isolate CPU helper tests from plugin imports.
import torch
import pytest
PADDING_SLOT_ID = -1

def compute_packed_draft_slots(block_table, query_start_loc, positions, block_size, row_indices):
    """Map draft rows on device using indices allocated before graph capture."""
    rows = row_indices[:positions.shape[0]].to(dtype=query_start_loc.dtype)
    requests = torch.searchsorted(query_start_loc[1:], rows, right=True).long()
    safe_requests = requests.clamp(max=block_table.shape[0] - 1)
    columns = torch.div(positions, block_size, rounding_mode='floor').long()
    safe_columns = columns.clamp(min=0, max=block_table.shape[1] - 1)
    pages = block_table[safe_requests, safe_columns].to(torch.int32)
    valid = (requests < block_table.shape[0]) & (positions >= 0) & (columns < block_table.shape[1]) & (pages >= 0)
    slots = pages * block_size + positions.to(torch.int32).remainder(block_size)
    return torch.where(valid, slots, PADDING_SLOT_ID)

def _reference_draft_slots(block_table, query_start_loc, positions, block_size):
    """Straightforward host reference for the device slot computation."""
    num_reqs = query_start_loc.shape[0] - 1
    out = []
    for token, position in enumerate(positions.tolist()):
        request = None
        for req in range(num_reqs):
            if query_start_loc[req] <= token < query_start_loc[req + 1]:
                request = req
                break
        if request is None or position < 0:
            out.append(PADDING_SLOT_ID)
            continue
        column = position // block_size
        if column < 0 or column >= block_table.shape[1]:
            out.append(PADDING_SLOT_ID)
            continue
        page = int(block_table[request, column])
        if page < 0:
            out.append(PADDING_SLOT_ID)
            continue
        out.append(page * block_size + position % block_size)
    return out

@pytest.mark.parametrize('block_table,query_start_loc,positions,block_size', [([[3, 4]], [0, 2], [0, 639], 640), ([[3, 4, 5], [10, 11, 12]], [0, 3, 5], [0, 639, 640, 0, 1280], 640), ([[3, 4], [10, 11]], [0, 3, 5], [0, -1, -1, 0, 0], 640), ([[3, 4]], [0, 2], [0, 640 * 8], 640), ([[3, -1]], [0, 2], [0, 640], 640), (torch.arange(1, 9721).view(1, -1), [0, 1], [311039], 640)])
def test_compute_packed_draft_slots_matches_reference(block_table, query_start_loc, positions, block_size):
    block_table = torch.tensor(block_table, dtype=torch.int32)
    query_start_loc = torch.tensor(query_start_loc, dtype=torch.int32)
    positions = torch.tensor(positions, dtype=torch.int64)
    row_indices = torch.arange(max(positions.shape[0], 1) + 8, dtype=torch.int32)
    actual = compute_packed_draft_slots(block_table, query_start_loc, positions, block_size, row_indices).tolist()
    expected = _reference_draft_slots(block_table, query_start_loc, positions, block_size)
    assert actual == expected

def test_compute_packed_draft_slots_output_length_tracks_positions():
    block_table = torch.tensor([[3, 4]], dtype=torch.int32)
    query_start_loc = torch.tensor([0, 3], dtype=torch.int32)
    positions = torch.tensor([0, 1, 2], dtype=torch.int64)
    row_indices = torch.arange(8, dtype=torch.int32)
    result = compute_packed_draft_slots(block_table, query_start_loc, positions, 640, row_indices)
    assert result.shape == (positions.shape[0],)
    assert result.dtype == torch.int32
