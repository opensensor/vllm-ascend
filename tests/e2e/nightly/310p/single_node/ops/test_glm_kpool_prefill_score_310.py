# SPDX-License-Identifier: Apache-2.0
"""Hardware qualification prepared offline; requires the separate prefill OPP."""

import pytest
import torch

from tools.glm_perf.kpool_prefill import select_prefill_request
from vllm_ascend.models.glm5next.kpool_ops import score_and_select_kpool_tokens


@pytest.fixture(autouse=True, scope="module")
def require_310p():
    torch_npu = pytest.importorskip("torch_npu")
    if not torch_npu.npu.is_available() or "310" not in torch_npu.npu.get_device_name(0):
        pytest.skip("requires Ascend 310P")
    torch.npu.set_device(0)
    torch_npu.npu.set_compile_mode(jit_compile=False)


def inputs(rows, pools, complete=None):
    torch.manual_seed(310)
    br, rs, offset = 160, 144, 16
    columns = (pools + br - 1) // br
    bs = br * rs + 256
    blocks = columns + 1
    storage = torch.full((offset + blocks * bs,), torch.nan, dtype=torch.float16)
    cache = storage.as_strided((blocks, br, 128), (bs, rs, 1), offset)
    cache[:-1].copy_(torch.randn(cache[:-1].shape).bfloat16().half())
    table = torch.arange(columns, dtype=torch.int32).flip(0).reshape(1, -1)
    q = torch.randn(rows, 32, 128).bfloat16().half()
    w = torch.randn(rows, 32)
    lengths = complete if complete is not None else [max(0, pools - rows + i) for i in range(rows)]
    pos = torch.tensor([x * 4 - 1 for x in lengths], dtype=torch.int32)
    ends = torch.tensor([rows], dtype=torch.int32)
    return [x.npu() for x in (q, w, storage, table, ends, pos)] + [pools, blocks, br, bs, rs, offset]


def reference(args):
    q, w, storage, table, _, pos = [x.cpu() for x in args[:6]]
    pools, blocks, br, bs, rs, off = args[6:]
    cache = storage.as_strided((blocks, br, 128), (bs, rs, 1), off)
    output = torch.full((q.shape[0], pools), -torch.inf)
    for row in range(q.shape[0]):
        complete = max(0, min((int(pos[row]) + 1) // 4, pools))
        if not complete:
            continue
        ids = torch.arange(complete)
        pages = table[0, ids // br].long()
        valid = (pages >= 0) & (pages < blocks)
        keys = cache[pages.clamp(0, blocks - 1), ids % br]
        scores = ((q[row].float() @ keys.float().T).relu() * w[row, :, None]).sum(0)
        output[row, :complete] = torch.where(valid, scores, -torch.inf)
    return output


@pytest.mark.parametrize(
    "rows,pools,complete",
    [
        (4, 168, [0, 63, 65, 161]),
        (4, 168, [159, 160, 161, 168]),
        (8, 8192, [0, 1, 63, 64, 159, 160, 161, 8192]),
        (128, 2048, None),
        (4, 77760, [0, 4000, 4160, 77760]),
    ],
)
def test_native_score_parity(rows, pools, complete):
    args = inputs(rows, pools, complete)
    saved = [x.cpu().clone() for x in args[:6]]
    actual = torch.ops._C_ascend.npu_glm_kpool_prefill_score_310(*args)
    torch.testing.assert_close(actual.cpu(), reference(args), rtol=3e-5, atol=1e-4)
    for value, old in zip(args[:6], saved, strict=True):
        torch.testing.assert_close(value.cpu(), old, rtol=0, atol=0, equal_nan=True)


def test_invalid_pages_and_padded_rows():
    args = inputs(4, 8192, [8192, 65, 0, 0])
    args[3][0, 0] = -1
    args[3][0, 1] = args[7]
    actual = torch.ops._C_ascend.npu_glm_kpool_prefill_score_310(*args)
    torch.testing.assert_close(actual.cpu(), reference(args), rtol=3e-5, atol=1e-4)


@pytest.mark.parametrize("rows,ties", [(5, False), (130, False), (640, False), (130, True)])
def test_complete_prefill_selection(rows, ties):
    args = inputs(rows, 2048)
    q, w, storage, table, _, pos = args[:6]
    pools, blocks, br, bs, rs, off = args[6:]
    if ties:
        q.zero_()
    cache = storage.as_strided((blocks, br, 1, 128), (bs, rs, rs, 1), off)
    ids = torch.arange(pools, device=q.device)
    keys = cache[table[0, ids // br].long(), ids % br, 0]
    expected = score_and_select_kpool_tokens(q, w, keys, pos, 2048, 4)
    actual = select_prefill_request(q, w, cache, table, pos, pools, 2048, 4)
    # Selection is a stronger gate than score tolerance: no changed key set
    # is silently accepted as parity. Tie order is checked separately.
    torch.testing.assert_close(actual.sort().values, expected.sort().values, rtol=0, atol=0)
    if ties:
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("bad", ["rows", "requests", "short_table"])
def test_bad_native_geometry_rejected(bad):
    args = inputs(4, 168)
    if bad == "rows":
        args[0], args[1], args[5] = args[0][:3], args[1][:3], args[5][:3]
    elif bad == "requests":
        args[3] = args[3].repeat(2, 1)
        args[4] = args[4].repeat(2)
    else:
        args[3] = args[3][:, :1].contiguous()
    with pytest.raises((RuntimeError, ValueError)):
        torch.ops._C_ascend.npu_glm_kpool_prefill_score_310(*args)
