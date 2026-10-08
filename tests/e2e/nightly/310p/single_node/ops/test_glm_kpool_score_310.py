# SPDX-License-Identifier: Apache-2.0
"""Isolated hardware gate for the opt-in live-length GLM paged scorer."""

import pytest
import torch
import torch_npu

from vllm_ascend.models.glm5next.kpool_ops import expand_kpool_groups, hadamard128, select_kpool_groups
from vllm_ascend.models.glm5next.ops.kpool_native import score_kpool_paged


@pytest.fixture(autouse=True, scope="module")
def require_310p():
    if not torch_npu.npu.is_available() or "310" not in torch_npu.npu.get_device_name(0):
        pytest.skip("requires Ascend 310P")
    torch.npu.set_device(0)
    torch_npu.npu.set_compile_mode(jit_compile=False)


def inputs(rows=2, pools=8192, complete=None, block_rows=160, offset=16):
    torch.manual_seed(310)
    columns = (pools + block_rows - 1) // block_rows
    blocks = columns * 2 + 1
    block_stride = block_rows * 128 + 256
    storage = torch.full((offset + blocks * block_stride,), float("nan"), dtype=torch.float16)
    cache = storage.as_strided((blocks, block_rows, 1, 128), (block_stride, 128, 128, 1), offset)
    table = torch.arange(columns * 2, dtype=torch.int32).reshape(2, columns).flip(1).contiguous()
    cache.copy_(torch.randn(cache.shape).bfloat16().float().half())
    # Leave the fallback page poisoned: it must not be read for unused capacity.
    cache[-1].fill_(float("nan"))
    q = torch.randn(rows, 32, 128).bfloat16().float().half()
    w = torch.randn(rows, 32)
    ends = torch.tensor([rows // 2, rows], dtype=torch.int32)
    lengths = complete if complete is not None else [33 + 641 * i for i in range(rows)]
    pos = torch.tensor([n * 4 - 1 for n in lengths], dtype=torch.int32)
    args = [x.npu() for x in (q, w, storage, table, ends, pos)]
    args += [pools, blocks, block_rows, block_stride, 128, offset]
    return args


def reference(args, device="cpu"):
    q, w, storage, table, ends, pos = [x.to(device) for x in args[:6]]
    pools, blocks, br, bs, rs, off = args[6:]
    cache = storage.as_strided((blocks, br, 128), (bs, rs, 1), off)
    output = torch.full((q.shape[0], pools), -float("inf"), device=device)
    for row in range(q.shape[0]):
        request = min(int(torch.searchsorted(ends, torch.tensor(row, device=device), right=True)), table.shape[0] - 1)
        complete = max(0, min((int(pos[row]) + 1) // 4, pools))
        if not complete:
            continue
        ids = torch.arange(complete, device=device)
        pages = table[request, ids // br].long()
        valid = (pages >= 0) & (pages < blocks)
        keys = cache[pages.clamp(0, blocks - 1), ids % br]
        score = ((q[row].float() @ keys.float().T).relu() * w[row, :, None]).sum(0)
        output[row, :complete] = torch.where(valid, score, -float("inf"))
    return output


def op(args):
    return torch.ops._C_ascend.npu_glm_kpool_score_310(*args)


def parity(actual, expected):
    torch.testing.assert_close(actual.cpu(), expected.cpu(), rtol=3e-5, atol=1e-4)


@pytest.mark.parametrize(
    "rows,pools,complete",
    [
        (1, 8192, [0]),
        (2, 8192, [1, 65]),
        (2, 8192, [159, 161]),
        (8, 8192, [0, 1, 63, 64, 65, 511, 512, 8192]),
        (2, 77760, [33, 674]),
        (2, 77760, [32768, 77760]),
    ],
)
def test_score_parity(rows, pools, complete):
    args = inputs(rows, pools, complete)
    saved = [v.cpu().clone() for v in args[:6]]
    parity(op(args), reference(args))
    for value, old in zip(args[:6], saved, strict=True):
        torch.testing.assert_close(value.cpu(), old, rtol=0, atol=0, equal_nan=True)


def test_invalid_pages_and_negative_padding():
    args = inputs(2, complete=[1025, 200])
    args[3][0, 0] = -1
    args[3][0, 1] = args[7]
    args[5][1] = -1
    actual, expected = op(args).cpu(), reference(args)
    bad = ~(torch.isclose(actual, expected, rtol=3e-5, atol=1e-4))
    if bad.any():
        print("bad indices/actual/expected", bad.nonzero(), actual[bad], expected[bad], flush=True)
    parity(actual, expected)


def test_exact_ties_and_topk():
    args = inputs(2, complete=[511, 2048])
    args[0].fill_(1)
    args[1].fill_(1)
    args[2].fill_(1)
    actual, expected = op(args), reference(args).npu()
    parity(actual, expected)
    torch.testing.assert_close(actual.topk(512).indices, expected.topk(512).indices, rtol=0, atol=0)


def test_graph_changes_lengths_pages_queries_and_weights():
    args = inputs(2, pools=32768, complete=[1, 65])
    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(stream):
        for _ in range(3):
            op(args)
    torch.npu.current_stream().wait_stream(stream)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, stream=stream):
        output = op(args)
    for lengths in ([2048, 3000], [32, 0], [32768, 160]):
        updated = inputs(2, pools=32768, complete=lengths)
        updated[0].mul_(0.5)
        updated[1].neg_()
        updated[3] = updated[3].flip(1).contiguous()
        updated[4][0] = 2  # both rows belong to the first request on this replay
        for target, source in zip(args[:6], updated[:6], strict=True):
            target.copy_(source)
        graph.replay()
        parity(output, reference(args))


@pytest.mark.parametrize("bad", ["dtype", "stride", "offset", "pools", "bounds", "rows"])
def test_invalid_geometry(bad):
    args = inputs()
    if bad == "dtype":
        args[1] = args[1].half()
    elif bad == "stride":
        args[9] = 32
    elif bad == "offset":
        args[11] = args[2].numel()
    elif bad == "pools":
        args[6] = 8193
    elif bad == "bounds":
        args[4] = args[4][:1]
    else:
        args[0] = args[0].repeat(5, 1, 1)
    with pytest.raises(RuntimeError):
        op(args)


def test_serving_wrapper_rotation_and_graph_capture():
    args = inputs(2, pools=32768, complete=[2048, 8192])
    q, weights, storage, table, ends, positions = args[:6]
    pools, blocks, br, bs, rs, offset = args[6:]
    cache = storage.as_strided((blocks, br, 1, 128), (bs, rs, rs, 1), offset)

    def run():
        return score_kpool_paged(q, weights, cache, table, ends, positions.long(), pools)

    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(stream):
        for _ in range(3):
            run()
    torch.npu.current_stream().wait_stream(stream)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, stream=stream):
        output = run()
    for factor in (0.5, -2.0):
        q.mul_(factor)
        graph.replay()
        rotated = hadamard128(q.cpu()).bfloat16().float()
        ref_args = [rotated, weights, storage, table, ends, positions, *args[6:]]
        expected = reference(ref_args)
        parity(output, expected)
        for row in range(2):
            actual_ids = select_kpool_groups(output[row : row + 1], positions[row : row + 1], 2048, 4)[0].cpu()
            expected_ids = select_kpool_groups(expected[row : row + 1], positions[row : row + 1].cpu(), 2048, 4)[0]
            torch.testing.assert_close(actual_ids.sort().values, expected_ids.sort().values, rtol=0, atol=0)


@pytest.mark.parametrize("pools", [8192, 32768, 77760])
def test_large_topk_padding_never_duplicates_valid_pools(pools):
    logits = torch.zeros(1, pools, device="npu")
    logits[:, :64] = torch.arange(64, device="npu", dtype=torch.float32)
    positions = torch.tensor([255], dtype=torch.int32, device="npu")
    for _ in range(3):
        selected, _, starts, counts = select_kpool_groups(logits, positions, 2048, 4)
        cpu = selected.cpu()
        assert int((cpu >= 0).sum()) == 64
        assert (cpu[:, 64:] == -1).all()
        torch.testing.assert_close(cpu[0, :64], torch.arange(63, -1, -1, dtype=torch.int32))
        expanded = expand_kpool_groups(selected, starts, counts, 4).cpu()
        assert int((expanded >= 0).sum()) == 256
        assert expanded[expanded >= 0].unique().numel() == 256
