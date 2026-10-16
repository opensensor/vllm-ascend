# SPDX-License-Identifier: Apache-2.0
"""Private eager experiment reusing existing fixed collective owners.

This does not create peer buffers or install a serving dispatch. Run only after
all workers and graph replays have drained, on their claimed collective stream.
"""

HIDDEN_SIZE = 2560


def chunk_plan(tokens, sizes):
    if type(tokens) is not int or not 1 <= tokens <= 2048:
        raise ValueError("bounded positive token count required")
    sizes = tuple(sizes)
    if not sizes or any(type(size) is not int or size < 1 for size in sizes):
        raise ValueError("positive integer bucket sizes required")
    sizes = tuple(sorted(set(sizes), reverse=True))
    best = [()] + [None] * tokens
    for total in range(1, tokens + 1):
        choices = [best[total - size] + (size,) for size in sizes if size <= total and best[total - size] is not None]
        if choices:
            best[total] = min(choices, key=len)
    if best[tokens] is None:
        raise ValueError("existing buckets cannot exactly cover this geometry")
    return best[tokens]


def reduce_chunks(live, tensor, plan):
    import torch

    if tensor.numel() != sum(plan) * HIDDEN_SIZE or not tensor.is_contiguous():
        raise ValueError("chunk plan differs from input geometry")
    dtype = str(tensor.dtype).removeprefix("torch.")
    if any((dtype, size * HIDDEN_SIZE) not in live.owners for size in plan):
        raise ValueError("every chunk must have an existing owner")

    def no_fallback(value):
        raise RuntimeError("qualified chunk unexpectedly selected fallback")

    flat = tensor.view(-1)
    outputs, offset = [], 0
    for size in plan:
        end = offset + size * HIDDEN_SIZE
        outputs.append(live.all_reduce(flat[offset:end], no_fallback))
        offset = end
    return torch.cat(outputs).view_as(tensor)


def run_gate(worker, config, resource_source, owner):
    # Worker-only imports: no context, group, or native allocation is created.
    from types import SimpleNamespace

    from vllm.distributed import get_tp_group

    from tools.qwen4exp.tp6_finalize_gate import worker_snapshot
    from tools.sidecard.qfn_prefill_collective_probe import _checked, _exchange, _exercise

    group = get_tp_group()
    live = group.device_communicator.custom_sum
    if tuple(live.members) != tuple(range(6)) or live.state != "active" or live.claimed_stream is None:
        raise RuntimeError("requires the active six-rank collective and its claimed stream")
    owner.update(tensors=[], events=[], completed=[])
    before = worker_snapshot(worker)
    geometry_owners = {key: id(value) for key, value in live.owners.items()}
    torch = live.torch
    with torch.npu.stream(live.claimed_stream):
        _checked(live, owner, "context", live._context)
        _checked(live, owner, "stream", live._stream)
        _checked(live, owner, "initial-drain", torch.npu.synchronize)
        for count in (131, 134):
            for dtype in ("float16", "float32"):
                sizes = [elements // HIDDEN_SIZE for kind, elements in live.owners if kind == dtype]
                plan = chunk_plan(count, sizes)
                private = SimpleNamespace(
                    all_reduce=lambda tensor, fallback, plan=plan: reduce_chunks(live, tensor, plan)
                )
                geometry = dict(dtype=dtype, elements=count * HIDDEN_SIZE, rows=count)
                result = _exercise(live, group, private, owner, geometry)
                # Reduction order may differ from HCCL on general finite values;
                # dyadic exact cases alone cannot qualify a model replacement.
                generator = torch.Generator().manual_seed(610160 + worker.rank)
                source = torch.randn(geometry["elements"], generator=generator).to(
                    device=worker.model_runner.device, dtype=getattr(torch, dtype)
                )
                expected = source.clone()
                _checked(live, owner, "random-ready", torch.npu.synchronize)
                _exchange(live, "random-start", None)
                actual = private.all_reduce(source, None)
                live.dist.all_reduce(expected, group=group.device_group)
                torch.npu.synchronize()
                a, b = actual.cpu(), expected.cpu()
                result.update(
                    plan=plan,
                    random_exact=bool(torch.equal(a, b)),
                    random_unequal=int((a != b).sum()),
                    random_max_abs=float((a.float() - b.float()).abs().max()),
                )
                owner["completed"].append(result)
        torch.npu.synchronize()
    after = worker_snapshot(worker)
    return dict(
        rank=worker.rank,
        passed=before == after and geometry_owners == {key: id(value) for key, value in live.owners.items()},
        before=before,
        after=after,
        results=owner["completed"],
        installed=False,
        new_peer_buffers=False,
    )
