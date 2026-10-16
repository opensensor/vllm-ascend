# SPDX-License-Identifier: Apache-2.0
"""Private real-weight PLE gate; never installs a serving method."""

from pathlib import Path

from tools.qwen4exp import tp6_finalize_gate
from tools.qwen4exp.tp6_finalize_gate import measurements, worker_snapshot


def run_gate(worker, config, resource_source, owner):
    import inspect
    from functools import partial

    import torch

    from vllm_ascend.models.qwen4_exp.model import _PLEInjection

    before = worker_snapshot(worker)
    manager = inspect.getclosurevars(_PLEInjection._forward_eager).nonlocals["manager"]
    layers = [
        layer
        for wrapper in worker._resident_wrappers()
        for layer in wrapper.runnable.modules()
        if isinstance(layer, _PLEInjection)
    ]
    assert len(layers) == 1 and not manager.failed and not manager.busy
    layer = layers[0]
    method = layer.ple.ple_method
    ns = {}
    exec(compile(resource_source, "<shared-ple-gather>", "exec"), ns)
    owner["gather"] = gather = ns["SharedPLEGather"](method, manager.metadata)
    owner["method"] = method
    worker._qfn_shared_gather_gate = owner
    results = []
    owner["results"] = results
    generator = torch.Generator().manual_seed(610160 + worker.rank)
    for count in (0, 1, 3, 18, 65, 131, 134, 256, 640, 2048):
        rows = torch.randint(method.num_embeddings, (count * 16,), generator=generator)
        if count:
            rows[: min(3, rows.numel())] = torch.tensor([0, method.num_embeddings - 1, 0])[: min(3, rows.numel())]
        expected = method.gather_rows(rows)
        actual = gather.gather_rows(rows)
        assert torch.equal(expected.view(torch.uint8), actual.view(torch.uint8))
        results.append(dict(kind="rows", tokens=count, exact=True))
    present = "gather_rows" in vars(method)
    previous = vars(method).get("gather_rows")

    def call(shared, hidden, tokens, starts, history):
        if shared:
            method.gather_rows = gather.gather_rows
        try:
            return layer._forward_eager(
                hidden,
                tokens.to(hidden.device),
                starts.to(hidden.device),
                history.to(hidden.device),
                input_ids_cpu=tokens,
                query_start_loc_cpu=starts,
                ngram_context_cpu=history,
            )
        finally:
            if present:
                method.gather_rows = previous
            elif "gather_rows" in vars(method):
                delattr(method, "gather_rows")

    with torch.inference_mode():
        for count in (65, 131, 134, 256, 640, 2048):
            tokens = torch.randint(manager.metadata["geometry"]["vocabulary"], (count,), generator=generator)
            tokens[3] = manager.metadata["geometry"]["eos"]
            starts = torch.tensor([0, 3, count], dtype=torch.int32)
            history = torch.randint(manager.metadata["geometry"]["vocabulary"], (2, 2), generator=generator)
            hidden = torch.randn((count, layer.ple.hc_hidden_size), generator=generator).to(
                device=worker.model_runner.device
            )
            operations = dict(
                retained=partial(call, False, hidden, tokens, starts, history),
                shared=partial(call, True, hidden, tokens, starts, history),
            )
            expected = operations["retained"]()
            actual = operations["shared"]()
            torch.npu.synchronize()
            assert torch.equal(expected.cpu().view(torch.uint8), actual.cpu().view(torch.uint8))
            result = dict(kind="complete_ple", tokens=count, exact=True, timing=measurements(operations))
            results.append(result)
    after = worker_snapshot(worker)
    return dict(rank=worker.rank, passed=before == after, before=before, after=after, results=results, installed=False)


def engine_source():
    # Bundle the gate for frozen runtimes without altering their import tree.
    import ast

    source = Path(__file__).read_text()
    node = next(n for n in ast.parse(source).body if getattr(n, "name", None) == "run_gate")
    gate = Path(tp6_finalize_gate.__file__).read_text() + "\n" + ast.get_source_segment(source, node)
    resource = Path(__file__).with_name("ple_shared_gather.py").read_text()
    return f"CONFIG = {{}}\nRESOURCE_SOURCE = {resource!r}\nGATE_SOURCE = {gate!r}\n" + tp6_finalize_gate.ENGINE_SOURCE
