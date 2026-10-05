# SPDX-License-Identifier: Apache-2.0

import ast
import dataclasses
import gc
import importlib.util
import sys
import textwrap
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

from tools.glm_perf.resident_candidates.indexer_projection import (
    ProjectionBank,
    make_capture,
    prepare_runner,
    rewrite_forward,
)
from tools.glm_perf.resident_control import Control, PatchSession

ROOT = Path(__file__).resolve().parents[3]


class FakeIndexer(torch.nn.Module):
    def __init__(self, head_dim=8, heads=2, hidden=16):
        super().__init__()
        self.head_dim, self.n_head, self.rope_dim = head_dim, heads, 0
        self._wk_weight_f32 = torch.randint(-2, 3, (head_dim + heads, hidden)).float()
        self._gate_weight_f32 = torch.randint(-2, 3, (head_dim, hidden)).float()
        self.k_norm = torch.nn.LayerNorm(head_dim)
        self.softmax_scale = head_dim**-0.5
        self.index_kpool, self.index_kpool_compress_ape = 4, torch.zeros(4, head_dim)
        self.wq_b = lambda value: (value[:, : head_dim * heads], None)
        self.indexer_op = lambda *args, **kwargs: (args, kwargs)


def test_prepare_is_bounded_and_reuses_storage_without_replacing_weights():
    indexer = FakeIndexer()
    key, gate = indexer._wk_weight_f32, indexer._gate_weight_f32
    bank = ProjectionBank()
    bank.prepare([indexer, indexer])
    entry = bank.entries[indexer]
    assert bank.prepared_bytes == (key.numel() + gate.numel()) * 4
    assert entry.packed.data_ptr() not in (key.data_ptr(), gate.data_ptr())
    bank.prepare([indexer])
    assert bank.entries[indexer].packed is entry.packed
    assert indexer._wk_weight_f32 is key and indexer._gate_weight_f32 is gate


def test_over_budget_rejected_before_any_allocation(monkeypatch):
    indexer = FakeIndexer()
    bank = ProjectionBank(max_bytes=1)
    monkeypatch.setattr(torch, "cat", lambda *args, **kwargs: pytest.fail("allocated before admission"))
    with pytest.raises(ValueError, match="budget"):
        bank.prepare([indexer])
    assert len(bank.entries) == 0 and bank.prepared_bytes == 0


@pytest.mark.parametrize("invalid", ["missing", "dtype", "shape", "strided"])
def test_invalid_cached_weights_rejected(invalid):
    indexer = FakeIndexer()
    if invalid == "missing":
        indexer._wk_weight_f32 = None
    elif invalid == "dtype":
        indexer._gate_weight_f32 = indexer._gate_weight_f32.half()
    elif invalid == "shape":
        indexer._gate_weight_f32 = indexer._gate_weight_f32[:1]
    else:
        indexer._gate_weight_f32 = torch.ones(8, 32)[:, ::2]
    with pytest.raises(ValueError, match="prepared contiguous"):
        ProjectionBank().prepare([indexer])


def test_missing_or_replaced_weight_refuses_forward_and_reprepare():
    indexer = FakeIndexer()
    bank = ProjectionBank()
    with pytest.raises(RuntimeError, match="before graph capture"):
        bank.project(indexer, torch.ones(2, 16))
    bank.prepare([indexer])
    indexer._gate_weight_f32 = indexer._gate_weight_f32.clone()
    with pytest.raises(RuntimeError, match="before graph capture"):
        bank.project(indexer, torch.ones(2, 16))
    with pytest.raises(ValueError, match="fresh candidate"):
        bank.prepare([indexer])


def test_bank_does_not_keep_indexer_alive():
    bank = ProjectionBank()
    indexer = FakeIndexer()
    bank.prepare([indexer])
    del indexer
    gc.collect()
    assert len(bank.entries) == 0


def test_target_and_draft_prepared_before_capture_and_repeated_capture_not_nested():
    target, draft = FakeIndexer(), FakeIndexer()
    runner = SimpleNamespace(
        model=SimpleNamespace(runnable=torch.nn.ModuleList([target])),
        drafter=SimpleNamespace(model=SimpleNamespace(runnable=torch.nn.ModuleList([draft]))),
    )
    bank = ProjectionBank()
    calls = []

    def capture(runner, marker):
        assert target in bank.entries and draft in bank.entries
        calls.append(marker)
        return 123

    wrapped = make_capture(capture, bank, FakeIndexer)
    assert wrapped(runner, "first") == 123
    addresses = [bank.entries[indexer].packed.data_ptr() for indexer in (target, draft)]
    # A mode change or repeated prepare must wrap the original capture, not
    # allocate/prepare a second bank through a nested previous replacement.
    second = make_capture(wrapped, bank, FakeIndexer)
    assert second.__glm_resident_original__ is capture
    assert second(runner, "second") == 123
    assert calls == ["first", "second"]
    assert addresses == [bank.entries[indexer].packed.data_ptr() for indexer in (target, draft)]


def test_invalid_draft_blocks_capture_before_target_pack(monkeypatch):
    target, draft = FakeIndexer(), FakeIndexer()
    draft._wk_weight_f32 = None
    runner = SimpleNamespace(
        model=torch.nn.ModuleList([target]), drafter=SimpleNamespace(model=torch.nn.ModuleList([draft]))
    )
    monkeypatch.setattr(torch, "cat", lambda *args, **kwargs: pytest.fail("allocated before validating draft"))
    capture = make_capture(lambda *args: pytest.fail("capture ran with invalid draft"), ProjectionBank(), FakeIndexer)
    with pytest.raises(ValueError, match="prepared contiguous"):
        capture(runner)


def test_no_indexer_fails_instead_of_benchmarking_inactive_candidate():
    with pytest.raises(ValueError, match="no GLM indexers"):
        prepare_runner(ProjectionBank(), SimpleNamespace(model=torch.nn.Module()), FakeIndexer)


def forward_source():
    source = (ROOT / "vllm_ascend/models/glm5next/attention.py").read_text()
    tree = ast.parse(source)
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Indexer")
    function = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "forward")
    return ast.get_source_segment(source, function)


@pytest.mark.parametrize("rows", [2, 8, 640])
@pytest.mark.parametrize("rope_dim", [0, 4])
def test_actual_forward_preserves_rope_norm_and_indexer_arguments(rows, rope_dim):
    indexer = FakeIndexer()
    indexer.rope_dim = rope_dim
    bank = ProjectionBank()
    bank.prepare([indexer])
    scope = {"torch": torch}
    exec(compile(ast.parse(forward_source()), "<baseline>", "exec"), scope)
    baseline = scope["forward"]
    scope = {"torch": torch, "_candidate_projection": bank.project}
    exec(compile(rewrite_forward(forward_source()), "<candidate>", "exec"), scope)
    candidate = scope["forward"]
    # Integer-valued small inputs make the CPU layout gate exact. Random model
    # weights and NPU GEMM accumulation still require their separate gate.
    hidden = torch.randint(-2, 3, (rows, 16)).float()
    query = hidden.clone()
    positions = torch.arange(rows)

    def rope(positions, q, k):
        return q.flip(-1), k.flip(-1)

    expected_args, expected_kwargs = baseline(indexer, hidden, query, positions, rope)
    actual_args, actual_kwargs = candidate(indexer, hidden, query, positions, rope)
    assert len(actual_args) == len(expected_args) and actual_kwargs.keys() == expected_kwargs.keys()
    for a, b in [
        *zip(actual_args, expected_args),
        *((actual_kwargs[key], expected_kwargs[key]) for key in actual_kwargs),
    ]:
        assert torch.equal(a, b) if isinstance(a, torch.Tensor) else a == b


def test_forward_has_one_mm_and_no_packing(monkeypatch):
    indexer = FakeIndexer()
    bank = ProjectionBank()
    bank.prepare([indexer])
    calls = []
    original_mm = torch.mm

    def mm(*args, **kwargs):
        calls.append(args[1].shape)
        return original_mm(*args, **kwargs)

    monkeypatch.setattr(torch, "mm", mm)
    monkeypatch.setattr(torch, "cat", lambda *args: pytest.fail("weight packing in forward"))
    key, gate = bank.project(indexer, torch.ones(8, 16))
    assert calls == [torch.Size([16, 18])]
    assert key.shape == (8, 10) and gate.shape == (8, 8)
    assert key.untyped_storage().data_ptr() == gate.untyped_storage().data_ptr()


def test_rewrite_changes_only_two_projection_assignments():
    original = ast.parse(forward_source())
    rewritten = rewrite_forward(forward_source())
    function = rewritten.body[0]
    pos = next(
        i
        for i, node in enumerate(function.body)
        if isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Tuple)
        and [item.id for item in node.targets[0].elts] == ["kw", "gate_score"]
    )
    function.body[pos] = ast.parse("kw = torch.mm(hidden_f32, self._wk_weight_f32.t())").body[0]
    original_body = original.body[0].body
    gate_pos = next(
        i
        for i, node in enumerate(original_body)
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) and node.targets[0].id == "gate_score"
    )
    function.body.insert(gate_pos, original_body[gate_pos])
    assert ast.dump(original) == ast.dump(rewritten)


def test_changed_projection_source_rejected():
    with pytest.raises(ValueError, match="re-audit"):
        rewrite_forward(forward_source().replace("kw = torch.mm", "kw = torch.matmul"))


def test_actual_resident_loader_reapply_and_rollback(monkeypatch, tmp_path):
    # Use a file-backed copy so inspect sees the real current Indexer.forward,
    # while dependency imports resolve to CPU fixtures rather than NPU runtime.
    path = tmp_path / "fixture.py"
    path.write_text("import torch\nclass Indexer(FakeIndexer):\n" + textwrap.indent(forward_source(), "    ") + "\n")
    name = "vllm_ascend.models.glm5next.attention"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    module.FakeIndexer = FakeIndexer
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    runner_name = "vllm_ascend._310p.model_runner_310p"
    runner_module = ModuleType(runner_name)

    class Runner:
        def capture_model(self):
            self.captures += 1
            indexer = self.model.runnable[0]
            return indexer(torch.ones(2, 16), torch.ones(2, 16), torch.arange(2), None)

    runner_module.NPUModelRunner310 = Runner
    monkeypatch.setitem(sys.modules, runner_name, runner_module)
    target, draft = module.Indexer(), module.Indexer()
    runner = Runner()
    runner.captures = 0
    runner.model = SimpleNamespace(runnable=torch.nn.ModuleList([target]))
    runner.drafter = SimpleNamespace(model=SimpleNamespace(runnable=torch.nn.ModuleList([draft])))
    source = (ROOT / "tools/glm_perf/resident_candidates/indexer_projection.py").read_text()
    baseline_forward, baseline_capture = module.Indexer.forward, Runner.capture_model
    pointers = [(m._wk_weight_f32.data_ptr(), m._gate_weight_f32.data_ptr()) for m in (target, draft)]
    session = PatchSession()
    candidate = Control("1" * 32, candidate="indexer_projection", source=source)
    try:
        session.prepare(dataclasses.asdict(candidate))
        assert module.Indexer.forward is baseline_forward
        session.apply(candidate.generation)
        with pytest.raises(RuntimeError, match="prepare target and draft"):
            target(torch.ones(2, 16), torch.ones(2, 16), torch.arange(2), None)
        runner.capture_model()
        assert runner.captures == 1
        # A fresh source generation prepares while the old methods are active.
        # It must recover the originals rather than wrapping the previous bank.
        second = Control("2" * 32, candidate="indexer_projection", source=source + "\n# next generation\n")
        session.prepare(dataclasses.asdict(second))
        session.apply(second.generation)
        runner.capture_model()
        assert runner.captures == 2
        assert pointers == [(m._wk_weight_f32.data_ptr(), m._gate_weight_f32.data_ptr()) for m in (target, draft)]
    finally:
        baseline = Control("3" * 32)
        session.prepare(dataclasses.asdict(baseline))
        session.apply(baseline.generation)
    assert module.Indexer.forward is baseline_forward and Runner.capture_model is baseline_capture
