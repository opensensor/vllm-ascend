# SPDX-License-Identifier: Apache-2.0
"""Independent vocabulary ownership and rejection of incompatible shards."""

import ast
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from vllm_ascend._310p import qwen4exp_draft_vocab as vocab
from vllm_ascend._310p.qwen4exp_mtp import is_qwen4exp_speculative_target


def layer(rows=9, start=0, end=7, tp=1):
    result = torch.nn.Embedding(rows, 4, dtype=torch.float16)
    result.tp_size = tp
    result.shard_indices = SimpleNamespace(org_vocab_start_index=start, org_vocab_end_index=end)
    return result


def checkpoint(path, values, names=("lm_head.weight",)):
    save_file({name: values.clone() for name in names}, path / "float.safetensors")
    (path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": dict.fromkeys(names, "float.safetensors")})
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_complete_vocab_chunk_copy_and_zero_padding(tmp_path, monkeypatch, dtype):
    values = torch.arange(28).reshape(7, 4).to(dtype) / 8
    checkpoint(tmp_path, values)
    monkeypatch.setattr(vocab, "VOCAB_COPY_ROWS", 2)
    target = layer()
    receipt = vocab.load_full_vocab_weight(target, tmp_path, ("lm_head.weight",), 7, 4)
    torch.testing.assert_close(target.weight[:7], values.half(), rtol=0, atol=0)
    assert torch.count_nonzero(target.weight[7:]) == 0
    assert receipt["tensor"] == "lm_head.weight"


@pytest.mark.parametrize("reason", ["shard", "tp", "width", "dtype", "missing", "ambiguous", "source", "nan"])
def test_reject_invalid_vocab(tmp_path, reason):
    values = torch.arange(28).reshape(7, 4).float()
    names = ("lm_head.weight", "alias.weight") if reason == "ambiguous" else ("lm_head.weight",)
    if reason == "source":
        values = values.to(torch.int8)
    if reason == "nan":
        values[0, 0] = float("nan")
    checkpoint(tmp_path, values, names)
    target = layer()
    if reason == "shard":
        target.shard_indices.org_vocab_start_index = 1
    elif reason == "tp":
        target.tp_size = 6
    elif reason == "width":
        target.weight = torch.nn.Parameter(torch.zeros(9, 3, dtype=torch.float16))
    elif reason == "dtype":
        target.float()
    before = target.weight.detach().clone()
    if reason == "missing":
        names = ("absent",)
    with pytest.raises(ValueError):
        vocab.load_full_vocab_weight(target, tmp_path, names, 7, 4)
    torch.testing.assert_close(target.weight, before)


def prepare_method():
    source = Path(__file__).resolve().parents[3] / "vllm_ascend/models/qwen3_dspark.py"
    tree = ast.parse(source.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "AscendQwen3DSparkForCausalLM")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "prepare_target_vocabulary")
    scope = {
        "torch": torch,
        "VllmConfig": object,
        "is_qwen4exp_speculative_target": is_qwen4exp_speculative_target,
        "get_rotation_path": lambda cfg: None,
        "TARGET_EMBED_WEIGHT_NAMES": ("model.embed_tokens.weight",),
        "TARGET_LM_HEAD_WEIGHT_NAMES": ("lm_head.weight",),
    }

    def factory(vocab_size, hidden_size, params_dtype):
        result = layer()
        result.quant_method = SimpleNamespace(process_weights_after_loading=lambda layer: None)
        return result

    scope.update(VocabParallelEmbedding=factory, ParallelLMHead=factory)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])), str(source), "exec"), scope)
    return scope["prepare_target_vocabulary"]


@pytest.mark.parametrize("missing_head", [False, True])
def test_missing_vocab_published_atomically_without_modifying_target(tmp_path, missing_head):
    values = torch.arange(28).reshape(7, 4).half()
    names = ("model.language_model.embed_tokens.weight",)
    if not missing_head:
        names += ("lm_head.weight",)
    checkpoint(tmp_path, values, names)
    target_embed, target_head = layer(), layer()
    before = [target_embed.weight.detach().clone(), target_head.weight.detach().clone()]
    draft = SimpleNamespace(
        config=SimpleNamespace(vocab_size=7, hidden_size=4),
        model=SimpleNamespace(fc=torch.nn.Linear(4, 4), embed_tokens=target_embed),
        lm_head=target_head,
        has_own_embed_tokens=False,
        has_own_lm_head=False,
    )
    cfg = SimpleNamespace(
        speculative_config=SimpleNamespace(method="dspark", draft_tensor_parallel_size=1),
        parallel_config=SimpleNamespace(tensor_parallel_size=6, pipeline_parallel_size=1),
        model_config=SimpleNamespace(
            architectures=["Qwen4ExpForConditionalGeneration"], model=str(tmp_path), hf_text_config=draft.config
        ),
    )
    if missing_head:
        with pytest.raises(ValueError, match="exactly one indexed"):
            prepare_method()(draft, cfg)
        assert draft.model.embed_tokens is target_embed and draft.lm_head is target_head
        assert not draft.has_own_embed_tokens and not draft.has_own_lm_head
    else:
        prepare_method()(draft, cfg)
        assert draft.model.embed_tokens is not target_embed and draft.lm_head is not target_head
        assert draft.has_own_embed_tokens and draft.has_own_lm_head
        torch.testing.assert_close(draft.model.embed_tokens.weight[:7], values)
        torch.testing.assert_close(draft.lm_head.weight[:7], values)
    torch.testing.assert_close(target_embed.weight, before[0])
    torch.testing.assert_close(target_head.weight, before[1])
