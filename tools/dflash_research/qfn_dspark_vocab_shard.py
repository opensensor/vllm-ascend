# SPDX-License-Identifier: Apache-2.0
"""Resident experiment: partition DSpark vocabulary work over target TP ranks.

The transformer remains replicated TP1. Only the vocabulary and Markov output
rows are partitioned; the existing QFN argmax reduction communicates winners.
Original weights and methods remain owned for verification and restoration.
"""

import hashlib
import inspect
import textwrap
from types import FunctionType

import torch
import torch.nn.functional as F

from vllm_ascend._310p.qwen4exp_mtp import greedy_tokens_from_local_logits

VOCAB_ALIGNMENT = 64
ARGMAX_SEAM = "next_token_ids = logits[:, idx].argmax(dim=-1)"


def shard_bounds(vocabulary, rank, world):
    if not 0 <= rank < world or vocabulary <= 0:
        raise ValueError("invalid vocabulary partition")
    width = ((vocabulary + world * VOCAB_ALIGNMENT - 1) // (world * VOCAB_ALIGNMENT)) * VOCAB_ALIGNMENT
    start = rank * width
    if start >= vocabulary:
        raise ValueError("empty vocabulary partition")
    return start, min(start + width, vocabulary), width


def shard_weight(weight, rank, world):
    start, end, width = shard_bounds(weight.shape[0], rank, world)
    local = weight[start:end]
    device = weight.device
    if device.type == "npu":
        # A view of an NZ tensor can keep the full tensor's storage descriptor.
        # Own the logical shard before packing it; merely calling contiguous()
        # on the view does not establish a valid NZ Matmul operand on 310P.
        local = local.cpu()
    if end - start < width:
        local = F.pad(local, (0, 0, 0, width - (end - start)))
    local = local.contiguous()
    if device.type == "npu":
        import torch_npu

        local = torch_npu.npu_format_cast(local.to(device), 29)
        assert torch_npu.get_npu_format(local) == 29
    return local


def build_loop(original, sample, expected_sha256):
    source = textwrap.dedent(inspect.getsource(original))
    if hashlib.sha256(source.encode()).hexdigest() != expected_sha256 or source.count(ARGMAX_SEAM) != 1:
        raise ValueError("DSpark sampling loop differs from the reviewed source")
    body = source.replace(ARGMAX_SEAM, "next_token_ids = _sample_sharded(self, logits[:, idx])")
    factory = "def factory(_sample_sharded):\n" + textwrap.indent(body, "    ") + "    return _run_merged_draft\n"
    namespace = dict(original.__globals__)
    exec(compile(factory, "<qfn-dspark-sharded-vocabulary>", "exec"), namespace)
    compiled = namespace["factory"](sample)
    return FunctionType(
        compiled.__code__, original.__globals__, compiled.__name__, compiled.__defaults__, compiled.__closure__
    )


def canonical_input(value, group):
    # TP1 draft replicas can inherit small rank-dependent target differences.
    # Partition a single rank's projection, not a mixture of six projections.
    owned = torch.empty(value.shape, dtype=value.dtype, device=value.device)
    owned.copy_(value)
    return group.broadcast(owned, src=0)


class ShardedVocabulary:
    def __init__(self, worker, expected_sha256):
        from vllm.distributed import get_tp_group

        from vllm_ascend._310p.spec_decode import llm_base_proposer_310

        runner = worker.model_runner
        self.proposer = runner.drafter
        self.model = self.proposer.model.unwrap()
        self.group = get_tp_group()
        assert self.group.world_size == 6 and self.group.rank_in_group == worker.rank
        assert runner.speculative_config.method == "dspark" and not runner.use_async_scheduling
        assert runner.speculative_config.draft_tensor_parallel_size == 1
        assert runner.speculative_config.enforce_eager and not self.proposer.use_cuda_graph
        assert not self.proposer._enable_probabilistic_draft_probs and self.proposer.dynamic_spec is None
        self.draft_k = self.proposer.num_speculative_tokens
        assert self.draft_k in (2, 7)
        assert self.model.draft_id_to_target_id is None
        assert self.model.lm_head.tp_size == 1 and not self.model.lm_head.num_added_embeddings
        assert self.model.logits_processor.soft_cap is None and self.model.logits_processor.scale == 1.0
        assert self.model.logits_processor.head_dtype in (None, torch.float16)
        for head in (self.model.lm_head, self.model.model.markov_head.markov_w2):
            assert head.weight.dtype == torch.float16
            assert head.weight.shape[0] == self.model.config.vocab_size
            assert type(head.quant_method).__name__ in (
                "UnquantizedEmbeddingMethod",
                "AscendUnquantizedEmbeddingMethod310",
            )
        self.rank = worker.rank
        self.start, self.end, self.width = shard_bounds(self.model.config.vocab_size, self.rank, self.group.world_size)
        self.padding = self.width - (self.end - self.start)
        self.head = shard_weight(self.model.lm_head.weight, self.rank, self.group.world_size)
        self.markov = shard_weight(self.model.model.markov_head.markov_w2.weight, self.rank, self.group.world_size)
        self.original_logits = self.model.compute_draft_logits
        self.original_bias = self.model.markov_bias
        self.private_checks = []
        with torch.inference_mode():
            for rows in (self.draft_k, self.draft_k * 4):
                for label, weight, original in (
                    ("head", self.head, self.original_logits),
                    ("markov", self.markov, self.original_bias),
                ):
                    generator = torch.Generator().manual_seed(731 + rows)
                    inputs = torch.randn(rows, weight.shape[1], generator=generator).half().to(weight.device)
                    expected = original(inputs)[:, self.start : self.end]
                    actual = F.linear(inputs, weight)[:, : self.end - self.start]
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    self.private_checks.append(dict(kind=label, rows=rows, exact=True))
        self.loop_module = llm_base_proposer_310
        self.original_loop = self.loop_module._original_run_merged_draft
        self.loop = build_loop(self.original_loop, self.sample, expected_sha256)
        self.verify = True
        self.calls = self.checked = self.mismatches = 0
        self.input_checks = self.input_differences = 0
        self.max_input_difference = 0.0
        self.full_logits = self.full_bias = None
        self.position = 0

    def install(self):
        self.model.compute_draft_logits = self.logits
        self.model.markov_bias = self.bias
        self.loop_module._original_run_merged_draft = self.loop

    def logits(self, hidden):
        hidden = self.align_input(hidden)
        self.position = 0
        self.calls += 1
        if self.verify:
            self.full_logits = self.original_logits(hidden).view(-1, self.draft_k, self.model.config.vocab_size)
        return F.linear(hidden, self.head)

    def bias(self, embeddings):
        embeddings = self.align_input(embeddings)
        if self.verify:
            self.full_bias = self.original_bias(embeddings)
        return F.linear(embeddings, self.markov)

    def align_input(self, value):
        aligned = canonical_input(value, self.group)
        if self.verify:
            self.input_checks += 1
            if not torch.equal(value, aligned):
                self.input_differences += 1
                error = float((value - aligned).abs().max().item())
                self.max_input_difference = max(self.max_input_difference, error)
        return aligned

    def sample(self, proposer, logits):
        assert proposer is self.proposer
        ids = greedy_tokens_from_local_logits(logits, self.start, self.padding, self.group.all_gather)
        if self.verify:
            expected = (self.full_logits[:, self.position] + self.full_bias).argmax(dim=-1)
            self.checked += 1
            if not torch.equal(ids, expected):
                self.mismatches += 1
                ids = expected
        self.position += 1
        return ids

    def configure(self, mode):
        if mode == "fast":
            assert self.checked >= 32 and self.mismatches == 0, "real-token qualification required"
            self.verify = False
            self.full_logits = self.full_bias = None
        elif mode == "restore":
            self.model.compute_draft_logits = self.original_logits
            self.model.markov_bias = self.original_bias
            self.loop_module._original_run_merged_draft = self.original_loop
        elif mode != "status":
            raise ValueError("unknown sharded vocabulary mode")
        return self.status()

    def status(self):
        return dict(
            rank=self.rank,
            start=self.start,
            end=self.end,
            width=self.width,
            verify=self.verify,
            calls=self.calls,
            checked=self.checked,
            mismatches=self.mismatches,
            input_checks=self.input_checks,
            input_differences=self.input_differences,
            max_input_difference=self.max_input_difference,
            canonical_rank=0,
            draft_k=self.draft_k,
            private_checks=self.private_checks,
        )
