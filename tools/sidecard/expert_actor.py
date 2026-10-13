# SPDX-License-Identifier: Apache-2.0
"""Complete CUDA backbone with resident hot/CUDA and cold/Ascend experts."""

import hashlib
import json
import time
from pathlib import Path

from .cold_decode_graph import ColdDecodeGraphs
from .cold_dispatch import create_dispatch
from .cold_placement import cuda_keys, load
from .indexer_projection_graph import IndexerProjectionGraph
from .indexer_projection_graph import enabled as projection_graph_enabled
from .layer_decode_graph import DecodeGraphs
from .layer_indexer import cuda_rotation_enabled
from .layer_math import TextLayer, rms_norm, validate_config
from .layer_weights import NPU_LAYERS, CudaExperts, load_dense, prefill_chunk_size, prefill_window_size

ARCHITECTURE = "cuda-backbone-ascend-experts"


class ExpertCoordinator:
    architecture = ARCHITECTURE

    def __init__(self, config_path):
        import torch

        self.torch = torch
        self.spec = json.loads(Path(config_path).read_bytes())
        fused_cuda_indexer = cuda_rotation_enabled(self.spec)
        indexer_graphs = projection_graph_enabled(self.spec)
        self.root, self.output = Path(self.spec["checkpoint"]), Path(self.spec["output"])
        self.output.mkdir(exist_ok=True)
        self.config = json.loads((self.root / "config.json").read_bytes())["text_config"]
        self.max_context = self.spec["max_context"]
        self.prefill_chunk_size = prefill_chunk_size(self.spec)
        self.prefill_window_size = prefill_window_size(self.spec)
        validate_config(self.config, self.max_context)
        self.identity = self.spec["checkpoint_sha256"]
        self.failed = self.stopping = False
        self.position = self.sequence = 0
        self.fingerprints = None
        self.metrics = {"cuda_backbone_ms": []}
        self.cuda = self.experts = None
        self.processes = []
        torch.set_num_threads(2)
        torch.cuda.set_device(0)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
        override = json.loads(Path(self.spec["native_gate_override"]).read_bytes())
        if (
            self.spec.get("routed_swiglu_limit") != 10.0
            or self.spec["cuda_backend"]["library_sha256"] != override["cuda_library_sha256"]
        ):
            raise ValueError("expert runtime requires authenticated clipped kernels")
        hot = load(self.spec)
        try:
            self.experts = create_dispatch(config_path, None, torch, wait=False)
            self.processes = self.experts.processes
            self.cuda = CudaExperts(
                self.root,
                self.identity,
                self.spec["cuda_backend"],
                torch,
                keys=cuda_keys(hot),
                max_chunk=self.prefill_chunk_size,
            )
            self.experts.cuda = self.cuda
            torch.set_num_threads(2)
            weights, self.globals = load_dense(self.root, tuple(range(45)), "cuda", torch, include_globals=True)
            self.layers = {
                index: TextLayer(
                    index,
                    weights[index],
                    self.config,
                    "cuda",
                    torch,
                    self.experts,
                    self.max_context,
                    vectorized_prefill_conv=self.spec.get("cuda_prefill_conv", False),
                    fused_cuda_indexer=fused_cuda_indexer,
                )
                for index in range(45)
            }
            self.prefix_graphs = self.suffix_graphs = self.middle_graphs = None
            if indexer_graphs:
                with torch.inference_mode():
                    for layer in self.layers.values():
                        if not layer.linear:
                            layer.indexer.projection_graph = IndexerProjectionGraph(layer.indexer)
                            layer.indexer.projection_graph_enabled = True
            if self.spec.get("cuda_decode_graphs", False):
                with torch.inference_mode():
                    dsa_graphs = self.spec.get("direct_dma", {}).get("cuda_dsa_graphs", False)
                    self.prefix_graphs = DecodeGraphs(
                        [self.layers[index] for index in range(18)], torch, "cuda", dsa_graphs=dsa_graphs
                    )
                    self.suffix_graphs = DecodeGraphs(
                        [self.layers[index] for index in range(33, 45)], torch, "cuda", dsa_graphs=dsa_graphs
                    )
                    self.middle_graphs = ColdDecodeGraphs(
                        [self.layers[index] for index in range(18, 33)], torch, dsa_graphs=dsa_graphs
                    )
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            self.experts.ready()
            if hasattr(self.experts, "prepare_layers"):
                with torch.inference_mode():
                    self.experts.prepare_layers(self.layers)
            (self.output / "memory-receipt.json").write_text(
                json.dumps(
                    {
                        "gpu_free_bytes": torch.cuda.mem_get_info()[0],
                        "gpu_torch_bytes": torch.cuda.memory_allocated(),
                        "cuda_expert_bytes": self.cuda.oracle.allocated_bytes,
                        "physical_npu_devices": [2, 3],
                        "attention_cache_owner": "cuda",
                        "hccl": False,
                    },
                    indent=2,
                )
                + "\n"
            )
        except BaseException:
            self.close()
            raise

    def reset(self, context_limit=None):
        if self.failed or self.stopping or any(process.poll() is not None for process in self.processes):
            raise RuntimeError("expert worker failed; restart its coordinator")
        self.position = 0
        self.request_context_limit = context_limit or self.max_context
        for layer in self.layers.values():
            layer.reset()
            if not layer.linear:
                layer.sparse_enabled = self.request_context_limit > self.config["index_topk"]

    def forward(self, tokens):
        if (
            self.stopping
            or not 1 <= len(tokens) <= self.prefill_chunk_size
            or self.position + len(tokens) > self.max_context
        ):
            raise ValueError("token chunk exceeds expert runtime admission")
        t = self.torch
        try:
            with t.inference_mode():
                begin = time.perf_counter()
                hidden = t.nn.functional.embedding(t.tensor(tokens, device="cuda"), self.globals["embedding"])
                state = (hidden, None, None, None)
                if self.prefix_graphs is not None:
                    state = self.prefix_graphs.forward(state)
                else:
                    for index in range(18):
                        state = self.layers[index].forward(*state)
                if self.middle_graphs is not None:
                    state = self.middle_graphs.forward(state)
                else:
                    for index in range(18, 33):
                        state = self.layers[index].forward(*state)
                if self.suffix_graphs is not None:
                    state = self.suffix_graphs.forward(state)
                else:
                    for index in range(33, 45):
                        state = self.layers[index].forward(*state)
                normalized = rms_norm(state[0][-1:], self.globals["norm"], self.config["rms_norm_eps"], t)
                logits = t.nn.functional.linear(normalized, self.globals["head"]).float()
                if not bool(t.isfinite(logits).all().cpu()):
                    raise FloatingPointError("expert model produced nonfinite logits")
                t.cuda.synchronize()
                if self.fingerprints is not None:
                    self.fingerprints.append(
                        {
                            "position": self.position,
                            "logits_sha256": hashlib.sha256(logits.cpu().numpy().tobytes()).hexdigest(),
                        }
                    )
                self.metrics["cuda_backbone_ms"].append((time.perf_counter() - begin) * 1000)
                self.position += len(tokens)
                self.sequence += 1
                return logits[0]
        except BaseException:
            self.failed = True
            raise

    def prefill(self, tokens, *, overlap=True, cancelled=None):
        window = getattr(self, "prefill_window_size", self.prefill_chunk_size)
        if window > self.prefill_chunk_size:
            result = None
            for offset in range(0, len(tokens), window):
                if cancelled is not None and cancelled.is_set():
                    raise InterruptedError("generation cancelled between complete model windows")
                result = self.prefill_window(tokens[offset : offset + window])
            return result
        result = None
        for offset in range(0, len(tokens), self.prefill_chunk_size):
            if cancelled is not None and cancelled.is_set():
                raise InterruptedError("generation cancelled between complete model chunks")
            result = self.forward(tokens[offset : offset + self.prefill_chunk_size])
        return result

    def prefill_window(self, tokens):
        """Advance small bounded windows by layer, batching only Ascend experts.

        Every CUDA projection, attention cache update and route calculation
        keeps the original chunk shape/order. Cold MLPs have no recurrent state.
        No decode graph or scratch-backed prepared tensor survives another replay.
        """
        if (
            self.stopping
            or not 1 <= len(tokens) <= self.prefill_window_size
            or self.position + len(tokens) > self.max_context
        ):
            raise ValueError("token window exceeds expert runtime admission")
        t = self.torch
        try:
            with t.inference_mode():
                begin = time.perf_counter()
                chunks = [
                    tokens[offset : offset + self.prefill_chunk_size]
                    for offset in range(0, len(tokens), self.prefill_chunk_size)
                ]
                states = [
                    (
                        t.nn.functional.embedding(t.tensor(chunk, device="cuda"), self.globals["embedding"]),
                        None,
                        None,
                        None,
                    )
                    for chunk in chunks
                ]
                for index, layer in self.layers.items():
                    if index in NPU_LAYERS and len(states) > 1:
                        prepared = []
                        for state in states:
                            value = layer.prepare_forward(*state)
                            prepared.append((*value, *layer.route(value[0])))
                        outputs = self.experts.batch_with_shared(layer, prepared)
                        states = [
                            layer.finish_experts(*value[:4], routed, shared)
                            for value, (routed, shared) in zip(prepared, outputs, strict=True)
                        ]
                    else:
                        states = [layer.forward(*state) for state in states]
                logits = None
                for chunk, state in zip(chunks, states, strict=True):
                    normalized = rms_norm(state[0][-1:], self.globals["norm"], self.config["rms_norm_eps"], t)
                    logits = t.nn.functional.linear(normalized, self.globals["head"]).float()
                    if not bool(t.isfinite(logits).all().cpu()):
                        raise FloatingPointError("expert model produced nonfinite logits")
                    if self.fingerprints is not None:
                        self.fingerprints.append(
                            dict(
                                position=self.position,
                                logits_sha256=hashlib.sha256(logits.cpu().numpy().tobytes()).hexdigest(),
                            )
                        )
                    self.position += len(chunk)
                    self.sequence += 1
                t.cuda.synchronize()
                self.metrics["cuda_backbone_ms"].append((time.perf_counter() - begin) * 1000)
                return logits[0]
        except BaseException:
            self.failed = True
            raise

    def close(self):
        if self.experts is not None:
            self.experts.close()
        for layer in getattr(self, "layers", {}).values():
            indexer = getattr(layer, "indexer", None)
            if indexer is not None and indexer.projection_graph is not None:
                indexer.projection_graph.close()
                indexer.projection_graph = None
                indexer.projection_graph_enabled = False
        for name in ("prefix_graphs", "middle_graphs", "suffix_graphs"):
            graph = getattr(self, name, None)
            if graph is not None:
                graph.close()
                setattr(self, name, None)
        if self.cuda is not None:
            self.cuda.close()
