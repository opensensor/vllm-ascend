# SPDX-License-Identifier: Apache-2.0
"""Opt-in TP2 Ascend fixed-math DSA graphs, with all attention state eager.

Integration contract for ``DecodeGraphs``: construct one adapter per DSA layer,
in ascending order on BOTH ranks, under inference mode on the owner stream.
``agree(layer_index, local_ready)`` MUST rendezvous the pair outside capture and
return the same bool to both owners. Install only if BOTH are ready; otherwise
retain this adapter's eager fallback. No environment switches are introduced.

Only prepare_attention/dsa_prepare and dsa_project are captured. Cache append,
pool/tail compression, positions, sparse selection, causal attention, both HCCL
SUM boundaries, routing and experts stay eager. This is partial capture, never
full sparse-attention or collective capture. Existing CUDA adapters are intact.
Output buffers are borrowed until the next replay; serialize requests as before.
Layer weights/layout remain immutable for the entire graph owner lifetime.

Capture failures can fall back only after owner-local synchronization and graph
cleanup succeed. Replay failures are fatal: retrying a partially executed step
could append cache entries or repeat collectives. The coordinator must abort
both owners using its existing failure protocol. Close only after owner work
is drained; this adapter does not change cache allocation or request admission.
"""

from .layer_decode_graph import CAPTURE_WARMUP_STEPS
from .layer_math import HIDDEN, STREAMS

DSA_LAYERS = (19, 23, 27, 31)
CAPTURE_SCOPE = "partial fixed-math; cache/indexer/attention/collectives/experts eager"


class CaptureCleanupError(RuntimeError):
    """Graph owner cannot prove a safe eager fallback; abort the pair."""


def cache_identity(layer):
    """Host-only metadata: no device readback, item(), or length in graph key."""
    indexer = layer.indexer
    return tuple(
        (value.data_ptr(), tuple(value.shape), value.dtype, value.device)
        for value in (layer.cache, indexer.pools, indexer.tail_keys, indexer.tail_gates)
    )


def dynamic_state(layer):
    return layer.length, layer.indexer.pool_count, layer.indexer.tail_length


class NpuDsaDecodeGraph:
    def __init__(self, layer, torch, *, agree, cold=False):
        if (
            layer.linear
            or not layer.device.startswith("npu:")
            or type(layer.tp_rank) is not int
            or layer.tp_rank not in (0, 1)
            or layer.index not in DSA_LAYERS
        ):
            raise ValueError("Ascend DSA capture requires assigned TP2 layers 19/23/27/31")
        if not callable(agree) or type(cold) is not bool:
            raise ValueError("capture requires a coordinated pair agreement and boolean cold policy")
        self.layer, self.torch, self.cold = layer, torch, cold
        self.diagnostic = getattr(layer, "execution_diagnostic", None)
        self.diagnostic_graph_keys = {}
        self.before = self.projected = None
        self.closed, self.enabled, self.failure = False, False, None
        self.identity = cache_identity(layer)
        original = dynamic_state(layer)
        local_ready = False
        try:
            inputs = (
                torch.zeros((1, HIDDEN), dtype=torch.float16, device=layer.device),
                torch.zeros((1, STREAMS, HIDDEN), dtype=torch.float32, device=layer.device),
                torch.ones((1, STREAMS, 1), dtype=torch.float32, device=layer.device),
                torch.eye(STREAMS, dtype=torch.float32, device=layer.device)[None].contiguous(),
            )
            self.before = self.capture(self.prepare, inputs)
            latent = torch.zeros(
                (1, layer.config["num_attention_heads"] // 2, layer.config["kv_lora_rank"]),
                dtype=torch.float16,
                device=layer.device,
            )
            self.projected = self.capture(lambda values: layer.dsa_project(values[0]), (latent,))
            if dynamic_state(layer) != original or cache_identity(layer) != self.identity:
                raise RuntimeError("fixed-math capture changed eager cache state")
            local_ready = True
        except CaptureCleanupError:
            raise
        except Exception as error:
            self.failure = f"{type(error).__name__}: {error}"
            # Also drain failed warmup, which may precede graph allocation.
            try:
                torch.npu.synchronize()
            except Exception as cleanup_error:
                raise CaptureCleanupError("cannot drain failed DSA warmup") from cleanup_error
            self.release(discard_diagnostics=True)
            if dynamic_state(layer) != original or cache_identity(layer) != self.identity:
                raise RuntimeError("capture mutated attention state; owner cannot safely fall back") from error
        try:
            paired_ready = agree(layer.index, local_ready)
            if type(paired_ready) is not bool or (paired_ready and not local_ready):
                raise ValueError("invalid pair capture agreement")
            self.enabled = paired_ready
            if not paired_ready:
                self.failure = self.failure or "peer rejected partial capture"
                self.release(discard_diagnostics=True)
        except Exception:
            self.release(discard_diagnostics=True)
            raise

    def capture(self, operation, inputs):
        backend = self.torch.npu
        for _ in range(CAPTURE_WARMUP_STEPS):
            operation(inputs)
        backend.synchronize()
        graph = backend.NPUGraph()
        key = f"npu-dsa:{self.layer.index}:{len(self.diagnostic_graph_keys)}"
        if self.diagnostic is not None:
            self.diagnostic.capture_begin(key)
        try:
            try:
                with backend.graph(graph):
                    outputs = operation(inputs)
            finally:
                if self.diagnostic is not None:
                    self.diagnostic.capture_end(key)
            backend.synchronize()
        except Exception:
            # A failed capture must be drained before graph storage is released.
            try:
                backend.synchronize()
                if hasattr(graph, "reset"):
                    graph.reset()
                if self.diagnostic is not None:
                    self.diagnostic.discard_capture(key)
            except Exception as error:
                raise CaptureCleanupError("cannot drain/release failed DSA capture") from error
            raise
        self.diagnostic_graph_keys[id(graph)] = key
        return graph, inputs, outputs

    def observe_replay(self, captured):
        if self.diagnostic is not None:
            self.diagnostic.replayed(self.diagnostic_graph_keys[id(captured[0])])

    def prepare(self, inputs):
        prepared = self.layer.prepare_attention(*inputs)
        return (*prepared, *self.layer.dsa_prepare(prepared[0]))

    @staticmethod
    def replay(captured, sources):
        graph, inputs, outputs = captured
        for target, source in zip(inputs, sources, strict=True):
            if source is None or (source.shape, source.dtype, source.device) != (
                target.shape,
                target.dtype,
                target.device,
            ):
                raise ValueError("DSA graph requires complete owner-local deferred state")
        for target, source in zip(inputs, sources, strict=True):
            target.copy_(source)
        graph.replay()
        return outputs

    def attention(self, state):
        """Same attention path and one eager TP SUM as prepare_forward."""
        if self.closed:
            raise RuntimeError("DSA graph owner is closed")
        if cache_identity(self.layer) != self.identity:
            raise RuntimeError("DSA cache storage changed; rebuild both owner adapters")
        if not self.enabled or state[0].shape[0] != 1:
            x, residual, post, comb = self.layer.prepare_attention(*state)
            projected = self.layer.dsa(x)
        else:
            x, residual, post, comb, qlatent, qr, kv = self.replay(self.before, state)
            self.observe_replay(self.before)
            # Positions, reservation, dense/sparse thresholds and tails are read
            # afresh by eager attention on every step, including after reset.
            latent = self.layer.dsa_attend(x, qlatent, qr, kv)
            projected = self.replay(self.projected, (latent,))
            self.observe_replay(self.projected)
        return self.layer.finish_attention(projected, residual, post, comb)

    def forward(self, state):
        prepared = self.attention(state)
        if self.cold:
            return (*prepared, *self.layer.route(prepared[0]))
        return self.layer.finish_forward(*prepared)

    def release(self, *, discard_diagnostics=False):
        if self.before is None and self.projected is None:
            return
        self.torch.npu.synchronize()
        for captured in (self.before, self.projected):
            if captured is not None:
                if hasattr(captured[0], "reset"):
                    captured[0].reset()
                key = self.diagnostic_graph_keys.pop(id(captured[0]), None)
                if discard_diagnostics and self.diagnostic is not None and key is not None:
                    self.diagnostic.discard_capture(key)
        self.before = self.projected = None

    def close(self):
        if not self.closed:
            self.release()
            self.enabled, self.closed = False, True
