# SPDX-License-Identifier: Apache-2.0
"""Bounded resident qualification of whole-model breakable prefill graphs.

Use only with the synchronous resident harness. Runtime qualification begins
with one full 640-token prefill request; smaller and mixed batches keep their
existing dispatch. Attention and indexer state updates read live metadata in
eager segments, while the surrounding model operations are captured.
"""

import gc
from copy import copy

PREFILL_GRAPH_TOKENS = 640


def release_native_scratch(native_resources, wrappers):
    """Drop idle experimental scratch only after all old graphs are released."""
    if any(wrapper.entries for wrapper in wrappers):
        raise RuntimeError("release old graphs before clearing native scratch")
    released = 0
    for resource in (native_resources or {}).values():
        fused = resource.get("fused_int4a4") if isinstance(resource, dict) else None
        scratch = getattr(fused, "scratch", None)
        if isinstance(scratch, dict):
            released += len(scratch)
            scratch.clear()
    return released


def original_prefill_function(function, closure_name):
    """Unwrap tagged policies and the older resident prototype closures."""
    seen = set()
    while id(function) not in seen:
        seen.add(id(function))
        original = getattr(function, "__glm_prefill_original__", None)
        if original is None and function.__name__ in ("draft_propose", "draft_dummy", "kda"):
            cells = dict(zip(function.__code__.co_freevars, function.__closure__ or ()))
            if closure_name in cells:
                original = cells[closure_name].cell_contents
        if original is None:
            return function
        function = original
    raise RuntimeError("cyclic prefill wrapper chain")


def eligible_prefill(arguments, *, capturing=False):
    return (
        arguments["num_tokens"] == PREFILL_GRAPH_TOKENS
        and (arguments["num_reqs"] == 1 or capturing)
        and arguments["max_num_scheduled_tokens"] == PREFILL_GRAPH_TOKENS
        and not arguments.get("force_eager", False)
        and arguments.get("force_uniform_decode") is not True
        and not arguments.get("force_has_lora", False)
        and arguments.get("force_num_active_loras", 0) in (None, 0)
        and arguments.get("num_encoder_reqs", 0) == 0
    )


class PrefillGraphPolicy:
    # Resident candidates execute in private modules with postponed annotations;
    # dataclass annotation lookup requires a registered module. Keep the state
    # on this explicit per-worker instance instead.
    def __init__(self):
        self.saved = {}
        self.capturing = False
        self.prefill_dispatches = 0
        self.prefill_replays = 0
        self.draft_dispatcher = None
        self.prefill_dispatcher = None
        self.released_scratch_entries = 0

    def configure(self, runner, mode):
        dispatcher, config = runner.cudagraph_dispatcher, runner.compilation_config
        if not self.saved:
            self.draft_dispatcher = copy(dispatcher)
            self.draft_dispatcher.compilation_config = copy(config)
            self.draft_dispatcher.cudagraph_keys = {key: set(value) for key, value in dispatcher.cudagraph_keys.items()}
            self.draft_dispatcher._bs_to_padded_graph_size = list(dispatcher._bs_to_padded_graph_size)
            self.saved = {
                "sizes": list(config.cudagraph_capture_sizes),
                "max_size": config.max_cudagraph_capture_size,
                "mode": dispatcher.cudagraph_mode,
                "keys": {key: set(value) for key, value in dispatcher.cudagraph_keys.items()},
                "padding": list(dispatcher._bs_to_padded_graph_size),
                "initialized": dispatcher.keys_initialized,
            }
        config.cudagraph_capture_sizes = sorted(set(self.saved["sizes"] + [PREFILL_GRAPH_TOKENS]))
        config.max_cudagraph_capture_size = PREFILL_GRAPH_TOKENS
        dispatcher._compute_bs_to_padded_graph_size()
        dispatcher.initialize_cudagraph_keys(mode.FULL_AND_PIECEWISE, runner.uniform_decode_query_len)
        dispatcher.cudagraph_keys[mode.PIECEWISE] = {
            key for key in dispatcher.cudagraph_keys[mode.PIECEWISE] if key.num_tokens == PREFILL_GRAPH_TOKENS
        }
        self.prefill_dispatcher = copy(dispatcher)
        self.prefill_dispatcher.compilation_config = copy(config)
        self.prefill_dispatcher.cudagraph_keys = {key: set(value) for key, value in dispatcher.cudagraph_keys.items()}
        self.prefill_dispatcher._bs_to_padded_graph_size = list(dispatcher._bs_to_padded_graph_size)

    def restore(self, runner):
        if not self.saved:
            return
        dispatcher, config = runner.cudagraph_dispatcher, runner.compilation_config
        config.cudagraph_capture_sizes = self.saved["sizes"]
        config.max_cudagraph_capture_size = self.saved["max_size"]
        dispatcher.cudagraph_mode = self.saved["mode"]
        dispatcher.cudagraph_keys = self.saved["keys"]
        dispatcher._bs_to_padded_graph_size = self.saved["padding"]
        dispatcher.keys_initialized = self.saved["initialized"]
        self.saved = {}


def extend_replacements(changes, native_resources=None):
    # Worker-only imports keep this diagnostic policy testable without vLLM/NPU.
    import torch
    from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphCapture, eager_break_during_capture
    from vllm.config import CUDAGraphMode
    from vllm.forward_context import get_forward_context

    from vllm_ascend._310p.model_runner_310p import NPUModelRunner310
    from vllm_ascend._310p.worker_310p import NPUWorker310
    from vllm_ascend.compilation.breakable_aclgraph import BreakableACLGraphWrapper
    from vllm_ascend.models.glm5next.sparse_attn_indexer_kpool import SparseAttnIndexerKpool
    from vllm_ascend.models.glm5next_w2 import kda_310
    from vllm_ascend.spec_decode.llm_base_proposer import AscendSpecDecodeBaseProposer
    from vllm_ascend.utils import weak_ref_tensor
    from vllm_ascend.worker.model_runner_v1 import NPUModelRunner

    result = dict(changes)
    policy = PrefillGraphPolicy()
    runner_prefix = "vllm_ascend._310p.model_runner_310p:NPUModelRunner310."
    worker_prefix = "vllm_ascend._310p.worker_310p:NPUWorker310."
    original_determine = getattr(
        NPUModelRunner310._determine_batch_execution_and_padding,
        "__glm_prefill_original__",
        NPUModelRunner310._determine_batch_execution_and_padding,
    )
    original_capture = getattr(
        NPUModelRunner310.capture_model, "__glm_prefill_original__", NPUModelRunner310.capture_model
    )
    original_apply = result.get(worker_prefix + "resident_apply", NPUWorker310.resident_apply)
    original_status = result.get(worker_prefix + "resident_status", NPUWorker310.resident_status)
    original_replay = getattr(
        BreakableACLGraphWrapper._replay, "__glm_prefill_original__", BreakableACLGraphWrapper._replay
    )
    original_indexer = getattr(
        SparseAttnIndexerKpool.forward, "__glm_prefill_original__", SparseAttnIndexerKpool.forward
    )
    proposer_prefix = "vllm_ascend.spec_decode.llm_base_proposer:AscendSpecDecodeBaseProposer."
    original_dummy = result.get(proposer_prefix + "dummy_run", AscendSpecDecodeBaseProposer.dummy_run)
    original_dummy = original_prefill_function(original_dummy, "original_dummy")
    original_propose = result.get(proposer_prefix + "_propose", AscendSpecDecodeBaseProposer._propose)
    original_propose = original_prefill_function(original_propose, "original_propose")
    kda_target = "vllm_ascend.models.glm5next_w2.kda_310:run_stateful_kda_310"
    original_kda = result.get(kda_target, kda_310.run_stateful_kda_310)
    original_kda = original_prefill_function(original_kda, "original_kda")

    def determine(self, **arguments):
        if not eligible_prefill(arguments, capturing=policy.capturing):
            return original_determine(self, **arguments)
        selected = dict(arguments, force_uniform_decode=False, use_cascade_attn=True)
        original_dispatcher = self.cudagraph_dispatcher
        try:
            if not policy.capturing:
                self.cudagraph_dispatcher = policy.prefill_dispatcher
            receipt = NPUModelRunner._determine_batch_execution_and_padding(self, **selected)
        finally:
            self.cudagraph_dispatcher = original_dispatcher
        if receipt[0] != CUDAGraphMode.PIECEWISE:
            raise RuntimeError("qualified prefill size did not select its piecewise graph")
        if not policy.capturing:
            policy.prefill_dispatches += 1
        return receipt

    def capture(self):
        # A long resident experiment retains multiple frozen native resources.
        # Their scratch is disposable, but can otherwise exhaust graph memory.
        # The harness has drained requests and cleared target/draft graphs.
        torch.npu.synchronize()
        wrappers = [self.model, self.drafter.model] if self.drafter is not None else [self.model]
        policy.released_scratch_entries += release_native_scratch(native_resources, wrappers)
        gc.collect()
        torch.npu.empty_cache()
        policy.configure(self, CUDAGraphMode)
        policy.capturing = True
        try:
            return original_capture(self)
        finally:
            policy.capturing = False
            # Only the qualified prefill call uses the enlarged dispatcher.
            # Decode, smaller prefills and draft must retain their original
            # configuration as well as their original graph descriptors.
            policy.restore(self)

    def apply(self, generation):
        session = self._resident_session()
        pending = session.pending
        if pending is not None and session.current is not None and pending[0].digest != session.current.digest:
            policy.restore(self.model_runner)
        return original_apply(self, generation)

    def replay(self, entry, args, kwargs):
        if (
            entry.batch_descriptor.num_tokens == PREFILL_GRAPH_TOKENS
            and get_forward_context().cudagraph_runtime_mode == CUDAGraphMode.PIECEWISE
        ):
            policy.prefill_replays += 1
        return original_replay(self, entry, args, kwargs)

    # CustomOp stores a bound dispatch implementation at construction. Wrap
    # forward, so already-loaded indexer instances execute this eager boundary.
    # Its output is the preallocated top-k buffer; writes use current metadata.
    indexer = eager_break_during_capture(original_indexer)

    def kda(self_attn, mixed_qkv, raw_gate, beta_raw, conv_weight_t):
        current = BreakableCUDAGraphCapture.current()
        if (
            current is None
            or not current._capturing
            or get_forward_context().cudagraph_runtime_mode != CUDAGraphMode.PIECEWISE
        ):
            return original_kda(self_attn, mixed_qkv, raw_gate, beta_raw, conv_weight_t)
        # The 310P instance-bound KDA forward bypasses the shipped model's
        # decorated stateful core. Keep this dynamic core outside capture and
        # copy its result into a stable buffer consumed by captured norm/proj.
        output = torch.empty(
            (1, mixed_qkv.shape[0], self_attn.local_num_heads, self_attn.head_dim),
            dtype=mixed_qkv.dtype,
            device=mixed_qkv.device,
        )
        weak_output = weak_ref_tensor(output)
        weak_inputs = tuple(weak_ref_tensor(value) for value in (mixed_qkv, raw_gate, beta_raw, conv_weight_t))

        def run_current_state():
            value = original_kda(self_attn, *weak_inputs)
            weak_output.copy_(value)

        current.add_eager(run_current_state)
        return output

    def draft_dummy(self, *args, **kwargs):
        tokens = kwargs.get("num_tokens", args[0] if args else 0)
        if policy.capturing and tokens == PREFILL_GRAPH_TOKENS:
            # Target piecewise capture must not register an unqualified MTP
            # graph with num_tokens / query_len dummy requests. Existing draft
            # decode graphs are still captured at their original descriptors.
            return None
        return original_dummy(self, *args, **kwargs)

    def draft_propose(self, num_speculative_tokens, *args, **kwargs):
        target_dispatcher = self.runner.cudagraph_dispatcher
        try:
            if policy.draft_dispatcher is not None:
                # The draft keeps its original keys, maximum and padding table.
                # Enlarging target prefill coverage must not change draft batch
                # padding or force the existing draft path into eager mode.
                self.runner.cudagraph_dispatcher = policy.draft_dispatcher
            return original_propose(self, num_speculative_tokens, *args, **kwargs)
        finally:
            self.runner.cudagraph_dispatcher = target_dispatcher

    def status(self):
        receipt = original_status(self)
        entries = []
        for key, entry in self.model_runner.model.entries.items():
            if key.num_tokens == PREFILL_GRAPH_TOKENS:
                graph = entry.capture
                entries.append(
                    {
                        "descriptor": str(key),
                        "graphs": graph._num_graphs if graph else 0,
                        "eager_breaks": graph._num_eager_breaks if graph else 0,
                    }
                )
        receipt["prefill_graph"] = {
            "tokens": PREFILL_GRAPH_TOKENS,
            "scope": "single-request full prefill chunks; live attention/indexer eager boundaries",
            "dispatches": policy.prefill_dispatches,
            "replays": policy.prefill_replays,
            "released_scratch_entries": policy.released_scratch_entries,
            "entries": entries,
            "free_memory_bytes": torch.npu.mem_get_info()[0],
        }
        return receipt

    for wrapper, original in (
        (determine, original_determine),
        (capture, original_capture),
        (replay, original_replay),
        (indexer, original_indexer),
        (draft_dummy, original_dummy),
        (draft_propose, original_propose),
        (kda, original_kda),
    ):
        wrapper.__glm_prefill_original__ = original
    result[runner_prefix + "_determine_batch_execution_and_padding"] = determine
    result[runner_prefix + "capture_model"] = capture
    result[worker_prefix + "resident_apply"] = apply
    result[worker_prefix + "resident_status"] = status
    result["vllm_ascend.compilation.breakable_aclgraph:BreakableACLGraphWrapper._replay"] = replay
    result["vllm_ascend.models.glm5next.sparse_attn_indexer_kpool:SparseAttnIndexerKpool.forward"] = indexer
    result[proposer_prefix + "dummy_run"] = draft_dummy
    result[proposer_prefix + "_propose"] = draft_propose
    result[kda_target] = kda
    return result
