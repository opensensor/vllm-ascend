# SPDX-License-Identifier: Apache-2.0
"""One explicit streaming composition; imports submit no device work.

Grouped eager W4 prefill is the candidate mode. Sparse decode, graph capture,
MTP W8A16, vision and unsupported geometries retain their original paths. The
resource loader is a separate, admitted operation and is never used by offline
tests. CPU callback composition is not hardware qualification.
"""

from dataclasses import asdict, dataclass
from functools import partial

from tools.qwen4exp.streaming_epilogue import PROJECTED_COLUMNS, WindowPlan, run_streaming_epilogue
from tools.qwen4exp.streaming_layer import LayerComposition, LayerPolicy, LayerResources
from tools.qwen4exp.streaming_memory import MAX_K
from tools.qwen4exp.streaming_operands import MAX_EXPERTS, MAX_TOP_K, prepare_grouped_operands
from tools.qwen4exp.streaming_protocol import canonical_sha256, file_sha256
from tools.qwen4exp.streaming_schedule import DeviceScheduleContext, SchedulePlan, validate_rank_plan

MOE_TARGET = "vllm_ascend.models.qwen4_exp.w4_moe:W4SparseMoE.forward"


@dataclass(frozen=True)
class CandidateConfig:
    schedule: SchedulePlan
    windows: WindowPlan = WindowPlan()
    layers: LayerPolicy = LayerPolicy()

    def __post_init__(self):
        if not isinstance(self.schedule, SchedulePlan):
            raise ValueError("candidate requires a validated rank schedule")
        if not isinstance(self.windows, WindowPlan) or self.windows.output_columns != MAX_K:
            raise ValueError("candidate targets the complete 2560-column model output")
        if not isinstance(self.layers, LayerPolicy):
            raise ValueError("candidate requires an immutable layer policy")

    @property
    def sha256(self):
        return canonical_sha256(asdict(self))


@dataclass(frozen=True)
class CandidateResources:
    inventory_sha256: str
    projection: object
    gather: object
    pack: object
    activation: object
    finalize: object
    route: object
    dispatch: object
    schedule: object
    context: object
    layers: LayerResources

    def __post_init__(self):
        digest = self.inventory_sha256
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("resource inventory requires a lowercase SHA256")
        if not all(
            callable(value)
            for value in (
                self.projection,
                getattr(self.projection, "columns", None),
                self.gather,
                self.pack,
                self.activation,
                self.finalize,
                self.route,
                self.dispatch,
                self.schedule,
            )
        ):
            raise ValueError("candidate native/arithmetic callbacks are incomplete")


class StreamingCandidate:
    """A single owner binds MoE and layer hooks after immutable admission.

    Composition helpers can be exercised with CPU callbacks. Installation must
    pass an Admission, which rehashes files and binds all policies/resources.
    Neither a namespace string nor CPU test success grants execution permission.
    """

    def __init__(self, config, resources, rank_receipts, *, admission=None):
        if not isinstance(config, CandidateConfig) or not isinstance(resources, CandidateResources):
            raise ValueError("validated candidate configuration and resources required")
        self.config, self.resources = config, resources
        self.rank_receipts = validate_rank_plan(config.schedule, rank_receipts)
        self.layer = LayerComposition(config.layers, resources.layers)
        self.admission = admission
        self.poisoned = False

    def require_admission(self):
        if self.admission is None:
            raise ValueError("hardware/evidence admission is pending")
        self.admission.require_execution(
            configuration_sha256=self.config.sha256,
            resources_sha256=self.resources.inventory_sha256,
            plan_sha256=self.config.schedule.sha256,
        )
        if tuple(self.admission.plan_receipts) != self.rank_receipts:
            raise ValueError("candidate rank receipts differ from admitted worker identities")

    def local(self, module, inputs, weights, ids):
        prepared = prepare_grouped_operands(
            inputs,
            weights,
            ids,
            pack=self.resources.pack,
            dispatch=partial(
                self.resources.dispatch, weight_dtype=module.compute_dtype, count_mode=module.grouped_route_count_mode
            ),
            gather=self.resources.gather,
            num_local_experts=module.num_local_experts,
            expert_offset=module.expert_offset,
        )
        projected = self.resources.projection(
            module.projections["gate_up_proj"], prepared.local_operands, prepared.group_ends
        )
        dispatch, group_ends = prepared.dispatch, prepared.group_ends
        # Projection/gather/pack launch on the current stream. Drop token and
        # compact operand owners before activation; native launch retains inputs
        # through enqueue and the caching allocator owns same-stream reuse.
        del prepared
        return run_streaming_epilogue(
            projected,
            dispatch,
            weights,
            group_ends,
            module.projections["down_proj"],
            activation=self.resources.activation,
            pack=lambda hidden, _ends: self.resources.pack(hidden),
            columns=self.resources.projection.columns,
            finalize=lambda routed, route_dispatch, route_weights: self.resources.finalize(
                routed, route_dispatch, route_weights, module.compute_dtype, self.config.windows.finalizer_policy
            ),
            complete=lambda _stage, _tensor: None,
            plan=self.config.windows,
        ).output

    def supports(self, module, inputs, *, capturing):
        # These are host metadata checks; no device route/count inspection.
        shared_policy = (
            "none"
            if not module.has_shared_expert
            else "replicated"
            if module.shared_expert_replicated
            else "tp_sharded"
        )
        return (
            not capturing
            and len(inputs.shape) == 2
            and 0 < inputs.shape[0] <= self.config.schedule.max_input_tokens
            and inputs.shape[1] == MAX_K
            and str(inputs.dtype) == "torch.float16"
            and str(module.params_dtype) == "torch.float16"
            and module.native_int4
            and module.grouped_routing
            and not (module.device_routing and inputs.shape[0] * module.top_k <= module.max_routed_rows)
            and module.grouped_activation == self.config.windows.activation_policy
            and module.grouped_finalize == self.config.windows.finalizer_policy
            and str(module.compute_dtype) == "torch.float32"
            and module.expert_tp_size == self.config.schedule.tp_size
            and shared_policy == self.config.schedule.policy.shared_policy
            and module.num_local_experts == MAX_EXPERTS
            and module.projections["gate_up_proj"].weight.shape == (MAX_EXPERTS, PROJECTED_COLUMNS, MAX_K // 2)
            and module.projections["down_proj"].weight.shape == (MAX_EXPERTS, MAX_K, PROJECTED_COLUMNS // 4)
            and 1 <= module.top_k <= MAX_TOP_K
        )

    def forward(self, original, module, inputs, *, capturing):
        if self.poisoned:
            raise RuntimeError("streaming candidate is poisoned; dispatch must remain held")
        if not self.supports(module, inputs, capturing=capturing):
            return original(module, inputs)
        try:
            output = self.resources.schedule(
                module,
                inputs,
                self.config.schedule,
                self.rank_receipts,
                route=lambda tensor: self.resources.route(module, tensor),
                local=partial(self.local, module),
                context=self.resources.context,
            )
            return output.to(module.params_dtype)
        except BaseException:
            # Do not catch an uncertain enqueue and silently retry the baseline.
            self.poisoned = True
            raise

    def replacements(self, *, moe_original, capturing, **layer_originals):
        self.require_admission()
        if not callable(moe_original) or any(
            hasattr(moe_original, key) for key in ("_qwen_w4_forward_base", "_qwen_streaming_owner")
        ):
            raise ValueError("streaming requires an original MoE method")
        hooks = {}
        for target, invoke in self.layer.replacements(**layer_originals).items():

            def layer_call(*args, _invoke=invoke, **kwargs):
                if self.poisoned:
                    raise RuntimeError("streaming candidate is poisoned; dispatch must remain held")
                try:
                    return _invoke(*args, **kwargs)
                except BaseException:
                    self.poisoned = True
                    raise

            layer_call._qwen_streaming_layer_owner = self
            hooks[target] = layer_call

        def forward(module, inputs):
            return self.forward(moe_original, module, inputs, capturing=capturing(inputs))

        forward._qwen_streaming_owner = self
        hooks[MOE_TARGET] = forward
        return hooks


def prepare_native_resources(bundle_root, config, admission):
    """Future admitted loader. Never call this during offline implementation.

    All byte/gate/whole-rank checks precede torch-npu imports, library registration
    and kernel discovery. Loading remains append-only; failure requires the
    outer owned-maintenance controller to keep dispatch held.
    """
    from pathlib import Path

    from tools.qwen4exp.build_streaming import verify_bundle

    root = Path(bundle_root).resolve(strict=True)
    bundle = verify_bundle(root, require_compiled=True)
    admission.require_execution(
        configuration_sha256=config.sha256,
        resources_sha256=bundle["resources_sha256"],
        plan_sha256=config.schedule.sha256,
    )
    runtime_root = Path(__file__).resolve().parents[2]
    for asset in bundle["assets"]:
        relative = Path(asset["path"])
        if relative.parts[0] == "sources" and relative.suffix in (".py", ".cpp", ".h"):
            active = runtime_root.joinpath(*relative.parts[1:])
            if not active.is_file() or file_sha256(active) != asset["sha256"]:
                raise ValueError("active runtime source differs from frozen candidate")
    # Source-byte checks cannot authenticate already imported code objects.
    # T9 must freeze the worker import root before startup and attest it on all
    # ranks; never update Python sources underneath existing workers.
    import torch
    import torch.nn.functional as F
    import torch_npu

    from tools.qwen4exp.native_prefill import NativeLocalRouteGather, NativeWY
    from tools.qwen4exp.native_state_layout import NativeStateLayout
    from tools.qwen4exp.native_streaming import NativeStreamingProjection
    from tools.qwen4exp.streaming_schedule import npu_streaming_prefill
    from vllm_ascend.models.qwen4_exp.grouped_expert_dispatch import build_grouped_expert_dispatch
    from vllm_ascend.models.qwen4_exp.w4_moe import DeferredReduceStream, finalize_grouped_routes, route_topk
    from vllm_ascend.models.qwen4_exp.w4a8_int4 import pack_activation_device

    namespace = bundle["namespace"]
    if len(bundle["bridges"]) != 1:
        raise ValueError("one matched versioned bridge required")
    torch.ops.load_library(str(root / bundle["bridges"][0]["path"]))
    factory = getattr(torch.classes, namespace).Kernel
    launch = getattr(torch.ops, namespace).launch

    def kernel(binary, entry):
        matches = [record for record in bundle["binaries"] if entry in record["entrypoints"]]
        if len(matches) != 1 or Path(matches[0]["path"]).stem != binary:
            raise ValueError("kernel identity differs from admitted entrypoint inventory")
        return factory(str(root / matches[0]["path"]), entry)

    projection = NativeStreamingProjection(
        kernel("native_streaming", "qwen_streaming_projection_v1"),
        launch,
        kernel("native_streaming", "qwen_streaming_columns_v1"),
    )
    state = NativeStateLayout(
        kernel("native_state_layout", "qwen_state_gather_v1"),
        kernel("native_state_layout", "qwen_state_scatter_v1"),
        launch,
    )
    resources = CandidateResources(
        inventory_sha256=bundle["resources_sha256"],
        projection=projection,
        gather=NativeLocalRouteGather(kernel("native_route_gather", "qwen_local_route_gather_v1"), launch),
        pack=pack_activation_device,
        activation=lambda tensor: torch_npu.npu_swiglu(tensor, dim=-1),
        finalize=finalize_grouped_routes,
        route=lambda module, tensor: route_topk(
            F.linear(tensor, module.gate),
            module.top_k,
            renormalize=module.renormalize,
            routed_scaling_factor=module.routed_scaling_factor,
        ),
        dispatch=build_grouped_expert_dispatch,
        schedule=npu_streaming_prefill,
        context=DeviceScheduleContext(DeferredReduceStream()),
        layers=LayerResources(state_io=state, wy=NativeWY(kernel("native_wy", "qwen_fused_wy_v1"), launch)),
    )
    return resources
