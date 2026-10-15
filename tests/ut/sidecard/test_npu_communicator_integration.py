# SPDX-License-Identifier: Apache-2.0
"""Execute the actual communicator classes with isolated dependency fakes.

Only top-level imports are omitted from the parsed module. Method bodies,
including optional imports, are compiled unchanged. No Torch/NPU initialization
or root vLLM test fixture is needed to qualify these integration hooks.
"""

import ast
import builtins
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "vllm_ascend/distributed/device_communicators/npu_communicator.py"


class Device:
    def __init__(self, kind, index=None):
        self.type, self.index = kind, index


class Tensor:
    pass


class ProcessGroup:
    pass


@pytest.fixture
def isolated():
    context = SimpleNamespace(
        config=None,
        device=2,
        members=(7, 3, 9, 2),
        rank=9,
        events=[],
        imports=[],
        factories=[],
        base_reductions=[],
        base_result=object(),
        custom_result=object(),
        factory_error=None,
        custom_error=None,
        close_error=None,
        use_fallback=False,
        replacement=None,
        supports_pipeline=False,
        custom_calls=[],
    )

    class Base:
        def __init__(self, cpu_group, device, device_group, unique_name, *, use_all2all):
            self.ranks, self.global_rank = context.members, context.rank
            context.events.append(("base-init", cpu_group, device, device_group, unique_name, use_all2all))

        def all_reduce(self, input_):
            context.base_reductions.append(input_)
            context.events.append(("base-reduce", input_))
            return context.base_result

        def destroy(self):
            context.events.append(("base-destroy",))

    class Custom:
        def __init__(self, state="active"):
            self.state = state

        def all_reduce(self, input_, fallback):
            context.custom_calls.append((input_, fallback))
            if context.custom_error is not None:
                self.state = "quarantined"
                raise context.custom_error
            if context.use_fallback:
                return fallback(context.replacement if context.replacement is not None else input_)
            return context.custom_result

        def close(self, *, graphs_retired):
            context.events.append(("custom-close", graphs_retired))
            if context.close_error is not None:
                self.state = "quarantined"
                raise context.close_error
            self.state = "closed"

    context.custom = Custom()

    def create(*args, **kwargs):
        context.factories.append((args, kwargs))
        if context.factory_error is not None:
            raise context.factory_error
        return context.custom

    def current_stream():
        return "owner-current-stream"

    def synchronize():
        context.events.append(("npu-sync",))

    def zeros(*args, **kwargs):
        return ("zeros", args, kwargs)

    torch = SimpleNamespace(
        device=Device,
        Tensor=Tensor,
        bool=object(),
        zeros=zeros,
        npu=SimpleNamespace(current_device=lambda: context.device, synchronize=synchronize),
    )
    dist = SimpleNamespace(ProcessGroup=ProcessGroup)

    class Pipeline:
        def __init__(self, cpu_group, ranks, rank, device):
            context.events.append(("pipeline-init", cpu_group, ranks, rank, device))

        def warmup(self, *, synchronize):
            context.events.append(("pipeline-warmup", synchronize))

        def send_tensor_dict(self, packet, dst):
            context.events.append(("pipeline-send", packet, dst))

        def recv_tensor_dict(self, src):
            context.events.append(("pipeline-recv", src))
            return "received-packet"

        def destroy(self):
            context.events.append(("pipeline-destroy",))

    capability = object()

    def hardware_profile():
        context.events.append(("hardware-profile",))

        def supports(value):
            assert value is capability
            return context.supports_pipeline

        return SimpleNamespace(supports=supports)

    def optional_import(name, globals=None, locals=None, fromlist=(), level=0):
        context.imports.append((name, fromlist, level))
        if name == "tools.sidecard.npu_collective_torch" and fromlist == ("create",) and level == 0:
            return SimpleNamespace(create=create)
        if name == "vllm_ascend.utils" and fromlist == ("current_stream",) and level == 0:
            return SimpleNamespace(current_stream=current_stream)
        raise AssertionError("unexpected isolated communicator import: " + name)

    namespace = dict(
        __name__="isolated_npu_communicator",
        __builtins__={**vars(builtins), "__import__": optional_import},
        torch=torch,
        dist=dist,
        get_current_vllm_config_or_none=lambda: context.config,
        DeviceCommunicatorBase=Base,
        HardwareCapability=SimpleNamespace(DISTRIBUTED_COMMUNICATION_ADAPTATION=capability),
        get_current_hardware_profile=hardware_profile,
        PipelineAllGatherTransport=Pipeline,
    )
    parsed = ast.parse(SOURCE.read_text(), filename=str(SOURCE))
    parsed.body = [node for node in parsed.body if not isinstance(node, (ast.Import, ast.ImportFrom))]
    exec(compile(parsed, str(SOURCE), "exec"), namespace)
    context.cls, context.manager_cls = namespace["NPUCommunicator"], namespace["_NpuAll2AllManager"]
    context.torch, context.dist, context.current_stream = torch, dist, current_stream
    context.cpu_group, context.device_group = ProcessGroup(), ProcessGroup()

    def construct(name="tp:0", *, use_all2all=False):
        requested = Device("npu", 99)
        return context.cls(context.cpu_group, requested, context.device_group, name, use_all2all=use_all2all)

    context.construct = construct
    return context


def configured(context, eager=True, custom=None):
    custom = {"explicit": "custom-config"} if custom is None else custom
    context.config = SimpleNamespace(
        additional_config={"npu_collective": custom},
        model_config=SimpleNamespace(enforce_eager=eager),
    )
    return custom


@pytest.mark.parametrize("config_kind", ("no-config", "no-additional", "empty", "unrelated", "explicit-none"))
def test_default_all_reduce_and_destroy_delegate_unchanged_without_optional_imports(isolated, config_kind):
    ctx = isolated
    additional = {
        "no-additional": None,
        "empty": {},
        "unrelated": {"other": True},
        "explicit-none": {"npu_collective": None},
    }
    if config_kind != "no-config":
        ctx.config = SimpleNamespace(additional_config=additional[config_kind], model_config=None)
    owner = ctx.construct(use_all2all=True)
    tensor = Tensor()
    assert owner.all_reduce(tensor) is ctx.base_result
    assert ctx.base_reductions == [tensor]
    assert owner.custom_sum is None and owner.device == ctx.device
    assert owner.ca_comm is None and owner.fi_pcie_ipc_ar_comm is None
    assert owner.supports_tensor_dict is False and owner.pipeline_transport is None
    owner.close_custom_collective()
    owner.destroy()
    assert [event[0] for event in ctx.events] == ["base-init", "base-reduce", "base-destroy"]
    assert ctx.events[0][-1] is True and not ctx.imports and not ctx.factories


@pytest.mark.parametrize("name", ("tp", "tp:0", "tp:nonmonotonic-group"))
def test_tp_factory_receives_actual_order_rank_groups_device_and_injected_dependencies(isolated, name):
    ctx = isolated
    custom = configured(ctx)
    owner = ctx.construct(name)
    assert owner.custom_sum is ctx.custom
    assert len(ctx.factories) == 1
    args, kwargs = ctx.factories[0]
    assert args == (custom, ctx.members, ctx.rank, ctx.cpu_group, ctx.device)
    assert args[0] is custom and type(args[1]) is tuple
    assert kwargs == dict(torch=ctx.torch, dist=ctx.dist, stream_getter=ctx.current_stream)
    assert [name for name, _, _ in ctx.imports] == ["tools.sidecard.npu_collective_torch", "vllm_ascend.utils"]
    tensor = Tensor()
    assert owner.all_reduce(tensor) is ctx.custom_result
    assert ctx.custom_calls[0][0] is tensor and not ctx.base_reductions


@pytest.mark.parametrize("name", ("ep:0", "pp:0", "dp:0", "tp-other", "notp:0", ""))
def test_non_tp_communicators_never_initialize_custom_sum_or_apply_its_eager_gate(isolated, name):
    ctx = isolated
    configured(ctx, eager=False)
    owner = ctx.construct(name)
    assert owner.custom_sum is None and not ctx.imports and not ctx.factories
    tensor = Tensor()
    assert owner.all_reduce(tensor) is ctx.base_result


@pytest.mark.parametrize("eager", (False, None, 0, 1, "true", [], object()))
def test_custom_tp_rejects_graph_or_nonbool_eager_before_importing_factory(isolated, eager):
    ctx = isolated
    configured(ctx, eager=eager)
    with pytest.raises(ValueError, match="explicitly eager"):
        ctx.construct()
    assert not ctx.factories and not ctx.imports


def test_custom_tp_rejects_missing_model_before_factory(isolated):
    ctx = isolated
    configured(ctx)
    ctx.config.model_config = None
    with pytest.raises(ValueError, match="explicitly eager"):
        ctx.construct()
    assert not ctx.factories and not ctx.imports


@pytest.mark.parametrize("replacement", (False, True))
def test_facade_fallback_callback_accepts_and_delegates_the_supplied_tensor(isolated, replacement):
    ctx = isolated
    configured(ctx)
    ctx.use_fallback = True
    tensor = Tensor()
    if replacement:
        ctx.replacement = Tensor()
    owner = ctx.construct()
    assert owner.all_reduce(tensor) is ctx.base_result
    assert ctx.base_reductions == [ctx.replacement if replacement else tensor]
    assert ctx.custom_calls[0][0] is tensor


@pytest.mark.parametrize("error_type", (RuntimeError, KeyboardInterrupt))
@pytest.mark.parametrize("partial", (False, True))
def test_factory_failure_retains_its_partial_facade_and_never_cleans_up_automatically(isolated, error_type, partial):
    ctx = isolated
    configured(ctx)
    error = error_type("native setup failed")
    if partial:
        error.collective_owner = ctx.custom
    ctx.factory_error = error
    owner = object.__new__(ctx.cls)
    with pytest.raises(error_type, match="native setup failed") as caught:
        ctx.cls.__init__(owner, ctx.cpu_group, None, ctx.device_group, "tp:0")
    assert caught.value is error
    assert owner.custom_sum is (ctx.custom if partial else None)
    assert not any(event[0].endswith(("close", "destroy")) for event in ctx.events)
    assert not ctx.base_reductions


def test_custom_submission_failure_never_falls_back_or_drops_retained_owner(isolated):
    ctx = isolated
    configured(ctx)
    owner = ctx.construct()
    ctx.custom_error = RuntimeError("partial native submission; retain owners")
    with pytest.raises(RuntimeError, match="partial native submission") as caught:
        owner.all_reduce(Tensor())
    assert caught.value is ctx.custom_error
    assert owner.custom_sum is ctx.custom and owner.custom_sum.state == "quarantined"
    assert not ctx.base_reductions
    assert not any(event[0].endswith(("close", "destroy")) for event in ctx.events)


def test_explicit_custom_close_is_idempotent_and_precedes_base_destruction(isolated):
    ctx = isolated
    configured(ctx)
    owner = ctx.construct()
    owner.close_custom_collective()
    owner.close_custom_collective()
    assert ctx.custom.state == "closed"
    assert [event[0] for event in ctx.events].count("custom-close") == 1
    owner.destroy()
    assert [event[0] for event in ctx.events][-2:] == ["custom-close", "base-destroy"]
    assert ("custom-close", True) in ctx.events


def test_destroy_closes_custom_before_base_without_prior_explicit_close(isolated):
    ctx = isolated
    configured(ctx)
    owner = ctx.construct()
    owner.destroy()
    assert [event[0] for event in ctx.events][-2:] == ["custom-close", "base-destroy"]


def test_failed_custom_close_blocks_base_destroy_and_retains_facade(isolated):
    ctx = isolated
    configured(ctx)
    owner = ctx.construct()
    ctx.close_error = RuntimeError("uncertain peer shutdown")
    with pytest.raises(RuntimeError, match="uncertain peer shutdown"):
        owner.destroy()
    assert owner.custom_sum is ctx.custom and owner.custom_sum.state == "quarantined"
    assert "base-destroy" not in [event[0] for event in ctx.events]


@pytest.mark.parametrize("custom_enabled", (False, True))
def test_existing_pipeline_transport_warmup_send_receive_and_teardown_are_preserved(isolated, custom_enabled):
    ctx = isolated
    if custom_enabled:
        configured(ctx, eager=False)
    ctx.supports_pipeline = True
    owner = ctx.construct("pp:17")
    assert owner.supports_tensor_dict and owner.custom_sum is None
    initialized = next(event for event in ctx.events if event[0] == "pipeline-init")
    assert initialized[1:4] == (ctx.cpu_group, ctx.members, ctx.rank)
    assert (initialized[4].type, initialized[4].index) == ("npu", ctx.device)
    assert ("pipeline-warmup", ctx.torch.npu.synchronize) in ctx.events
    packet = {"hidden": Tensor()}
    assert owner.send_tensor_dict(packet, dst=3) is None
    assert owner.recv_tensor_dict(src=7) == "received-packet"
    owner.send_tensor_dict(packet)
    assert owner.recv_tensor_dict() == "received-packet"
    assert ("pipeline-send", packet, 3) in ctx.events and ("pipeline-send", packet, None) in ctx.events
    assert ("pipeline-recv", 7) in ctx.events and ("pipeline-recv", None) in ctx.events
    owner.destroy()
    assert [event[0] for event in ctx.events][-2:] == ["pipeline-destroy", "base-destroy"]
    assert not ctx.factories and not ctx.imports


def test_tensor_dict_without_pipeline_remains_an_error_and_noop_all2all_contract_is_unchanged(isolated):
    ctx = isolated
    owner = ctx.construct()
    with pytest.raises(RuntimeError, match="pipeline transport is unavailable"):
        owner.send_tensor_dict({})
    with pytest.raises(RuntimeError, match="pipeline transport is unavailable"):
        owner.recv_tensor_dict()
    manager = owner.all2all_manager
    assert isinstance(manager, ctx.manager_cls) and manager.support_fault_tolerance is False
    for result in (manager.query_fault(), manager.query_active_mask()):
        assert result == ("zeros", (1,), dict(dtype=ctx.torch.bool, device="cpu"))


def test_graph_model_requires_explicit_graph_admission(isolated):
    ctx = isolated
    config = configured(ctx, eager=False, custom={"execution_mode": "graph"})
    owner = ctx.construct()
    assert owner.custom_sum is ctx.custom
    assert ctx.factories[0][0][0] is config
