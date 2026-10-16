# SPDX-License-Identifier: Apache-2.0
"""Compose complete expert streaming with qualified attention/normalization.

Worker-only imports are lazy. Shadow reruns only the stateless MoE branch; it
never calls attention/KDA twice. Tensor readback is confined to untimed shadow
qualification, while ordinary progress/audit reads only host metadata.
"""

import hashlib
import inspect
from pathlib import Path

from tools.glm_perf.native_launch_audit import NativeLaunchAudit
from tools.glm_perf.prefill_progress import PrefillProgress

WORKER_PREFIX = "vllm_ascend._310p.worker_310p:NPUWorker310."
MOE_FORWARD = "vllm_ascend.models.glm5next_w2.moe:Glm5NextW2MoE.forward"
SOURCE_BOUNDARY = "\nimport sys,types\n_progress_module="
BASELINE_VERSION = 1021
DECODE_VERSION = 984
PREFILL_BUDGET = 1280
SHADOW_CHUNKS = (640, 1280)


class ShadowExpert:
    """Compare independent native outputs before the surrounding shared branch."""

    def __init__(self, current, reference, state, inventory):
        self.current = current
        self.reference = reference
        self.state = state
        self.inventory = inventory
        self.reference_mode = False

    def __getattr__(self, name):
        return getattr(self.current, name)

    def __call__(self, x, *args, **kwargs):
        if self.reference_mode:
            return self.reference(x, *args, **kwargs)
        result = self.current(x, *args, **kwargs)
        if x.shape[0] in SHADOW_CHUNKS:
            import torch

            expected = self.reference(x, *args, **kwargs)
            actual_cpu, expected_cpu = result.detach().cpu(), expected.detach().cpu()
            pointer = args[0].data_ptr()
            geometry = self.current.geometry(x, *args[:6])
            bank = self.inventory.get(pointer, {"ordinal": -1})
            finite = bool(torch.isfinite(actual_cpu).all() and torch.isfinite(expected_cpu).all())
            equal = pointer in self.inventory and torch.equal(
                actual_cpu.view(torch.int32), expected_cpu.view(torch.int32)
            )
            self.state["cases"].append(
                {
                    "bank": bank["ordinal"],
                    "bank_pointer": pointer,
                    "bits": geometry.gate_bits,
                    "tokens": x.shape[0],
                    "experts": geometry.experts,
                    "hidden": geometry.hidden,
                    "intermediate": geometry.intermediate,
                    "finite": finite,
                    "bitwise_equal": equal,
                }
            )
            if not finite or not equal:
                self.state["mismatches"] += 1
                # Finish the same collective schedule on every rank. The host
                # rejects recorded mismatches after the full request completes.
                # Raising on one rank here could strand peers inside HCCL.
                pass
        return result


def wrap_moe_forward(original, proxy, state):
    """All ranks execute both pure MoE calls in the same order, before readback."""
    original = getattr(original, "__streaming_moe_original__", original)

    def forward(self, hidden_states, router_logits, *args, **kwargs):
        if hidden_states.shape[0] not in SHADOW_CHUNKS:
            return original(self, hidden_states, router_logits, *args, **kwargs)
        import torch

        actual = original(self, hidden_states, router_logits, *args, **kwargs)
        proxy.reference_mode = True
        try:
            expected = original(self, hidden_states, router_logits, *args, **kwargs)
        finally:
            proxy.reference_mode = False
        a, b = actual.detach().cpu(), expected.detach().cpu()
        finite = bool(torch.isfinite(a).all() and torch.isfinite(b).all())
        equal = torch.equal(a.view(torch.int32), b.view(torch.int32))
        state["whole_moe"].append({"tokens": hidden_states.shape[0], "finite": finite, "bitwise_equal": equal})
        if not finite or not equal:
            state["mismatches"] += 1
            # Complete all layers/ranks before the controller rejects the gate.
            pass
        return actual

    forward.__dict__.update(getattr(original, "__dict__", {}))
    forward.__streaming_moe_original__ = original
    return forward


def extend_replacements(
    parent_factory, native_resources, *, version, build_dir, expected_inventory, shadow=False, profile_dir=None
):
    from vllm.logger import logger

    from vllm_ascend._310p.worker_310p import NPUWorker310

    current = native_resources[f"reconstruction_v{version}"]["fused_int4a4"]
    reference = native_resources["reconstruction_v1021"]["fused_int4a4"]
    if current.shared_prefill_max_tokens != PREFILL_BUDGET:
        if version == BASELINE_VERSION:
            raise ValueError("qualified reference prefill capacity changed")
        # Newly admitted version has never owned a live graph. Its standalone
        # validation scratch is retired; current decode graphs own v984 only.
        current.scratch.clear()
        current.configs.clear()
        current.configure_prefill_scratch(PREFILL_BUDGET)
    # The parent retains normalization resources through this dictionary. Its
    # retired-graph cleanup owns clearing scratch only after graph retirement.
    state = {"cases": [], "whole_moe": [], "mismatches": 0, "progress": [], "profiler_active": False}
    inventory = {}
    proxy = ShadowExpert(current, reference, state, inventory) if shadow else current
    selected = dict(native_resources)
    selected["reconstruction_v1001"] = dict(native_resources["reconstruction_v1001"], fused_int4a4=proxy)
    changes = parent_factory(selected)
    current_audit = NativeLaunchAudit(current, Path(build_dir), logger.info)
    progress = PrefillProgress()
    execute_base = changes.get(WORKER_PREFIX + "execute_model", NPUWorker310.execute_model)
    execute_base = getattr(execute_base, "__prefill_progress_original__", execute_base)
    capture_base, status_base = (changes[WORKER_PREFIX + name] for name in ("resident_capture", "resident_status"))

    def execute(self, scheduler_output, *args, **kwargs):
        current_audit.begin_requests()
        rows = progress.scheduled(scheduler_output) if self.rank == 0 else []
        for row in rows:
            state["progress"].append(
                {
                    "request_id": row[0],
                    "prompt_tokens": row[1],
                    "reused_at_admission": row[2],
                    "prompt_chunk": row[3],
                    "scheduled_through": row[4],
                    "remaining_to_schedule": row[5],
                    "forward_returned": False,
                }
            )
            logger.info(
                "Prefill scheduled: request=%s prompt_tokens=%d reused_at_admission=%s prompt_chunk=%d "
                "scheduled_through=%d remaining_to_schedule=%d",
                *row,
            )
        result = execute_base(self, scheduler_output, *args, **kwargs)
        for event in state["progress"][-len(rows) :] if rows else []:
            event["forward_returned"] = True  # Enqueued/returned is NOT device completion.
        return result

    execute.__prefill_progress_original__ = execute_base

    def capture(self):
        try:
            wanted = expected_inventory[str(self.rank)]
            pointer_map = {row["pointer"]: dict(row, ordinal=i) for i, row in enumerate(wanted)}
            observed = {}
            for wrapper in self._resident_wrappers():
                for owner in wrapper.runnable.modules():
                    bank = getattr(owner, "w2_experts", None)
                    if bank is not None:
                        pointer = bank.gate_up_packed_bank.data_ptr()
                        hidden = bank.down_packed_bank.shape[1]
                        observed[pointer] = bank.gate_up_packed_bank.shape[-1] * 8 // hidden
            if set(observed) != set(pointer_map) or any(
                observed[pointer] != pointer_map[pointer]["bits"] for pointer in observed
            ):
                raise ValueError("loaded bank pointers/precisions differ from frozen reference")
            inventory.update(pointer_map)
            # Parent clears all retired scratch and then recaptures; shadow has
            # no bulk calls during decode capture, and retains the reference
            # resource until the untimed comparison finishes.
            return capture_base(self)
        except Exception as error:
            self._resident_session().graphs_dirty = True
            return self._resident_error(error)

    def status(self):
        row = status_base(self)
        row["fp16_scale_storage"]["prefill_version"] = version
        row["streaming"] = {
            "version": version,
            "decode_version": DECODE_VERSION,
            "shadow": shadow,
            "cases": list(state["cases"]),
            "whole_moe": list(state["whole_moe"]),
            "mismatches": state["mismatches"],
            "progress": list(state["progress"]),
            "profiler_active": bool(
                getattr(getattr(self, "profiler", None), "is_running", False)
                or getattr(getattr(self, "profiler", None), "_active", False)
            ),
            "bank_pointers": sorted(inventory),
            "native_launch_audit": current_audit.snapshot(),
        }
        return row

    changes[WORKER_PREFIX + "execute_model"] = execute
    changes[WORKER_PREFIX + "resident_capture"] = capture
    changes[WORKER_PREFIX + "resident_status"] = status
    if shadow:
        from vllm_ascend.models.glm5next_w2.moe import Glm5NextW2MoE

        changes[MOE_FORWARD] = wrap_moe_forward(Glm5NextW2MoE.forward, proxy, state)
    if profile_dir is not None:
        from vllm.config import ProfilerConfig

        original = getattr(NPUWorker310.profile, "__glm_profile_original__", NPUWorker310.profile)

        def profile(self, is_start=True, profile_prefix=None):
            try:
                if is_start:
                    previous = getattr(self, "profiler", None)
                    if getattr(previous, "is_running", False) or getattr(previous, "_active", False):
                        raise RuntimeError("another profiler is active; do not replace it")
                    # A stopped wrapper retains its first trace name/directory.
                    # Retire only that inactive wrapper before configuring this run.
                    self.profiler = None
                    self.profiler_config = ProfilerConfig(
                        profiler="torch",
                        torch_profiler_dir=profile_dir,
                        torch_profiler_record_shapes=True,
                        torch_profiler_with_memory=True,
                        torch_profiler_with_stack=False,
                        ignore_frontend=True,
                    )
                original(self, is_start, profile_prefix)
                state["profiler_active"] = bool(is_start)
                row = status(self)
                if row["streaming"]["profiler_active"] != bool(is_start):
                    raise RuntimeError("profiler wrapper did not acknowledge the requested state")
                return row
            except Exception as error:
                return self._resident_error(error)

        profile.__glm_profile_original__ = original
        changes[WORKER_PREFIX + "profile"] = profile
    return changes


def compose_source(
    frozen_source, *, source_sha256, version, build_dir, expected_inventory, shadow=False, profile_dir=None
):
    """Reuse the sealed fusion prefix; never execute its old queue/audit tail."""
    if hashlib.sha256(frozen_source.encode()).hexdigest() != source_sha256:
        raise ValueError("qualified replacement source changed")
    if frozen_source.count(SOURCE_BOUNDARY) != 1:
        raise ValueError("qualified source boundary changed; refuse partial composition")
    prefix = frozen_source.split(SOURCE_BOUNDARY, 1)[0]
    # Embed this exact helper in the replacement capsule so a later on-disk
    # package edit cannot silently alter the submitted replacement digest.
    helper = Path(inspect.getsourcefile(compose_source)).read_text()
    audit = Path(inspect.getsourcefile(NativeLaunchAudit)).read_text()
    progress = Path(inspect.getsourcefile(PrefillProgress)).read_text()
    for dependency in (
        "from tools.glm_perf.native_launch_audit import NativeLaunchAudit\n",
        "from tools.glm_perf.prefill_progress import PrefillProgress\n",
    ):
        if helper.count(dependency) != 1:
            raise ValueError("sealed helper dependency import changed")
        helper = helper.replace(dependency, "")

    def capsule(variable, stem, text):
        name = stem + "_" + hashlib.sha256(text.encode()).hexdigest()[:16]
        return (
            f"{variable}=types.ModuleType({name!r})\n"
            f"sys.modules[{variable}.__name__]={variable}\n"
            f"exec(compile({text!r},'<sealed-{stem}>','exec'),{variable}.__dict__)\n"
        )

    # Register the versioned progress capsule before executing its dataclass.
    # Workers do not import unversioned new files from their old tools package.
    dependencies = capsule("_streaming_audit", "glm_streaming_audit", audit)
    dependencies += capsule("_streaming_progress", "glm_streaming_progress", progress)
    return (
        prefix
        + "\n_streaming_parent=replacements\nimport types,sys\n"
        + dependencies
        + (
            "_streaming_helper=types.ModuleType('glm_streaming_candidate')\n"
            "_streaming_helper.__file__=" + repr(str(Path(__file__).resolve())) + "\n"
            "_streaming_helper.__dict__.update(NativeLaunchAudit=_streaming_audit.NativeLaunchAudit,"
            "PrefillProgress=_streaming_progress.PrefillProgress)\n"
            "exec(compile(" + repr(helper) + ",'<sealed-streaming-helper>','exec'),_streaming_helper.__dict__)\n"
            "def replacements(native_resources):\n"
            "    return _streaming_helper.extend_replacements(_streaming_parent,native_resources,"
            f"version={version!r},build_dir={str(build_dir)!r},expected_inventory={expected_inventory!r},"
            f"shadow={shadow!r},profile_dir={profile_dir!r})\n"
        )
    )
