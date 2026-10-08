# SPDX-License-Identifier: Apache-2.0
"""Replace only KDA prefill's cache reads/writes; retain its exact arithmetic."""

import ast
import inspect
import textwrap


def wrap_prefill(original, native):
    original = getattr(original, "__glm_state_rows_original__", original)
    tree = ast.parse(textwrap.dedent(inspect.getsource(original)))

    class ReplaceCopies(ast.NodeTransformer):
        def __init__(self):
            self.reads = self.writes = 0

        def visit_Call(self, node):
            node = self.generic_visit(node)
            if isinstance(node.func, ast.Name) and node.func.id == "_prefill_initial_state":
                self.reads += 1
                gather = ast.Call(
                    ast.Attribute(ast.Name("_resident_state_rows", ast.Load()), "gather", ast.Load()),
                    node.args,
                    node.keywords,
                )
                return ast.copy_location(ast.Call(ast.Attribute(gather, "float", ast.Load()), [], []), node)
            return node

        def visit_Assign(self, node):
            node = self.generic_visit(node)
            if len(node.targets) == 1 and isinstance(node.targets[0], ast.Subscript):
                target = node.targets[0]
                if (
                    isinstance(target.value, ast.Name)
                    and target.value.id == "recurrent_state"
                    and isinstance(target.slice, ast.Name)
                    and target.slice.id == "state_indices"
                ):
                    self.writes += 1
                    call = ast.Call(
                        ast.Attribute(ast.Name("_resident_state_rows", ast.Load()), "scatter", ast.Load()),
                        [ast.Name("recurrent_state", ast.Load()), ast.Name("state_indices", ast.Load()), node.value],
                        [],
                    )
                    return ast.copy_location(ast.Expr(call), node)
            return node

    transform = ReplaceCopies()
    tree = transform.visit(tree)
    if (transform.reads, transform.writes) != (1, 1):
        raise ValueError("KDA prefill cache access changed; refusing a partial replacement")
    namespace = dict(original.__globals__, _resident_state_rows=native)
    exec(compile(ast.fix_missing_locations(tree), "<glm-kda-selected-state-rows>", "exec"), namespace)
    wrapped = namespace[original.__name__]
    slot_position = list(inspect.signature(original).parameters).index("state_indices")
    prefill_limit = native.MAX_SELECTED_ROWS

    def bounded(*args, **kwargs):
        slots = kwargs.get("state_indices")
        if slots is None:
            slots = args[slot_position]
        # FULL-decode dummy batches may exceed the serving prefill limit.
        # Preserve their original graph path; real prefills use prepared copies.
        if slots.numel() > prefill_limit:
            return original(*args, **kwargs)
        return wrapped(*args, **kwargs)

    bounded.__glm_state_rows_original__ = original
    return bounded


def prepare_models(native, runner):
    roots = [runner.model]
    drafter = getattr(runner, "drafter", None)
    if drafter is not None:
        draft_model = getattr(drafter, "model", None)
        if draft_model is not None:
            roots.append(draft_model)
    count = 0
    for root in roots:
        for module in root.modules():
            if not hasattr(module, "A_log") or not hasattr(module, "kv_cache"):
                continue
            cache = module.kv_cache
            if isinstance(cache, (tuple, list)) and len(cache) == 2 and cache[1].ndim == 4:
                native.prepare(cache[1])
                count += 1
    if not count:
        raise ValueError("no bound KDA cache found before selected-row capture")
    return count


def extend_replacements(changes, native):
    from vllm_ascend.models.glm5next_w2 import kda_310

    result = dict(changes)
    result["vllm_ascend.models.glm5next_w2.kda_310:_run_prefill"] = wrap_prefill(kda_310._run_prefill, native)
    prefix = "vllm_ascend._310p.worker_310p:NPUWorker310."
    capture_base, status_base = result[prefix + "resident_capture"], result[prefix + "resident_status"]

    def capture(self):
        # The first resident_apply enters the previously installed hook. The
        # new capture hook runs after activation, before any dummy graph uses
        # model cache geometry. Preparing in apply misses that first activation.
        try:
            prepare_models(native, self.model_runner)
            return capture_base(self)
        except Exception as error:
            return self._resident_error(error)

    def status(self):
        receipt = status_base(self)
        geometries = sorted({(g.shape, g.stride, g.payload, g.span) for g, _, _ in native.configs})
        receipt["kda_selected_state_rows"] = {
            "gathers": native.gathers,
            "scatters": native.scatters,
            "geometries": geometries,
            "whole_bank_materialization": False,
        }
        return receipt

    capture.__dict__.update(getattr(capture_base, "__dict__", {}))
    status.__dict__.update(getattr(status_base, "__dict__", {}))
    result[prefix + "resident_capture"], result[prefix + "resident_status"] = capture, status
    return result
