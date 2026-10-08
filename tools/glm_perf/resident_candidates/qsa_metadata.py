# SPDX-License-Identifier: Apache-2.0
"""Compose fused attention metadata with existing resident capture policies."""

from ..instance_bindings import InstanceBindings
from ..qsa_metadata import geometry
from ..qsa_metadata_binding import bind_attention
from ..resident_rpc_guard import WORKER_PREFIX, guard_replacements

MAX_PREFILL_ROWS = 640
MAX_REQUESTS = 4
POOL_SIZE = 4
PHYSICAL_BLOCK_SIZE = 32
LOGICAL_BLOCK_SIZE = 640


class PreparedMetadata:
    """Adapt earlier frozen helpers without changing their code or descriptors."""

    def __init__(self, native, geometry_function):
        self.native, self.geometry_function = native, geometry_function
        self.fallbacks = 0

    def __getattr__(self, name):
        return getattr(self.native, name)

    def available(self, ids, positions, table, **options):
        if any(value.device != self.device for value in (ids, positions, table)):
            return False
        try:
            case = self.geometry_function(ids, positions, table, **options)
        except ValueError:
            return False
        return case in self.configs


def prepare_runner(native, runner, geometry_type):
    """Prepare descriptors from host shapes before target and draft capture."""
    roots = [runner.model]
    draft = getattr(getattr(runner, "drafter", None), "model", None)
    if draft is not None:
        roots.append(draft)
    tables = [
        table.block_table.gpu
        for table in runner.input_batch.block_table.block_tables
        if not table.is_mamba_group and table.block_size == PHYSICAL_BLOCK_SIZE
    ]
    if not tables:
        raise ValueError("no qualified sparse-attention block table found")
    cases, seen = [], set()
    for root in roots:
        for module in root.modules():
            owner = getattr(module, "impl", None)
            indexer = getattr(owner, "glm_indexer", None)
            if owner is None or indexer is None or id(owner) in seen or not hasattr(owner, "_forward_decode_fused"):
                continue
            seen.add(id(owner))
            if indexer.index_kpool != POOL_SIZE:
                raise ValueError("only pool size four is qualified for fused metadata")
            ids = indexer.topk_indices_buffer
            budget = indexer.topk_tokens // POOL_SIZE
            for table in tables:
                for rows in range(1, min(ids.shape[0], MAX_PREFILL_ROWS) + 1):
                    for requests in range(1, min(table.shape[0], MAX_REQUESTS) + 1):
                        for position_bytes in (4, 8):
                            cases.append(
                                geometry_type(
                                    rows,
                                    requests,
                                    budget,
                                    0,
                                    *ids.stride(),
                                    1,
                                    position_bytes,
                                    table.shape[1],
                                    *table.stride(),
                                    LOGICAL_BLOCK_SIZE // PHYSICAL_BLOCK_SIZE,
                                )
                            )
    if not seen:
        raise ValueError("no loaded GLM sparse-attention instance found")
    native.prepare(cases)
    return [dict(shape=list(table.shape), stride=list(table.stride())) for table in tables]


def extend_replacements(changes, native, geometry_type, *, geometry_function=geometry):
    """Preserve parent hooks and restore per-instance bindings on transition."""
    bindings = InstanceBindings()
    native = PreparedMetadata(native, geometry_function)
    state = {"attention_instances": 0, "prepared_tables": []}
    capture_base, apply_base, status_base = (
        changes[WORKER_PREFIX + name] for name in ("resident_capture", "resident_apply", "resident_status")
    )

    def capture(self):
        try:
            state["prepared_tables"] = prepare_runner(native, self.model_runner, geometry_type)
            state["attention_instances"] = bind_attention(bindings, self.model_runner, native)
            result = capture_base(self)
            if "error" in result:
                bindings.restore()
            return result
        except Exception as error:
            bindings.restore()
            self._resident_session().graphs_dirty = True
            return self._resident_error(error)

    def apply(self, generation):
        bindings.restore()
        return apply_base(self, generation)

    def status(self):
        receipt = status_base(self)
        receipt["fused_qsa_metadata"] = dict(
            state,
            calls=native.calls,
            descriptors=len(native.configs),
            metadata_fallbacks=native.fallbacks,
            capture_geometry=dict(native.geometries),
            instance_methods=len(bindings.originals),
            prepared_outside_capture=True,
        )
        return receipt

    result = dict(changes)
    for name, function in (("resident_capture", capture), ("resident_apply", apply), ("resident_status", status)):
        result[WORKER_PREFIX + name] = function
    return guard_replacements(result)
