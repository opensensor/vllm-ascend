# SPDX-License-Identifier: Apache-2.0
"""Dependency proof for v2 prefetch; explicit async execution, no device claims."""

from dataclasses import dataclass

from tools.qwen4exp.streaming_next_memory import MAX_GROUPS, METADATA_SLOTS, SLOTS


@dataclass(frozen=True)
class Command:
    name: tuple
    dependencies: frozenset
    reads: frozenset
    writes: frozenset


def commands(groups, metadata_slots=METADATA_SLOTS):
    if type(groups) is not int or not 1 <= groups <= MAX_GROUPS or metadata_slots not in (2, 3):
        raise ValueError("invalid v2 schedule geometry")
    result = []
    for g in range(groups):
        slot = g % SLOTS
        meta = g % metadata_slots
        dep = set()
        if g >= SLOTS:
            dep.add(("readback", g - SLOTS))
        if g >= metadata_slots:
            dep.add(("consume", g - metadata_slots))
        result.append(
            Command(
                ("produce", g),
                frozenset(dep),
                frozenset({"resident_weight"}),
                frozenset({("operand", slot), ("metadata", meta)}),
            )
        )
        dep = {("produce", g)}
        if g:
            dep.add(("readback", g - 1))
        result.append(Command(("cube", g), frozenset(dep), frozenset({("operand", slot)}), frozenset({"CO1"})))
        dep = {("cube", g)}
        if g >= SLOTS:
            dep.add(("consume", g - SLOTS))
        result.append(Command(("readback", g), frozenset(dep), frozenset({"CO1"}), frozenset({("raw", slot)})))
        dep = {("readback", g)}
        if g:
            dep.add(("consume", g - 1))
        result.append(
            Command(
                ("consume", g),
                frozenset(dep),
                frozenset({("raw", slot), ("metadata", meta), "weight_metadata"}),
                frozenset({"accumulator", ("float", slot)}),
            )
        )
    return tuple(result)


def submission_order(groups):
    order = [("produce", 0)]
    for g in range(groups):
        order.append(("cube", g))
        if g + 1 < groups:
            order.append(("produce", g + 1))
        if g:
            order.append(("consume", g - 1))
        order.append(("readback", g))
    return (*order, ("consume", groups - 1))


def validate_prefetch(groups, metadata_slots=METADATA_SLOTS):
    """Prove source submission can prefetch without waiting on future consumers.

    Resource access lifetimes span asynchronous command completion. Returned DAG
    supports arbitrarily delayed commands; every conflicting pair must have a
    dependency path. Submission is separately checked for impossible wait order.
    """
    graph = {c.name: c for c in commands(groups, metadata_slots)}
    order = submission_order(groups)
    position = {name: i for i, name in enumerate(order)}
    ancestors = {}

    def before(name):
        if name not in ancestors:
            ancestors[name] = set(graph[name].dependencies)
            for dep in graph[name].dependencies:
                if position[dep] >= position[name]:
                    raise ValueError("prefetch depends on a consumer not yet submitted")
                ancestors[name].update(before(dep))
        return ancestors[name]

    for name in order:
        before(name)
    for i, left in enumerate(order):
        a = graph[left]
        for right in order[i + 1 :]:
            b = graph[right]
            conflict = (a.writes & (b.reads | b.writes)) | (b.writes & a.reads)
            if conflict and left not in ancestors[right] and right not in ancestors[left]:
                raise ValueError(f"unordered resource access: {left}, {right}, {conflict}")
    return tuple(graph[name] for name in order)
