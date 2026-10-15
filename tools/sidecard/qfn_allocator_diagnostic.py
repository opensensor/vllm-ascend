# SPDX-License-Identifier: Apache-2.0
"""Read allocator counters in an existing, drained resident worker context.

No allocation, cache release, synchronization, submission, or owner state change
is performed. Unsupported SDK memory categories are reported with their status.
"""

import ctypes as c

# Installed acl/acl_rt.h enum values; these query DDR, not HBM categories.
DDR_MEMORY_CATEGORIES = ((2, "ddr_huge"), (3, "ddr_normal"), (6, "ddr_p2p_huge"), (7, "ddr_p2p_normal"))
P2P_MEMORY_CATEGORIES = DDR_MEMORY_CATEGORIES[2:]


def _memory_query(owner):
    # A separate callable avoids modifying argtypes/restype on shared bindings.
    prototype = c.CFUNCTYPE(c.c_int, c.c_int, c.POINTER(c.c_size_t), c.POINTER(c.c_size_t))
    function = prototype(("aclrtGetMemInfo", owner.acl))

    def query(attribute):
        free, total = c.c_size_t(), c.c_size_t()
        status = int(function(attribute, c.byref(free), c.byref(total)))
        return dict(
            status=status,
            free_bytes=free.value if status == 0 else None,
            total_bytes=total.value if status == 0 else None,
        )

    return query


def inspect_allocator(live, *, query_factory=_memory_query):
    if live.state != "active" or not live.owners:
        raise RuntimeError("allocator diagnostic requires an existing active collective")
    live._context()
    owner = next(iter(live.owners.values()))
    before = [(key, id(item), item.state, item.calls) for key, item in live.owners.items()]
    query = query_factory(owner)
    pools = []
    for attribute, name in DDR_MEMORY_CATEGORIES:
        try:
            record = query(attribute)
        except Exception as error:
            record = dict(error=f"{type(error).__name__}: {error}")
        pools.append(dict(attribute=attribute, name=name, **record))
    stats = {}
    for name in ("memory_allocated", "memory_reserved", "memory_stats"):
        try:
            stats[name] = getattr(live.torch.npu, name)(live.device_index)
        except Exception as error:
            stats[name] = dict(error=f"{type(error).__name__}: {error}")
    if before != [(key, id(item), item.state, item.calls) for key, item in live.owners.items()]:
        raise RuntimeError("serving owner identity or counters changed during read-only diagnostic")
    return dict(
        rank=live.rank,
        physical_device=live.device_index,
        collective_identity=live.identity,
        pools=pools,
        torch=stats,
        serving_owner_state_unchanged=True,
        interpretation="pool totals are not largest allocatable block or an allocation guarantee",
    )


def read_p2p_pools(live, *, query_factory=_memory_query):
    """Cold admission snapshot from the existing SDK owner, without allocation."""
    if live.state != "active" or not live.owners:
        raise RuntimeError("P2P admission requires an existing active collective")
    live._context()
    query = query_factory(next(iter(live.owners.values())))
    return [dict(attribute=attribute, name=name, **query(attribute)) for attribute, name in P2P_MEMORY_CATEGORIES]
