# SPDX-License-Identifier: Apache-2.0
"""Read-only pool diagnostics require no Torch import or device initialization."""

from types import SimpleNamespace

import pytest

from tools.sidecard import qfn_allocator_diagnostic as diagnostic


def live_fixture():
    calls = []
    owner = SimpleNamespace(state="active", calls=7)
    npu = SimpleNamespace(
        memory_allocated=lambda device: calls.append(("allocated", device)) or 100,
        memory_reserved=lambda device: calls.append(("reserved", device)) or 200,
        memory_stats=lambda device: calls.append(("stats", device)) or {"inactive_split_bytes.all.current": 50},
    )
    live = SimpleNamespace(
        state="active",
        owners={("float16", 2560): owner},
        rank=2,
        device_index=2,
        identity="unchanged",
        _context=lambda: calls.append("context"),
        torch=SimpleNamespace(npu=npu),
    )
    return live, calls


def test_reads_exact_ddr_categories_and_torch_counters_without_mutation():
    live, calls = live_fixture()

    def factory(owner):
        assert owner is next(iter(live.owners.values()))

        def query(attribute):
            calls.append(("pool", attribute))
            return dict(status=0, free_bytes=10, total_bytes=20)

        return query

    result = diagnostic.inspect_allocator(live, query_factory=factory)
    assert calls == [
        "context",
        ("pool", 2),
        ("pool", 3),
        ("pool", 6),
        ("pool", 7),
        ("allocated", 2),
        ("reserved", 2),
        ("stats", 2),
    ]
    assert result["rank"] == 2 and result["serving_owner_state_unchanged"]
    assert result["torch"]["memory_stats"]["inactive_split_bytes.all.current"] == 50
    assert next(iter(live.owners.values())).calls == 7


def test_preserves_unsupported_sdk_status_and_per_query_exception():
    live, _ = live_fixture()

    def query(attribute):
        if attribute == 6:
            raise RuntimeError("unsupported pool")
        return dict(status=207000, free_bytes=None, total_bytes=None)

    result = diagnostic.inspect_allocator(live, query_factory=lambda owner: query)
    assert len(result["pools"]) == 4
    assert result["pools"][0]["status"] == 207000
    assert result["pools"][2]["error"] == "RuntimeError: unsupported pool"
    assert result["pools"][3]["free_bytes"] is None


def test_detects_serving_owner_change_and_never_repairs_it():
    live, _ = live_fixture()

    def query(attribute):
        next(iter(live.owners.values())).calls += 1
        return dict(status=0)

    with pytest.raises(RuntimeError, match="identity or counters changed"):
        diagnostic.inspect_allocator(live, query_factory=lambda owner: query)
    assert next(iter(live.owners.values())).calls == 11


def test_rejects_inactive_collective_before_any_query():
    live, calls = live_fixture()
    live.state = "quarantined"
    with pytest.raises(RuntimeError, match="existing active"):
        diagnostic.inspect_allocator(live, query_factory=lambda owner: pytest.fail("unexpected binding"))
    assert calls == []


def test_admission_snapshot_queries_only_p2p_without_torch_stats():
    live, calls = live_fixture()

    def query(attribute):
        calls.append(attribute)
        return dict(status=0, free_bytes=0, total_bytes=100)

    result = diagnostic.read_p2p_pools(live, query_factory=lambda owner: query)
    assert calls == ["context", 6, 7]
    assert [item["name"] for item in result] == ["ddr_p2p_huge", "ddr_p2p_normal"]


def test_binding_uses_existing_dso_and_suppresses_failed_output_values(monkeypatch):
    acl = object()

    def prototype(*types):
        def bind(symbol):
            assert symbol == ("aclrtGetMemInfo", acl)

            def query(attribute, free, total):
                free._obj.value, total._obj.value = 10, 20
                return 207000 if attribute == 6 else 0

            return query

        return bind

    monkeypatch.setattr(diagnostic.c, "CFUNCTYPE", prototype)
    query = diagnostic._memory_query(SimpleNamespace(acl=acl))
    assert query(2) == dict(status=0, free_bytes=10, total_bytes=20)
    assert query(6) == dict(status=207000, free_bytes=None, total_bytes=None)
