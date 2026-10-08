# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import pytest

from tools.glm_perf import worker_affinity as affinity


@pytest.mark.parametrize("value,expected", [("0", {0}), ("2-4,8,4", {2, 3, 4, 8})])
def test_cpu_list(value, expected):
    assert affinity.parse_cpu_list(value) == expected


@pytest.mark.parametrize("value", ["", "-1", "4-2", "a", "1,", "1-2-3", "65536"])
def test_bad_cpu_list(value):
    with pytest.raises(ValueError):
        affinity.parse_cpu_list(value)


def mock_host(monkeypatch):
    masks = {0: set(range(8)), 100: set(range(8)), 200: set(range(8)), 201: {2, 3}}
    identities = {100: 1.0, 200: 2.0}

    def process(pid):
        return SimpleNamespace(
            pid=pid,
            name=lambda: f"process-{pid}",
            create_time=lambda: identities[pid],
            children=lambda recursive: [SimpleNamespace(pid=200)],
            threads=lambda: [SimpleNamespace(id=tid) for tid in ([100] if pid == 100 else [200, 201])],
        )

    monkeypatch.setattr(affinity.psutil, "Process", process)
    monkeypatch.setattr(affinity.os, "sched_getaffinity", lambda tid: set(masks[tid]))
    monkeypatch.setattr(affinity.os, "sched_setaffinity", lambda tid, cpus: masks.__setitem__(tid, set(cpus)))
    return masks, identities


def test_plan_does_not_mutate_and_apply_restores_every_thread(monkeypatch):
    masks, _ = mock_host(monkeypatch)
    before = {key: set(value) for key, value in masks.items()}
    plan = affinity.plan_bindings(100, {200: {4, 5}})
    assert masks == before
    affinity.apply_bindings(plan)
    assert masks[200] == masks[201] == {4, 5}
    assert masks[100] == before[100]
    affinity.apply_bindings(plan, restore=True)
    assert masks == before


def test_recycled_pid_rejected_before_any_write(monkeypatch):
    masks, identities = mock_host(monkeypatch)
    plan = affinity.plan_bindings(100, {100: {0}, 200: {4}})
    identities[200] = 3.0
    with pytest.raises(RuntimeError, match="recycled"):
        affinity.apply_bindings(plan)
    assert masks[100] == masks[200] == set(range(8))


@pytest.mark.parametrize("bindings", [{300: {1}}, {200: {99}}, {200: set()}, {}])
def test_unrelated_process_or_unavailable_cpus_rejected(monkeypatch, bindings):
    mock_host(monkeypatch)
    with pytest.raises(ValueError):
        affinity.plan_bindings(100, bindings)
