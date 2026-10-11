# SPDX-License-Identifier: Apache-2.0
"""Explicit opt-in paths; ordinary test collection never opens an NPU."""


def pytest_addoption(parser):
    group = parser.getgroup("GLM channel-scale candidate")
    group.addoption("--glm-channel-overlay", default=None)
    group.addoption("--glm-channel-bundle", default=None)
    group.addoption("--glm-channel-device", type=int, default=0)
    owner = parser.getgroup("GLM owned indexer transport")
    owner.addoption(
        "--glm-indexer-owner-hccl", action="store_true", help="Requires exclusive reservation of all test chips"
    )
    owner.addoption("--glm-indexer-owner-tp-size", type=int, choices=(2, 4), default=2)
    owner.addoption("--glm-indexer-owner-rank", type=int, default=1)
    owner.addoption("--glm-indexer-overlap-prefill", action="store_true")
    group.addoption("--glm-next-math-cpu-transport", action="store_true", help="Run synthetic Gloo TP2/TP4 gates")
