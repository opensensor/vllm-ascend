# SPDX-License-Identifier: Apache-2.0
"""Explicit opt-in paths; ordinary test collection never opens an NPU."""


def pytest_addoption(parser):
    group = parser.getgroup("GLM channel-scale candidate")
    group.addoption("--glm-channel-overlay", default=None)
    group.addoption("--glm-channel-bundle", default=None)
    group.addoption("--glm-channel-device", type=int, default=0)
