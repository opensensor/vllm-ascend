# SPDX-License-Identifier: Apache-2.0
"""Explicit opt-ins for sidecard hardware and completed-generation gates."""


def pytest_addoption(parser):
    parser.addoption(
        "--sidecard-stage-specs",
        nargs=2,
        default=None,
        metavar=("SOCKET_SPEC", "DMA_SPEC"),
        help="Fresh complete-stage specifications on idle assigned NPU chips 2,3 and CUDA device 0",
    )
    parser.addoption(
        "--sidecard-execution-manifest",
        default=None,
        help="Authenticated execution gate manifest granting exclusive chips 2,3 and CUDA 0 use",
    )
    parser.addoption("--sidecard-acceptance-url", default=None, help="Explicit completed-generation service URL")
    parser.addoption(
        "--sidecard-acceptance-fixtures", default=None, help="Private completed-generation fixture manifest"
    )
    parser.addoption("--sidecard-acceptance-output", default=None, help="Completed-generation receipt directory")


def pytest_configure(config):
    names = ("--sidecard-acceptance-url", "--sidecard-acceptance-fixtures", "--sidecard-acceptance-output")
    supplied = [config.getoption(name) is not None for name in names]
    if any(supplied) and not all(supplied):
        import pytest

        raise pytest.UsageError("sidecard acceptance requires URL, fixtures and output options together")
