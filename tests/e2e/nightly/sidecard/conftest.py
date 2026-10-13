# SPDX-License-Identifier: Apache-2.0
"""An explicit specification pair grants this test exclusive accelerator use."""


def pytest_addoption(parser):
    parser.addoption(
        "--sidecard-stage-specs",
        nargs=2,
        default=None,
        metavar=("SOCKET_SPEC", "DMA_SPEC"),
        help="Fresh complete-stage specifications on idle assigned NPU chips 2,3 and CUDA device 0",
    )
