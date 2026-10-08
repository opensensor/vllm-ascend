# SPDX-License-Identifier: Apache-2.0


def pytest_addoption(parser):
    parser.addoption(
        "--glm-qsa-metadata-build", default=None, help="explicit append-only fused metadata build directory"
    )
