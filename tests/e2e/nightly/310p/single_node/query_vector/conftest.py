# SPDX-License-Identifier: Apache-2.0


def pytest_addoption(parser):
    parser.addoption("--glm-query-vector-build", default=None, help="explicit append-only vector converter build")
