# SPDX-License-Identifier: Apache-2.0
"""Explicit opt-in to a dedicated, already started GLM profile server."""

import json
import urllib.request

import pytest

from tools.glm_perf.deployment_profiles import profile


def pytest_addoption(parser):
    parser.addoption("--glm-profile-url", default=None, help="Dedicated GLM server; these tests submit real requests")
    parser.addoption("--glm-profile-cards", type=int, choices=(2, 3), default=3)


@pytest.fixture
def glm_profile_server(request):
    url = request.config.getoption("--glm-profile-url")
    if not url:
        pytest.skip("hardware qualification requires --glm-profile-url")
    selected = profile(request.config.getoption("--glm-profile-cards"))

    def call(path, body=None):
        payload = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(url.rstrip("/") + path, data=payload, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=300) as response:
            return json.loads(response.read())

    info = call("/server_info?config_format=json")["vllm_config"]
    assert info["parallel_config"]["tensor_parallel_size"] == selected.tensor_parallel_size
    assert info["parallel_config"]["pipeline_parallel_size"] == selected.pipeline_parallel_size
    assert info["scheduler_config"]["max_num_batched_tokens"] == 640
    if selected.mtp_tokens:
        assert info["speculative_config"]["num_speculative_tokens"] == selected.mtp_tokens
    else:
        assert info.get("speculative_config") is None
    assert selected.name in {row["id"] for row in call("/v1/models")["data"]}
    return selected, call
