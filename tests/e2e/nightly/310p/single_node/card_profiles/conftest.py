# SPDX-License-Identifier: Apache-2.0
"""Explicit opt-in to a dedicated, already started GLM profile server."""

import json
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from tools.glm_perf.deployment_profiles import profile


def pytest_addoption(parser):
    parser.addoption("--glm-profile-url", default=None, help="Dedicated GLM server; these tests submit real requests")
    parser.addoption("--glm-profile-cards", type=int, choices=(2, 3), default=3)
    parser.addoption("--glm-profile-process-receipt", help="Reviewed launch receipt when dev server_info is disabled")


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

    try:
        info = call("/server_info?config_format=json")["vllm_config"]
    except urllib.error.HTTPError as error:
        receipt_path = request.config.getoption("--glm-profile-process-receipt")
        if error.code != 404 or not receipt_path:
            raise
        receipt = json.loads(Path(receipt_path).read_text())
        command = receipt["command"]
        assert command[command.index("--served-model-name") + 1] == selected.name
        assert command[command.index("--load-format") + 1] == "glm_native_int4"
        # The native loader admits this precision profile only with its matching
        # topology/MTP. Combine the reviewed CLI with the live served identity.
        info = {
            "parallel_config": {
                "tensor_parallel_size": int(command[command.index("--tensor-parallel-size") + 1]),
                "pipeline_parallel_size": int(command[command.index("--pipeline-parallel-size") + 1]),
            },
            "scheduler_config": {"max_num_batched_tokens": int(command[command.index("--max-num-batched-tokens") + 1])},
            "speculative_config": (
                json.loads(command[command.index("--speculative-config") + 1])
                if "--speculative-config" in command
                else None
            ),
        }
    assert info["parallel_config"]["tensor_parallel_size"] == selected.tensor_parallel_size
    assert info["parallel_config"]["pipeline_parallel_size"] == selected.pipeline_parallel_size
    assert info["scheduler_config"]["max_num_batched_tokens"] == 640
    if selected.mtp_tokens:
        assert info["speculative_config"]["num_speculative_tokens"] == selected.mtp_tokens
    else:
        assert info.get("speculative_config") is None
    assert selected.name in {row["id"] for row in call("/v1/models")["data"]}
    return selected, call
