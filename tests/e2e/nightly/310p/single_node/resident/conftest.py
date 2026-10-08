import pytest

from tools.glm_perf.resident_harness import ResidentClient


def pytest_addoption(parser):
    parser.addoption("--resident-glm-url", default=None, help="opt in to modifying this resident diagnostic server")
    parser.addoption("--resident-glm-workers", type=int, default=4)


@pytest.fixture
def resident_client(request):
    url = request.config.getoption("--resident-glm-url")
    if url is None:
        pytest.skip("pass --resident-glm-url to qualify an already loaded GLM server")
    client = ResidentClient(url, expected_workers=request.config.getoption("--resident-glm-workers"))
    if client.request("/is_paused", method="GET")["is_paused"]:
        pytest.fail("resume the diagnostic server before qualification")
    return client
