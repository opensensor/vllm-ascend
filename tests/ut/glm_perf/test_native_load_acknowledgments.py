# SPDX-License-Identifier: Apache-2.0
"""A matching preparation digest does not acknowledge device validation."""

import pytest

from tools.glm_perf.resident_harness import ResidentClient
from tools.glm_perf.resident_native import NativeManifest


@pytest.mark.parametrize("malformed", [False, True])
def test_resume_confirms_all_status_replies_before_resuming(malformed):
    class Client(ResidentClient):
        def __init__(self):
            super().__init__("http://localhost", expected_workers=2)
            self.calls = []

        def request(self, path, payload=None, method="POST"):
            operation = payload["method"] if payload else path
            self.calls.append(operation)
            if path == "/resume":
                return {"status": "resumed"}
            if len(self.calls) == 1:
                return {"results": [{}, None if malformed else {}]}
            return {
                "results": [
                    {
                        "rank": rank,
                        "weight_storage_digest": str(rank),
                        "graphs_dirty": False,
                        "generation": "current",
                        "mode": "graph",
                        "candidate": "current",
                        "digest": "current",
                    }
                    for rank in range(2)
                ]
            }

    client = Client()
    assert client.resume() == {"status": "resumed"}
    assert client.calls == ["resident_status", "resident_status", "/resume"]


@pytest.mark.parametrize("stale_reply", ["prepared", "unvalidated_status", "malformed"])
def test_native_load_confirms_validation_without_repeating_mutation(stale_reply):
    manifest = NativeManifest(
        {
            "name": "native_v1",
            "libraries": [{"path": "/test.so", "sha256": "0" * 64}],
            "assets": [],
            "operators": ["native_v1::launch"],
            "validation_source": "def validate(): return {'passed': True}",
        }
    )

    class Client(ResidentClient):
        def __init__(self):
            super().__init__("http://localhost", expected_workers=2)
            self.paused = False
            self.loaded = False
            self.probed_validation = False
            self.calls = []

        def request(self, path, payload=None, method="POST"):
            operation = payload["method"] if payload else path
            self.calls.append(operation)
            if path == "/is_paused":
                return {"is_paused": self.paused}
            if path.startswith("/pause"):
                self.paused = True
                return {"status": "paused"}
            if path == "/resume":
                self.paused = False
                return {"status": "resumed"}
            prepared = {"native_digest": manifest.digest}
            validated = {**prepared, "validation": {"passed": True}}
            if operation == "resident_native_prepare":
                return {"results": [{"rank": rank, **prepared} for rank in range(2)]}
            if operation == "resident_native_load":
                self.loaded = True
                if stale_reply == "malformed":
                    return {"results": [validated, None]}
                if stale_reply == "prepared":
                    return {"results": [validated, prepared]}
                return {
                    "results": [
                        {"rank": rank, "native_loaded": {manifest.name: prepared}, "native_failed": False}
                        for rank in range(2)
                    ]
                }
            assert operation == "resident_status"
            if self.loaded:
                self.probed_validation = True
            return {
                "results": [
                    {
                        "rank": rank,
                        "pid": rank + 100,
                        "weight_storage_digest": str(rank),
                        "graphs_dirty": False,
                        "generation": "current",
                        "digest": "current",
                        "mode": "graph",
                        "candidate": "current",
                        "native_failed": False,
                        "native_loaded": {manifest.name: validated} if self.loaded else {},
                    }
                    for rank in range(2)
                ]
            }

    client = Client()
    receipts = client.load_native(manifest)
    assert all(receipt["validation"]["passed"] for receipt in receipts)
    assert client.probed_validation
    assert client.calls.count("resident_native_load") == 1
    assert not client.paused
