"""Record final engine, worker-owner, graph, and health status without resetting owners."""

import hashlib
import json
import subprocess
import urllib.request
import uuid
from pathlib import Path

from tools.sidecard.qfn_resident_server import QfnResidentClient

ROOT = Path("/srv/ai/storage/artifacts/npu-collective-qfn-20261014-024609")
SOURCE = """
def inspect_worker(worker):
    import ctypes
    import inspect
    import torch
    acl=ctypes.CDLL('libascendcl.so')
    capabilities=[]
    if hasattr(acl,'aclrtHostMemMapCapabilities'):
        acl.aclrtHostMemMapCapabilities.argtypes=[ctypes.c_uint32,ctypes.c_int,ctypes.POINTER(ctypes.c_int)]
        acl.aclrtHostMemMapCapabilities.restype=ctypes.c_int
        for name,kind in [('aicpu',1),('aicore',2),('aivector',3),('pciedma',4),('sdma',6)]:
            value=ctypes.c_int(-1)
            status=acl.aclrtHostMemMapCapabilities(torch.npu.current_device(),kind,ctypes.byref(value))
            capabilities.append(dict(engine=name,status=status,value=value.value))
    modules=[]
    for name,module in worker.model_runner.model.named_modules():
        if type(module).__name__=='AscendQwen4ExpNGramEmbedding':
            table=module.ple_method
            modules.append(dict(name=name,ngram_size=module.ngram_size,heads=module.ngram_heads,
                head_dim=module.head_dim,eos=module.eos_token_id,table_type=type(table).__name__,
                table_rows=table.num_embeddings,table_bytes=table.table_bytes,
                shard_rows=getattr(table,'shard_rows',None),split_parts=getattr(table,'_split_ngram_parts',None),
                checkpoint=str(getattr(table,'_checkpoint_dir','')),source=inspect.getfile(type(module)),
                multipliers=module._host_multipliers,sizes=module._host_vocab_sizes,offsets=module._host_vocab_offsets))
    return dict(rank=worker.rank,passed=True,capabilities=capabilities,modules=modules)
class Candidate:
    def validate(self,engine):
        if not engine._pause_complete() or engine.batch_queue or engine.scheduler.get_num_unfinished_requests():
            raise RuntimeError('drain before inspection')
        return dict(passed=True)
    def apply(self,engine):
        self.validate(engine)
        return dict(passed=True,workers=engine.model_executor.collective_rpc(inspect_worker))
def prepare(engine): return Candidate()
"""


def main():
    output = ROOT / "resident-live-v7/ple-lookup-capabilities-v1.json"
    if output.exists():
        raise FileExistsError(output)
    client = QfnResidentClient("http://127.0.0.1:8001", ROOT / "resident-live-v7/control.token")
    client.request("/pause?mode=wait&clear_cache=false")
    revision, digest = "status_" + uuid.uuid4().hex, hashlib.sha256(SOURCE.encode()).hexdigest()
    prepared = client.request(
        "/qfn_engine_control", dict(operation="prepare", revision=revision, sha256=digest, source=SOURCE)
    )
    assert prepared.get("passed"), prepared
    result = dict(
        owners=client.request("/qfn_engine_control", dict(operation="apply", revision=revision, sha256=digest))
    )
    output.write_text(json.dumps(result, indent=2) + "\n")
    assert result["owners"].get("passed"), result
    owners = result["owners"]["applied"]["workers"]
    assert len(owners) == 4
    result["workers"] = client.rpc("resident_status")
    assert all(not row["graphs_dirty"] and not row["native_failed"] for row in result["workers"])
    result["graphs"] = client.rpc("qfn_graph_inventory")
    client.resume()
    result["engine"] = client.request("/qfn_engine_control", dict(operation="status"))
    result["paused"] = client.request("/is_paused", method="GET")
    with urllib.request.urlopen(client.base_url + "/health", timeout=10) as response:
        result["health"] = response.status
    result["sensors"] = subprocess.run(
        ["npu-smi", "info"], capture_output=True, text=True, check=True, timeout=20
    ).stdout
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            dict(engine=result["engine"], paused=result["paused"], health=result["health"], owners=owners), indent=2
        )
    )


if __name__ == "__main__":
    main()
