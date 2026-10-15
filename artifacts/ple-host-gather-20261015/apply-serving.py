import argparse
import hashlib
import json
import uuid
from pathlib import Path

from tools.glm_perf.resident_control import Control
from tools.sidecard.qfn_resident_server import QfnResidentClient

root = Path('/srv/ai/storage/artifacts/ple-host-gather-20261015')
parser = argparse.ArgumentParser(description='Install native PLE in a drained qualified resident server')
parser.add_argument('--run-root', type=Path, default=root / 'serving-v1')
parser.add_argument('--receipt', type=Path, default=root / 'serving-apply-v1.json')
args = parser.parse_args()
library = root / 'serving-source-v1/libple_host_gather.so'
binary = Path('/srv/ai/storage/artifacts/npu-collective-qfn-20261014-024609/ple-request-build-v1/native_ple_request.bin')
config = dict(table=str(root / 'serving-table-v1.json'), library=str(library),
              library_sha256=hashlib.sha256(library.read_bytes()).hexdigest(), binary=str(binary),
              binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest())
source = (root / 'serving-source-v1/tools/qwen4exp/ple_serving.py').read_text()
source += '\ndef replacements():\n    return make_replacements(' + repr(config) + ')\n'
(root / 'serving-candidate-v1.py').write_text(source)
client = QfnResidentClient('http://127.0.0.1:8001', args.run_root / 'control.token', timeout=1800)
if any('native_ple' in result for result in client.rpc('resident_status')):
    raise RuntimeError('Native PLE already installed; preserve its owners until worker process shutdown')
control = Control(generation=uuid.uuid4().hex, candidate='native_host_ple', source=source, recapture=True)
print('Applying native PLE and recapturing target/draft graphs', flush=True)
result = client.switch(control)
args.receipt.write_text(json.dumps(result, indent=2))
print('Native PLE applied and server resumed', flush=True)
