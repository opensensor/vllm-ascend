"""Record lease ownership without exposing private control credentials."""
import json
import os
from pathlib import Path

ROOT = Path('/srv/ai/storage/artifacts/npu-collective-qfn-20261014-024609')
pid = int((ROOT/'resident-live-v9-api.pid').read_text())
command = Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
assert b'tools.sidecard.qfn_resident_server' in command
assert command.count(b'--dma-reserve-device') == 4
fds = []
for entry in Path(f'/proc/{pid}/fd').iterdir():
    try:
        if os.readlink(entry) == '/dev/ascend_device_dma':
            fds.append(int(entry.name))
    except FileNotFoundError:
        pass
assert len(fds) == 4, fds
rows = []
for path in sorted(Path('/sys/bus/pci/devices').glob('*/msg/device_dma')):
    selector = path.parent.parent/'davinci_dev_id'
    original = selector.read_text().strip()
    try:
        for bank in (0, 1):
            selector.write_text(str(bank))
            rows.append(dict(field.split('=', 1) for field in path.read_text().split()))
    finally:
        selector.write_text(original)
assert len(rows) == 4 and {int(row['dev_id']) for row in rows} == set(range(4))
assert all(row['reserved'] == '1' and row['host_traffic_channels'] == '2' and
           row['host_routing_isolated'] == '1' and row['device_submit'] == '0' for row in rows)
first_line = (ROOT/'resident-live-v9/server.log').open().readline()
startup = json.loads(first_line)
assert len(startup['device_dma_reservations']) == 4
result = dict(passed=True, api_pid=pid, lease_fds=sorted(fds), startup=startup, live_devices=rows)
(ROOT/'ple-driver-serving-reservations-v2.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(dict(passed=True, api_pid=pid, active_reservations=len(rows))))
