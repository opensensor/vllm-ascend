"""Exercise ACL startup/host DMA before, during, and after all-rank leases."""
import concurrent.futures
import dataclasses
import json
import pathlib
import subprocess
from ple_driver import reserve_devices

ROOT = pathlib.Path('/srv/ai/storage/artifacts/npu-collective-qfn-20261014-024609')
result = {'passed': False, 'phases': []}


def probe(rank):
    row = {'rank': rank, 'passed': False}
    try:
        process = subprocess.run([str(ROOT / 'ple_host_dma_probe'), str(rank)],
                                 text=True, capture_output=True, timeout=60)
        row.update(returncode=process.returncode, stdout=process.stdout, stderr=process.stderr)
        records = [json.loads(line) for line in process.stdout.splitlines() if line.startswith('{')]
        row['results'] = records
        row['passed'] = process.returncode == 0 and len(records) == 5 and all(r.get('exact') for r in records[1:])
    except Exception as error:
        row['error'] = repr(error)
    return row


try:
    for phase in ('before', 'leased', 'after_close'):
        with reserve_devices(range(4) if phase == 'leased' else []) as leases:
            entry = {'phase': phase, 'leases': [dataclasses.asdict(lease.query()) for lease in leases], 'rounds': []}
            result['phases'].append(entry)
            for trial in range(3):
                with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
                    rows = list(pool.map(probe, range(4)))
                entry['rounds'].append(rows)
                if not all(row['passed'] for row in rows):
                    raise RuntimeError(f'{phase} trial {trial} failed')
            for lease in leases:
                lease.query()
    result['passed'] = True
    result['exact_transfers'] = 144
finally:
    (ROOT / 'ple-driver-deployment-v2/dma-coexist.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'passed': result['passed'], 'phases': len(result['phases'])}), flush=True)
