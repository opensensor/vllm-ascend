"""Retain this experiment's sealed table, then qualify four simultaneous readers."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import fcntl
import json
import os
import subprocess
import time

ROOT = Path('/srv/ai/storage/artifacts/ple-host-gather-20261015')
PARENT = 341769
PYTHON = '/srv/ai/venvs/qwen38-w4-test-ce1862/bin/python'
metadata = json.loads((ROOT / 'full-v5/metadata.json').read_text())
command = Path(f'/proc/{PARENT}/cmdline').read_bytes().split(b'\0')
assert b'--worker' not in command and os.fsencode(str(ROOT / 'full-v5')) in command
fd = os.open(f'/proc/{PARENT}/fd/3', os.O_RDONLY | os.O_CLOEXEC)
try:
    info = os.fstat(fd)
    seals = fcntl.F_SEAL_WRITE | fcntl.F_SEAL_GROW | fcntl.F_SEAL_SHRINK | fcntl.F_SEAL_SEAL
    assert info.st_ino == metadata['shared_inode'] and info.st_size == metadata['table_bytes']
    assert fcntl.fcntl(fd, fcntl.F_GET_SEALS) & seals == seals
    print(json.dumps({'stage': 'retained_same_table', 'inode': info.st_ino, 'bytes': info.st_size}), flush=True)
    deadline = time.monotonic() + 600
    while Path(f'/proc/{PARENT}').exists():
        assert time.monotonic() < deadline, 'sequential matrix did not finish'
        time.sleep(1)
    assert Path('/sys/module/drv_pcie_host/parameters/dma_query_spin_us').read_text().strip() == '5'
    output = ROOT / 'concurrent-v6'
    output.mkdir(exist_ok=False)
    (output / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    (output / 'meminfo-before.txt').write_text(Path('/proc/meminfo').read_text())
    def run(rank):
        cpu = [8, 24, 40, 56][rank]
        args = [PYTHON, '-m', 'tools.qwen4exp.benchmark_ple_host_gather_310', '--worker', '--fd', str(fd),
                '--rank', str(rank), '--cpu', str(cpu), '--native-only', '--native-producer',
                '--metadata', str(output / 'metadata.json'), '--build',
                '/srv/ai/storage/artifacts/npu-collective-qfn-20261014-024609/ple-request-build-v1',
                '--library', str(ROOT / 'native-v6/libple_host_gather.so'),
                '--output', str(output / f'spin5-rank{rank}.jsonl')]
        with (output / f'rank{rank}.stdout').open('w') as stdout, (output / f'rank{rank}.stderr').open('w') as stderr:
            subprocess.run(args, cwd=ROOT / 'native-v6', pass_fds=(fd,), stdout=stdout, stderr=stderr,
                           check=True, timeout=240)
        rows = [json.loads(line) for line in (output / f'spin5-rank{rank}.jsonl').read_text().splitlines()]
        cases = [row for row in rows if 'metrics' in row]
        assert len(cases) == 9 and all(row['exact'] and row['shared_inode'] == info.st_ino for row in cases)
        print(json.dumps({'stage': 'concurrent_rank_passed', 'rank': rank, 'cpu': cpu}), flush=True)
        return {'rank': rank, 'cpu': cpu, 'checked_requests': sum(row['checked_requests'] for row in cases)}
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(run, range(4)))
    (output / 'result.json').write_text(json.dumps({'passed': True, 'same_shared_inode': info.st_ino,
                                                 'table_bytes': info.st_size, 'results': results}, indent=2) + '\n')
finally:
    os.close(fd)
print('Shared table released', flush=True)
