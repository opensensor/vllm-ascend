from pathlib import Path
import json
import subprocess

root = Path('/srv/ai/storage/artifacts/ple-dma-spin-20261015')
parameter = '/sys/module/drv_pcie_host/parameters/dma_query_spin_us'
phases = [('baseline', 0), ('spin2', 2), ('spin5', 5), ('spin2-repeat', 2), ('baseline-repeat', 0)]
try:
    for phase, budget in phases:
        subprocess.run(['sudo', '-n', 'tee', parameter], input=f'{budget}\n', text=True,
                       stdout=subprocess.DEVNULL, check=True)
        for rank in range(4):
            result = subprocess.run(['timeout', '-k', '2s', '30s', 'taskset', '-c', '8',
                                     str(root / 'ple_host_dma_latency_probe'), str(rank)],
                                    text=True, capture_output=True, check=True)
            (root / f'{phase}-rank{rank}.jsonl').write_text(result.stdout)
            (root / f'{phase}-rank{rank}.stderr').write_text(result.stderr)
            rows = [json.loads(line) for line in result.stdout.splitlines()]
            assert len(rows) == 4 and all(row['exact'] for row in rows)
            print(json.dumps({'phase': phase, 'budget_us': budget, 'rank': rank,
                              'p50_us': [row['p50_ns'] / 1000 for row in rows],
                              'voluntary_switches': [row['voluntary_switches'] for row in rows]}), flush=True)
finally:
    subprocess.run(['sudo', '-n', 'tee', parameter], input='0\n', text=True,
                   stdout=subprocess.DEVNULL, check=True)
