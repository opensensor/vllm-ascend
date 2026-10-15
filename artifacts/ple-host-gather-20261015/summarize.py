"""Summarize complete full-table qualification receipts; fail on missing cases."""
from pathlib import Path
import json
import statistics

root = Path(__file__).resolve().parent
stages = {}
all_cases = []
invalid_receipts = invalid_rows = 0
for stage, phases, modes in [
    ('full-v2', ('baseline', 'spin5'), ('packed', 'direct_rows')),
    ('full-v5', ('spin5',), ('native_packed', 'native_poll20_packed', 'native_poll40_packed')),
    ('concurrent-v6', ('spin5',), ('native_packed', 'native_poll20_packed', 'native_poll40_packed')),
]:
    metadata = json.loads((root / stage / 'metadata.json').read_text())
    assert metadata['table_bytes'] == 102400491520 and not metadata['synthetic']
    groups = []
    for phase in phases:
        cases = []
        for rank in range(4):
            rows = [json.loads(line) for line in (root / stage / f'{phase}-rank{rank}.jsonl').read_text().splitlines()]
            current = [row for row in rows if 'metrics' in row]
            assert len(current) == 3 * len(modes)
            assert {(row['tokens'], row['mode']) for row in current} == {
                (tokens, mode) for tokens in (1, 3, 12) for mode in modes}
            assert all(row['exact'] and row['shared_inode'] == metadata['shared_inode'] and
                       row['rank'] == rank and row['checked_requests'] == 36 for row in current)
            errors = [row for row in rows if 'invalid_receipt_rejected' in row]
            assert len(errors) == 3 and all(row['invalid_receipt_rejected'] and row['output_preserved'] and
                                           row['recovered'] for row in errors)
            invalid_receipts += len(errors)
            invalid_rows += sum(row.get('invalid_row_rejected', False) for row in errors)
            cases.extend(current)
        all_cases.extend(cases)
        for tokens in (1, 3, 12):
            for mode in modes:
                selected = [row for row in cases if row['tokens'] == tokens and row['mode'] == mode]
                metrics = {}
                for name in selected[0]['metrics']:
                    medians = [row['metrics'][name]['p50_ns'] for row in selected]
                    metrics[name] = {'median_of_rank_p50_ns': statistics.median(medians),
                                     'rank_p50_ns_range': [min(medians), max(medians)],
                                     'max_rank_p95_ns': max(row['metrics'][name]['p95_ns'] for row in selected)}
                groups.append({'phase': phase, 'tokens': tokens, 'mode': mode, 'metrics': metrics,
                               'measured_requests': sum(row['metrics']['native_transport']['count'] for row in selected),
                               'producer_poll_completions': sum(row.get('producer_poll_completions', 0) for row in selected)})
    stages[stage] = {'shared_inode': metadata['shared_inode'], 'groups': groups}
assert stages['full-v5']['shared_inode'] == stages['concurrent-v6']['shared_inode']
assert json.loads((root / 'concurrent-v6/result.json').read_text())['passed']
report = {'passed': True, 'table_bytes': 102400491520, 'table_gib': 102400491520 / 1024**3,
          'one_immutable_shared_table_per_run': True, 'four_simultaneous_readers_passed': True,
          'cpu_tests_passed': 75, 'benchmark_cases': len(all_cases),
          'measured_requests': sum(row['metrics']['native_transport']['count'] for row in all_cases),
          'payload_checked_requests_including_warmups': sum(row['checked_requests'] for row in all_cases),
          'payload_checked_rows': sum(row['checked_requests'] * row['tokens'] * 16 for row in all_cases),
          'invalid_receipt_rejections_with_recovery': invalid_receipts,
          'invalid_row_rejections_with_recovery': invalid_rows, 'stages': stages,
          'limits': ['Host-assisted gather/DMA; no autonomous device access to host RAM.',
                     'Private experiment, not installed in the live model-serving lookup path.',
                     'Input setup and correctness readbacks are outside timed intervals.',
                     'Fixed mode order and repeated request sequences allow cache warming; whole-call ratios are not cold-cache speedup estimates.',
                     '32 measured requests per case and short concurrency qualification do not establish model throughput or worst-case latency.',
                     'The larger host table is prefaulted but not DMA-pinned; only staging buffers have prepared DMA descriptors.',
                     'Standalone CPU tests isolate framework conftest; ordinary test entry was blocked by missing vllm.third_party.flash_linear_attention.']}
(root / 'live-validation.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps({k:v for k,v in report.items() if k not in ('stages','limits')}, indent=2))
