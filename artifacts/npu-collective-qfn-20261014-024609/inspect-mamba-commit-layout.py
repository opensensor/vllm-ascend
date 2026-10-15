"""Read-only resident candidate preparation: inspect layouts, never apply."""
import dataclasses
import json
import uuid
from pathlib import Path
from tools.glm_perf.resident_control import Control
from tools.sidecard.qfn_resident_server import QfnResidentClient

ROOT = Path('/srv/ai/storage/artifacts/npu-collective-qfn-20261014-024609')
source = '''
import gc
import json
from pathlib import Path
from vllm_ascend._310p.worker_310p import NPUWorker310

def replacements():
    workers = [o for o in gc.get_objects() if type(o) is NPUWorker310]
    if len(workers) != 1:
        raise RuntimeError('expected one worker for read-only candidate preflight')
    worker = workers[0]
    runner = worker.model_runner
    if getattr(runner, 'execute_model_state', None) is not None:
        raise RuntimeError('candidate preflight requires drained inference')
    bufs = getattr(runner, '_mamba_bufs', None)
    ctx = getattr(bufs, 'postprocess_align', None)
    report = dict(rank=worker.rank, has_context=ctx is not None, states=[])
    if ctx is not None:
        report.update(block_size=ctx.block_size, groups=ctx.mamba_group_ids)
        for group in ctx.mamba_group_ids:
            for name in runner.kv_cache_config.kv_cache_groups[group].layer_names:
                layer = runner.compilation_config.static_forward_context[name]
                for index, state in enumerate(layer.kv_cache):
                    report['states'].append(dict(group=group, layer=name, index=index,
                        shape=list(state.shape), stride=list(state.stride()),
                        dtype=str(state.dtype), contiguous=state.is_contiguous(),
                        element_size=state.element_size()))
        batch = runner.input_batch
        mapped = getattr(batch, '_prefix_mamba_postprocess_tables', None)
        report['compact_host_tables'] = None if mapped is None else {
            str(k): dict(shape=list(v.shape), dtype=str(v.dtype)) for k, v in mapped.items()}
        report['buffers'] = {name: dict(shape=list(getattr(ctx, name).gpu.shape),
            dtype=str(getattr(ctx, name).gpu.dtype)) for name in (
            'mamba_state_idx_buf', 'num_scheduled_tokens_buf',
            'num_computed_tokens_buf', 'num_draft_tokens_buf')}
    path = Path('/srv/ai/storage/artifacts/npu-collective-qfn-20261014-024609/iteration-closure-v1') / f'live-commit-layout-rank{worker.rank}.json'
    path.write_text(json.dumps(report, indent=2) + '\\n')
    return {'vllm_ascend._310p.worker_310p:NPUWorker310.qfn_cpu_profile_stop': NPUWorker310.qfn_cpu_profile_stop}
'''
client = QfnResidentClient('http://127.0.0.1:8001', ROOT/'resident-live-v4/control.token')
assert not client.request('/is_paused', method='GET')['is_paused']
control = Control(generation=uuid.uuid4().hex, candidate='mamba_commit_layout_preflight_only', source=source, recapture=False)
result = dict(before=client.rpc('resident_status'), control=dataclasses.asdict(control))
client.request('/pause?mode=wait&clear_cache=false')
try:
    result['prepared'] = client.rpc('resident_prepare', json.dumps(dataclasses.asdict(control)))
    assert all(r['generation'] == control.generation for r in result['prepared'])
    result['after'] = client.rpc('resident_status')
    for before, after in zip(result['before'], result['after']):
        for key in ('pid', 'generation', 'digest', 'graphs_dirty', 'weight_storage_digest'):
            assert before[key] == after[key]
finally:
    client.resume()
    (ROOT/'iteration-closure-v1/layout-inspection.json').write_text(json.dumps(result, indent=2)+'\n')
print('Read-only layout inspection complete; no control applied; server resumed.')
