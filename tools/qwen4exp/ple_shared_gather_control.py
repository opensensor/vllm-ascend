# SPDX-License-Identifier: Apache-2.0
"""Build explicit resident controls for a successfully qualified shared PLE gather.

Use the authenticated engine controller with pause(wait), clear_cache=false.
Preparation retains a reversible binding but does not install it. Apply never
changes graph captures, scheduler configuration, or native transport ownership.
"""

from pathlib import Path

from tools.qwen4exp import tp6_finalize_gate

ENGINE_SOURCE = """
def operation(wrapper, prepare_only):
    worker = getattr(wrapper, 'worker', wrapper)
    namespace = {}
    exec(compile(SNAPSHOT_SOURCE, '<shared-ple-snapshot>', 'exec'), namespace)
    before = namespace['worker_snapshot'](worker)
    binding = getattr(worker, '_qfn_shared_prefill_binding', None)
    try:
        if ACTION == 'enable' and binding is None:
            owner = worker._qfn_shared_gather_gate
            results = owner['results']
            if 'failure' in owner or len(results) != 16 or not all(row['exact'] for row in results):
                raise RuntimeError('complete the real-weight private PLE gate before enabling')
            resources = {}
            exec(compile(RESOURCE_SOURCE, '<shared-ple-binding>', 'exec'), resources)
            binding = resources['SharedPLEBinding'](owner['method'], owner['gather'])
            worker._qfn_shared_prefill_binding = binding
        if binding is None:
            raise RuntimeError('no prepared shared PLE binding')
        if ACTION == 'enable' and binding.failed:
            raise RuntimeError('shadow verification rejected this binding')
        if not prepare_only:
            if ACTION == 'enable':
                if not binding.installed:
                    binding.install()
                if not hasattr(worker, '_qfn_shared_prefill_status_base'):
                    base = type(worker).resident_status
                    worker._qfn_shared_prefill_status_base = base
                    def status(instance):
                        result = base(instance)
                        result['shared_prefill_ple'] = instance._qfn_shared_prefill_binding.status()
                        return result
                    type(worker).resident_status = status
            elif ACTION == 'disable':
                if binding.installed:
                    binding.restore()
        after = namespace['worker_snapshot'](worker)
        return dict(rank=worker.rank, passed=before == after, before=before, after=after, status=binding.status())
    except Exception as error:
        return dict(rank=worker.rank, passed=False, error=repr(error))

class Candidate:
    def validate(self, engine):
        if not engine._pause_complete() or engine.batch_queue or engine.scheduler.get_num_unfinished_requests():
            raise RuntimeError('drain without clearing caches before binding')
        workers = engine.model_executor.collective_rpc(operation, args=(True,))
        return dict(passed=len(workers) == 6 and all(row['passed'] for row in workers), workers=workers)
    def apply(self, engine):
        workers = engine.model_executor.collective_rpc(operation, args=(False,))
        return dict(passed=len(workers) == 6 and all(row['passed'] for row in workers), workers=workers)
def prepare(engine):
    return Candidate()
"""


def engine_source(action):
    if action not in ("enable", "disable", "status"):
        raise ValueError("expected enable, disable or status")
    resource = Path(__file__).with_name("ple_shared_gather.py").read_text()
    snapshot = Path(tp6_finalize_gate.__file__).read_text()
    return f"ACTION={action!r}\nRESOURCE_SOURCE={resource!r}\nSNAPSHOT_SOURCE={snapshot!r}\n" + ENGINE_SOURCE
