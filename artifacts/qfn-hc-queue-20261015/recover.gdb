set pagination off
set confirm off
set $qfn_gil = (int) PyGILState_Ensure()
call (int) PyRun_SimpleString("exec(compile(open('/srv/ai/storage/artifacts/qfn-hc-queue-20261015/recover.py').read(), '<qfn-admission-recovery>', 'exec'), {'__name__': 'qfn_admission_recovery'})")
call (void) PyGILState_Release($qfn_gil)
detach
quit
