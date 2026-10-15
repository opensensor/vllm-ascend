# W4 bridge callback ownership audit

The isolated v2 bridge gives every queued callback copy one shared tensor
owner. A successful native submission clears that owner's references once,
instead of clearing only one callback's private vector. This addresses a
concrete ownership difference; it is not yet a measured serving improvement.

## Installed runtime evidence

Read-only inspection used the installed `torch_npu` package under
`/srv/ai/venvs/qwen38-w4-test-ce1862/lib/python3.12/site-packages/torch_npu`.
Its version is `2.13.0.rc1`, Git revision
`462f4e18a8303698aca3997edfcc5dd3b1707968`.

In `include/torch_npu/csrc/framework/OpParamMaker.h`:

- Line 238 takes `SetCustomHandler(PROC_FUNC func)` by value; line 239 assigns
  `execParam.customHandler = func` by copy.
- Line 337 copies that function again into exported execution parameters.
- Line 390 clears the command's original function; this does not clear any
  independent lambda captures in the exported copies.

The installed `libtorch_npu.so` corroborates the copy behavior. Disassembly of
`OpCommand::SetCustomHandler`, starting at `0x57b38c0`, contains two function
manager calls with clone opcode 2, at `0x57b3919` and `0x57b395d`. Each ordinary
lambda copy also copies its captured `vector<at::Tensor>`.

The asynchronous `OpCommand::Run` path exports execution parameters, enqueues
them, releases command sources and destroys a temporary function. The wheel
does not contain the queue implementation sources, and its internal symbols
are stripped. These observations prove callback copying; they do not prove
how long each queue slot retains a copy or whether every failure path retries.

Inspected source and binary SHA256:

| File | SHA256 |
| --- | --- |
| `OpParamMaker.h` | `f38b2570be985d042386ea66b364e4e9108f83d8b84da763d931673decb581a5` |
| `NPUDefine.h` | `424c4454784af1b006bcd20a9b6663a5157c0512753e1ab8794fac5d6f487d66` |
| `libtorch_npu.so` | `f571f55ddd0a0889de1b6297625f7eb897a0bf1e554a720893c035bc8a0f514d` |

## Lifetime and failure contract

`tools/qwen4exp/native_owned_bridge.cpp` preserves the qualified bridge's
native entry ABI, device checks, stream selection and binary retention. The
generic reconstruction bridge remains unchanged.

Each launch creates one `SharedLaunchArguments` object. Every function copy
captures the same shared pointer. On successful `aclrtLaunchKernelWithArgsArray`
return, the argument vector clears at the same host submission point used by
the existing bridge. There is no new device synchronization. This retains the
existing assumption of allocator safety for eager operations submitted on the
current NPU stream; it does not establish arbitrary cross-stream safety.

An atomic one-shot guard rejects any second host invocation before native
submission. Concurrent copies cannot read the argument vector while another
copy clears it. The guard also rejects retries after failure: a failed native
submission is not assumed to be free of side effects. Native graph replay
executes recorded device work; it must not invoke this host callback again.
Current qualification is limited to the eager prefill use case.

Failure or exception during native submission leaves the vector intact. The
worker-owned kernel resource retains that shared owner independently of the
framework's callbacks. An outer catch also covers command construction,
handler installation and `Run`, because queue acceptance may be uncertain
when these throw. Successful execution does not acquire the failure mutex.
There is no eviction, unload or failed-operand retry in this bridge.

## Validation and build isolation

Five CPU tests pass in `tests/ut/qwen38_1m/test_shared_launch_arguments.py`.
The compiled C++ test exercises independent vector retention in the old
design, release across shared callback copies after success, retained operands
after nonzero return and exception, and rejection of concurrent duplicate
callbacks. A source guard checks that the real command setup and submission
remain inside the outer retention catch. Builder tests verify the generic v1
default, explicit v2 source freezing and rejection of the owned bridge under
the old namespace. All scoped repository hooks pass.

The builder accepts an absolute `--bridge-source` path. Selecting
`native_owned_bridge.cpp` requires version 2 or later, freezes the sibling
`shared_launch_arguments.h`, compiles that selected source, and records
`bridge_source` plus both source digests in provenance. Version 2 uses
`qwen_route_finalize_v2`; the device kernel entry names remain v1. Existing v1
objects and worker resources must remain retained during the separate v2 gate.

Native compilation, exact tensor gates, graph behavior and serving performance
remain separate validation steps. This audit performed no NPU work or service
changes. The controller must run short private numerical gates before binding
v2 into inference, then compare matched uncached prefill requests and allocator
events. An ownership test alone cannot qualify the performance hypothesis.

## Why this is worth testing

The controller's finalizer trace reports 240 finalizer calls taking 0.306 s,
with W4 matmul time essentially unchanged at 4.843 s. Physical allocations
increased from 92 calls / 0.0189 s to 1,627 / 2.96 s; physical frees increased
from none to 1,482 / 1.54 s, and device synchronizations from 2 / 0.148 s to
143 / 2.79 s. These are profiler categories, not additive wall-clock costs.

This is evidence of allocator pressure during the slower integrated run.
Independent callback tensor ownership offers one concrete mechanism for that
pressure. The v2 experiment must still establish whether removing those extra
references reduces allocator churn and end-to-end latency without changing
outputs.

## Isolated v3 queue-drain experiment

V2 did not establish a serving gain; the controller reported a 31.37-second
trial. The next isolated build keeps the same shared owner and native math,
but bounds host submission backlog. `--flush-host-queue` is accepted only
with version 3 and the explicit owned bridge. Provenance records the boolean
`flush_host_queue`; v1 and v2 defaults remain false.

The launch saves the current NPU stream and obtains its native handle without
draining. After `OpCommand.Run`, the producer thread calls that saved stream's
`stream(true)` and checks that its handle is unchanged. This occurs inside the
outer failure-retention guard. It never drains from inside the queued handler,
which would risk waiting on its own queue. The queue drain can indirectly wait
for device work if an earlier callback synchronizes; it is not a promise of
zero device waiting.

The version 3 resource is marked `eager_only`. The finalizer binding takes the
installed original path during graph capture, including when a descriptor
already exists. CPU regression checks cover eager execution, capture fallback,
the producer-only drain location, build flag restrictions and provenance.
The combined ownership/builder/finalizer suites pass 34 tests. Native build,
numerical checks and measured serving behavior remain controller-run gates.

The controller subsequently built v3, passed exact finalization checks at
650/20480 rows on all four ranks, and passed eight bounded model smoke cases.
Three uncached 8K requests reached median 367.17 prompt tok/s versus a fresh
353.74 control. V3 is retained through resident reload. The v2 ownership-only
change still has no demonstrated serving gain. See
[full trial results and limitations](w4-prefill-model-trials.md).
