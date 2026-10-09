# Contained host build

The final append-only qwen_streaming_v5 bundle compiles four native binaries
(six entrypoints) and one versioned bridge for dav-2002 using CANN 9.1.0 and
matching Torch 2.13.0+cpu / torch-npu 2.13.0rc1 headers. Torch ABI 1 comes from
the guarded CPU metadata query with backend autoload disabled.

All six compiler commands execute inside the static Landlock/seccomp helper.
Actual stderr attestations, helper/source/policy hashes and stdout/stderr logs
are bound to the host receipt. The final 110-process trace records 40 other-device
open attempts, all denied; there are zero successful opens beyond explicit safe
/dev/null, /dev/zero, /dev/random and /dev/urandom exceptions. Driver ioctls,
sockets and descriptor transfer are denied. No kernel/library was loaded or
inference submitted, and no server was operated.

All frozen sources and actual helper/binary/bridge/receipt/log bytes were rehashed
locally after transfer; current source bytes match the bundle. The full temporary
bundle is retained locally and on the target host. Native outputs remain unqualified
for model/service use; reference/image/quality/thermal/memory gates are pending.

The preliminary uncontained v3 build accessed driver-manager nodes inside SDK
constructors. It is quarantined and its no-device-open receipt claim is invalid.
The earlier v4 contained snapshot passed the same enforcement but was superseded
by v5 after source formatting. No existing bundle was overwritten or recompiled.
