# Host compiler containment

Added a standalone Linux host helper in
`tools/qwen4exp/host_compile_sandbox.cpp`. It links no CANN SDK and must be built
**statically**, so dynamic preload constructors cannot run before its confinement.
The builder must also clear `LD_PRELOAD` and `LD_AUDIT` before starting the helper
bootstrap/compiler. Device work and server access are not part of this helper.

The explicit CLI is:

```text
host-compile-sandbox --allow-read ABSOLUTE_PATH ... \
  --allow-write ABSOLUTE_PATH ... -- COMMAND ARGUMENTS
```

Only declared roots are allowed. Read-only roots grant read/execute; write roots
add ordinary file/directory operations. Broad `/`, `/dev`, `/srv`, `/home`, `/root`,
`/run` and `/var` grants are rejected after realpath resolution. Device creation is
not granted. The only permitted `/dev` exceptions are explicit null/zero/random/
urandom paths with verified Linux character-device identities. Symlink aliases
follow their target and cannot escape the filesystem policy.

The helper requires Landlock ABI 3 or newer, covering truncation confinement.
It applies `no_new_privs`, restricts filesystem access, closes all inherited
descriptors above 2, and checks that standard descriptors are ordinary streams or
explicitly safe character devices. Block devices and inherited sockets are rejected.
A seccomp filter additionally denies device ioctls, network/local sockets and
message-based descriptor transfer, preventing a driver-daemon bypass of filesystem
confinement. Its audited syscall filter supports x86-64 only; unsupported kernel,
architecture or restriction setup exits 125 before executing the child.

After confinement, preload/audit variables are discarded and Torch device backend
autoload is disabled in the child environment. `--probe` only queries the Landlock
ABI; it is not evidence that a compiler invocation was restricted. A successful
execution emits a stderr JSON receipt after actual restrictions and before child
startup. The builder must bind helper source/binary hashes, exact allow roots,
command argv and actual child results, and verify driver-open outcomes separately.

Validation: **17 real Linux tests passed**, using local ordinary files and symlink
aliases rather than NPU driver probes. Tests cover allowed reads/writes/execution,
denied outside reads, alias escape, read-only overwrite/truncation, inherited
descriptor closure, broad-root rejection, socket creation rejection, safe `/dev/null`
read/write with ioctl denied, static ELF bootstrap, sanitized child environment and
fail-closed child execution. C++ compilation uses `-static -Wall -Wextra -Werror`;
scoped Ruff and formatting passed. The local kernel reports Landlock ABI 8.

No CANN compiler or SDK library was executed by this worker, no NPU driver file was
opened for these tests, and no inference or server action occurred. This local
result does not prove the server's kernel support or CANN behavior under confinement.
The coordinator must perform the contained compiler/strace validation and reject
compilation if the SDK requires forbidden devices, ioctls or IPC.

Filesystem confinement restricts opens and mutations, not every metadata syscall.
Allow roots remain explicit trusted toolchain/build configuration; this helper is
not a general container isolation system. Its purpose is enforcing the offline
compiler boundary against SDK initialization/driver access, not predicting thermal
or performance results.
