/*
 * A constructor for the AICPU kernel shared object.
 *
 * The card's AICPU kernels are plain unsigned ARM64 shared objects that the
 * device dlopen()s when it loads an AICPU package.  A shared object may carry
 * an ELF .init_array entry, which the dynamic loader runs at dlopen() time --
 * before any op is dispatched.  That gives a direct, unambiguous proof that the
 * library we substituted was actually loaded and mapped on the device, without
 * needing to understand the op ABI or to change any numerics.
 *
 * The constructor writes a marker file into the device's filesystem (the
 * AICPU kernel tmpfs is writable by the loading uid) and then returns, so the
 * library keeps working normally.
 *
 * Written in ARM64 assembly because the host has aarch64-linux-gnu-as but no
 * aarch64 gcc.
 *
 *   aarch64-linux-gnu-as -o probe_init.o probe_init.s
 *   aarch64-linux-gnu-ld -shared -o libaicpu_probe.so probe_init.o
 *
 * The resulting object is then linked into the AICPU package by
 * rebuild_aicpu_kernels.py's replacement mechanism.
 */
    .text
    .global probe_ctor
    .type   probe_ctor, %function

/*
 * void probe_ctor(void)
 *
 *   fd = open("/home/HwHiAiUser/aicpu_proof_marker", O_WRONLY|O_CREAT|O_TRUNC, 0644)
 *   if (fd >= 0) { write(fd, "AICPUPROOF-LOADED\n", 19); close(fd); }
 *
 * O_WRONLY=1, O_CREAT=0x40, O_TRUNC=0x200  ->  0x241
 */
probe_ctor:
    /* x8 = SYS_openat, x0 = AT_FDCWD (-100) */
    mov     x0, #-100
    adrp    x1, path
    add     x1, x1, :lo12:path
    mov     x2, #0x241                  /* O_WRONLY|O_CREAT|O_TRUNC */
    mov     x3, #0644
    mov     x8, #56                    /* __NR_openat */
    svc     #0

    cmp     x0, #0
    b.lt    .Ldone

    /* write(fd, buf, 19) -- fd is still in x0 */
    mov     x1, x0
    mov     x2, msg
    mov     x3, #19
    mov     x8, #64                    /* __NR_write */
    svc     #0

    /* close(fd) */
    mov     x8, #57                    /* __NR_close */
    svc     #0

.Ldone:
    ret
    .size   probe_ctor, .-probe_ctor

    .section .init_array,"aw",@init_array
    .quad   probe_ctor

    .section .rodata
path:
    .asciz  "/home/HwHiAiUser/aicpu_proof_marker"
msg:
    .ascii  "AICPUPROOF-LOADED\n"
