/*
 * A constructor for the AICPU kernel shared object.
 *
 * The card's AICPU kernels are plain unsigned ARM64 shared objects that the
 * device dlopen()s.  A shared object may carry an ELF .init_array entry, which
 * the dynamic loader runs at dlopen() time -- before any op is dispatched --
 * so a library carrying one is provably loaded and executed on the device.
 *
 * The oracle is the device's kernel log: the constructor writes a marker to
 * /dev/kmsg, and msnpureport collects the device kernel log into
 * hisi_logs/device-N/<timestamp>/log/kernel.log.  That makes "did our code
 * run on the device?" answerable from the host.
 *
 * Two formats are attempted for /dev/kmsg, because the kernel wants
 * "<prio>,<seq>,<time>[,flags];<message>" but also accepts a bare message.
 * A plain file is written as well, under a path the device can create.
 *
 * Written in ARM64 assembly because the host has aarch64-linux-gnu-as but no
 * aarch64 gcc.
 *
 *   aarch64-linux-gnu-as -o probe_init.o probe_init.s
 */
    .text
    .global probe_ctor
    .type   probe_ctor, %function

/*
 * void probe_ctor(void)
 *
 *   fd = open("/dev/kmsg", O_WRONLY);            -> fd 1
 *   write(fd, "AICPUPROOF-LOADED\n", 19)
 *   close(fd)
 *
 *   fd = open("/var/log/npu/aicpu_proof.txt", O_WRONLY|O_CREAT|O_TRUNC, 0644)
 *   write(fd, "AICPUPROOF-LOADED\n", 19)
 *   close(fd)
 */
probe_ctor:
    /* --- /dev/kmsg --- */
    mov     x0, #-100                       /* AT_FDCWD */
    adrp    x1, path_kmsg
    add     x1, x1, :lo12:path_kmsg
    mov     x2, #0x1                        /* O_WRONLY */
    mov     x3, #0
    mov     x8, #56                        /* __NR_openat */
    svc     #0
    cmp     x0, #0
    b.lt    .Lfile

    mov     x1, x0
    adrp    x2, msg_kmsg
    add     x2, x2, :lo12:msg_kmsg
    mov     x3, #22                        /* len("6,1,1,-;AICPUPROOF\n") */
    mov     x8, #64                        /* __NR_write */
    svc     #0
    mov     x8, #57                        /* __NR_close */
    svc     #0

.Lfile:
    /* --- a plain file too --- */
    mov     x0, #-100
    adrp    x1, path_file
    add     x1, x1, :lo12:path_file
    mov     x2, #0x241                     /* O_WRONLY|O_CREAT|O_TRUNC */
    mov     x3, #0644
    mov     x8, #56
    svc     #0
    cmp     x0, #0
    b.lt    .Ldone
    mov     x1, x0
    adrp    x2, msg
    add     x2, x2, :lo12:msg
    mov     x3, #19
    mov     x8, #64
    svc     #0
    mov     x8, #57
    svc     #0

.Ldone:
    ret
    .size   probe_ctor, .-probe_ctor

    .section .init_array,"aw",@init_array
    .quad   probe_ctor

    .section .rodata
path_kmsg:
    .asciz  "/dev/kmsg"
path_file:
    .asciz  "/var/log/npu/aicpu_proof.txt"
msg_kmsg:
    .ascii  "6,1,1,-;AICPUPROOF\n"
msg:
    .ascii  "AICPUPROOF-LOADED\n"
