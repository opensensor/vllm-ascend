#!/usr/bin/env python3
"""Classify exported DSMI functions as stubs or real implementations.

`dlsym` on a shared library returns the PLT trampoline, not the function body,
so a byte-level stub test on the dlsym result is meaningless.  Instead this
parses the ELF symbol table for the real addresses and checks the prologue.

A stub in this driver is exactly:

    mov  eax, 0xfffe      ; b8 fe ff 00 00
    pop  rbp              ; 5d
    ret                   ; c3

i.e. the function ignores all arguments and returns DRV_ERROR_INVALID_VALUE.
"""
import subprocess
import sys

LIB = "/usr/local/Ascend/driver/lib64/driver/libdrvdsmi_host.so"

# A stub stores its args then does: mov eax,0xfffe ; pop rbp ; ret
STUB_TAIL = b"\xb8\xfe\xff\x00\x00\x5d\xc3"

INTERESTING = [
    "dsmi_dft_get_elable",
    "dsmi_cmd_dft_get_elabel",
    "dsmi_get_custom_op_secverify_cert_show_info",
    "dsmi_cmd_get_custom_cert_show_info",
    "dsmi_get_rootkey_status",
    "dsmi_cmd_get_sign_cert",
    "dsmi_cmd_set_custom_sign_cert",
    "dsmi_attest_get_akcert",
    "cert_expired_config_para_check",
    "dsmi_get_board_id",
    "dsmi_get_board_info",
    "dsmi_get_system_time",
    "dsmi_cmd_get_user_config",
    "dsmi_cmd_clear_user_config",
]


def load_symbols(path):
    """Return {name: (value, size)} for FUNC symbols, via nm."""
    out = subprocess.run(["nm", "-D", "--defined-only", "-S", path],
                         capture_output=True, text=True).stdout
    syms = {}
    for line in out.splitlines():
        parts = line.split()
        if len(parts) >= 4 and parts[2] in ("T", "t", "W", "w"):
            addr = int(parts[0], 16)
            size = int(parts[1], 16)
            syms[parts[3]] = (addr, size)
        elif len(parts) == 3 and parts[1] in ("T", "t", "W", "w"):
            syms[parts[2]] = (int(parts[0], 16), 0)
    return syms


def main():
    with open(LIB, "rb") as fh:
        blob = fh.read()

    # map virtual address -> file offset using the ELF program headers
    segments = []
    for line in subprocess.run(["readelf", "-l", "-W", LIB],
                               capture_output=True, text=True).stdout.splitlines():
        f = line.split()
        if len(f) >= 6 and f[0] == "LOAD":
            off = int(f[1], 16)
            vaddr = int(f[2], 16)
            filesz = int(f[4], 16)
            segments.append((vaddr, off, filesz))

    def vaddr_to_off(v):
        for vaddr, off, filesz in segments:
            if vaddr <= v < vaddr + filesz:
                return off + (v - vaddr)
        return None

    syms = load_symbols(LIB)
    print(f"{'symbol':<46} {'kind':<6} {'addr':<12} prologue")
    print("-" * 100)
    for name in INTERESTING:
        if name not in syms:
            print(f"{name:<46} {'ABSENT':<6}")
            continue
        addr, _size = syms[name]
        off = vaddr_to_off(addr)
        if off is None:
            print(f"{name:<46} {'?':<6} 0x{addr:x}  (no file mapping)")
            continue
        pro = blob[off:off + 32]
        # the stub's "mov eax,0xfffe ; pop rbp ; ret" must appear in the first
        # few instructions, and the function must be short (it ignores args)
        kind = "STUB" if STUB_TAIL in pro[:24] else "real"
        print(f"{name:<46} {kind:<6} 0x{addr:x}   {pro.hex()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
