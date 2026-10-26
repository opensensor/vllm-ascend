#!/usr/bin/env python3
"""Inject a constructor into an ELF shared object, without relinking it.

The card's AICPU kernel shared objects are plain unsigned ARM64 ELF files that
the device dlopen()s.  To prove that a *substituted* library is the one that
actually got loaded, the library can carry an ELF ``.init_array`` entry: the
dynamic loader runs it at dlopen() time, before any op is dispatched.  The
constructor here writes a marker file into the device's filesystem.

The libraries are large and stripped, so relinking them is impractical.  Instead
this tool edits the ELF in place:

  1. append the constructor's code as a new section (``objcopy --add-section``),
  2. extend ``.init_array`` by one 8-byte pointer to it,
  3. fix up the section header table, the program headers and ``DT_INIT_ARRAY``
     / ``DT_INIT_ARRAYSZ`` in the dynamic section.

The result is a valid ELF that the loader will map, and whose constructor runs.

Usage:
    python3 inject_ctor.py <library.so> <ctor.o> <ctor_symbol> <marker_path>
"""
import struct
import subprocess
import sys


def read_elf(path):
    with open(path, "rb") as fh:
        return bytearray(fh.read())


def elf_info(d):
    """Return the header fields we need."""
    if d[:4] != b"\x7fELF":
        raise ValueError("not an ELF file")
    assert d[4] == 2, "only 64-bit ELF is supported"
    e_type, e_machine = struct.unpack_from("<HH", d, 16)
    e_entry, e_phoff, e_shoff = struct.unpack_from("<QQQ", d, 24)
    e_phentsize, e_phnum = struct.unpack_from("<HH", d, 54)
    e_shentsize, e_shnum, e_shstrndx = struct.unpack_from("<HHH", d, 58)
    return {
        "e_type": e_type, "e_machine": e_machine,
        "e_phoff": e_phoff, "e_phentsize": e_phentsize, "e_phnum": e_phnum,
        "e_shoff": e_shoff, "e_shentsize": e_shentsize, "e_shnum": e_shnum,
        "e_shstrndx": e_shstrndx,
    }


def shdr(d, i):
    o = elf_info(d)["e_shoff"] + i * elf_info(d)["e_shentsize"]
    name, stype, flags, addr, off, size, link, info, align, entsize = \
        struct.unpack_from("<IIQQQQIIQQ", d, o)
    return {"name_off": name, "type": stype, "flags": flags, "addr": addr,
            "off": off, "size": size, "link": link, "info": info,
            "align": align, "entsize": entsize, "hdr_off": o}


def write_shdr(d, i, s):
    o = elf_info(d)["e_shoff"] + i * elf_info(d)["e_shentsize"]
    struct.pack_into("<IIQQQQIIQQ", d, o, s["name_off"], s["type"], s["flags"],
                     s["addr"], s["off"], s["size"], s["link"], s["info"],
                     s["align"], s["entsize"])


def strtab(d):
    info = elf_info(d)
    s = shdr(d, info["e_shstrndx"])
    return d[s["off"]:s["off"] + s["size"]]


def section_name(d, i, st):
    s = shdr(d, i)
    end = st.find(b"\x00", s["name_off"])
    return st[s["name_off"]:end].decode(errors="replace")


def add_str(strtab_bytes, name):
    """Return (new_strtab, offset_of_name)."""
    off = len(strtab_bytes)
    new = bytes(strtab_bytes) + name.encode() + b"\x00"
    return new, off


def find_section(d, name):
    st = strtab(d)
    for i in range(elf_info(d)["e_shnum"]):
        if section_name(d, i, st) == name:
            return i, shdr(d, i)
    return None, None


def main():
    if len(sys.argv) != 5:
        print(__doc__)
        return 2
    lib_path, obj_path, sym, marker = sys.argv[1:5]

    d = read_elf(lib_path)
    info = elf_info(d)

    # 1. find the constructor's code in the object file
    sym_off = subprocess.run(
        ["aarch64-linux-gnu-nm", obj_path],
        capture_output=True, text=True, check=True).stdout
    addr = None
    for line in sym_off.splitlines():
        parts = line.split()
        if len(parts) == 3 and parts[2] == sym:
            addr = int(parts[0], 16)
    if addr is None:
        print(f"symbol {sym} not found in {obj_path}")
        return 1

    # 2. extract the object's .text verbatim -- that is the machine code we want
    code = subprocess.run(
        ["aarch64-linux-gnu-objcopy", "-O", "binary",
         "--only-section=.text", obj_path, "/dev/stdout"],
        capture_output=True, check=True).stdout
    if not code:
        print("could not extract .text from", obj_path)
        return 1
    d = read_elf(lib_path)

    # 4. place the code in the slack at the end of the file, in the first
    #    PT_LOAD segment, and give it a virtual address
    info = elf_info(d)
    loads = []
    for i in range(info["e_phnum"]):
        o = info["e_phoff"] + i * info["e_phentsize"]
        p_type, p_flags = struct.unpack_from("<II", d, o)
        p_offset, p_vaddr = struct.unpack_from("<QQ", d, o + 8)
        p_filesz, p_memsz = struct.unpack_from("<QQ", d, o + 32)
        if p_type == 1:
            loads.append((i, p_offset, p_vaddr, p_filesz, p_memsz, p_flags))
    # the PT_LOAD that covers the end of the file
    # The code must land INSIDE a loadable segment, otherwise the loader will
    # not map it and the .init_array pointer would dangle.  Prefer the
    # executable segment (R E); extend it if the new code does not fit.
    rx = None
    for i, off, va, fsz, msz, fl in loads:
        if fl & 0x1:                      # PF_X
            rx = (i, off, va, fsz, msz, fl)
            break
    if rx is None:
        print("no executable PT_LOAD segment")
        return 1
    seg_i, seg_off, seg_va, seg_fsz, seg_msz, seg_flags = rx

    # file offset: right after the segment's existing file content, aligned
    code_off = (seg_off + seg_fsz + 15) & ~15
    vaddr = seg_va + (code_off - seg_off)

    # make sure there is room in the file
    if code_off + len(code) > len(d):
        d.extend(b"\x00" * (code_off + len(code) - len(d)))

    d[code_off:code_off + len(code)] = code

    # extend that segment to cover the new code
    new_fsz = code_off + len(code) - seg_off
    new_msz = max(seg_msz, new_fsz)
    o = info["e_phoff"] + seg_i * info["e_phentsize"]
    struct.pack_into("<Q", d, o + 32, new_fsz)
    struct.pack_into("<Q", d, o + 40, new_msz)

    # 5. append the pointer to .init_array
    ia_idx, ia = find_section(d, ".init_array")
    if ia is None:
        print("library has no .init_array")
        return 1
    ptr_off = ia["off"] + ia["size"]
    struct.pack_into("<Q", d, ptr_off, vaddr)
    ia["size"] += 8
    write_shdr(d, ia_idx, ia)

    # 6. update DT_INIT_ARRAY / DT_INIT_ARRAYSZ
    dyn_idx, dyn = find_section(d, ".dynamic")
    if dyn is not None:
        p = dyn["off"]
        end = dyn["off"] + dyn["size"]
        while p < end:
            tag, val = struct.unpack_from("<qQ", d, p)
            if tag == 0:
                break
            if tag == 25:      # DT_INIT_ARRAY
                struct.pack_into("<Q", d, p + 8, ia["addr"])
            elif tag == 27:    # DT_INIT_ARRAYSZ
                struct.pack_into("<Q", d, p + 8, ia["size"])
            p += 16

    with open(lib_path, "wb") as fh:
        fh.write(d)

    print(f"injected {sym} @ vaddr 0x{vaddr:x} (file 0x{code_off:x})")
    print(f"  .init_array now {ia['size']} bytes at vaddr 0x{ia['addr']:x}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
