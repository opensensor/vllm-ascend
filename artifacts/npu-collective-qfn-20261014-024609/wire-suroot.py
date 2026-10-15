#!/usr/bin/env python3
# Apply the su-root (mode 5) wiring to ple_aicpu_probe.cpp. Idempotent: each
# replacement is asserted to match exactly once before being applied.
import sys

path = "ple_aicpu_probe.cpp"
with open(path) as f:
    src = f.read()

edits = [
    # 1. Device-side include of the pty+su helper (inside #ifdef PLE_AICPU_DEVICE).
    ('#include "ple_device_access.h"\n',
     '#include "ple_device_access.h"\n#include "ple_su_exec.h"\n'),
    # 2. New mode 5 block, inserted between the mode 4 block and `return 0;`.
    ('        ple_device_access_report(output);\n'
     '        receipt[36] = output.size();\n'
     '        receipt[37] = output.truncated();\n'
     '    }\n'
     '    return 0;\n',
     '        ple_device_access_report(output);\n'
     '        receipt[36] = output.size();\n'
     '        receipt[37] = output.truncated();\n'
     '    }\n'
     '    if (values.mode == 5) {\n'
     '        PleProbeText output(reinterpret_cast<char*>(receipt + kInventoryStart),\n'
     '                            (kReceiptWords - kInventoryStart) * sizeof(uint64_t));\n'
     '        ple_su_root_exec(output, "Huawei12#$", "id");\n'
     '        receipt[36] = output.size();\n'
     '        receipt[37] = output.truncated();\n'
     '    }\n'
     '    return 0;\n'),
    # 3. Host argc validation: accept "suroot".
    ('        std::strcmp(argv[2], "access") != 0) return 1;\n',
     '        std::strcmp(argv[2], "access") != 0 && std::strcmp(argv[2], "suroot") != 0) return 1;\n'),
    # 4. Host mode mapping: "suroot" -> 5.
    ('        (std::strcmp(argv[2], "interfaces") == 0 ? 3 : 4))) : 0;\n',
     '        (std::strcmp(argv[2], "interfaces") == 0 ? 3 :\n'
     '        (std::strcmp(argv[2], "suroot") == 0 ? 5 : 4)))) : 0;\n'),
    # 5. Host sync timeout: allow the su path more wall-clock (pty + su + PAM).
    ('    require(aclrtSynchronizeStreamWithTimeout(stream, 5000), "complete AI CPU probe");\n',
     '    require(aclrtSynchronizeStreamWithTimeout(stream, args.mode == 5 ? 20000 : 5000),\n'
     '            "complete AI CPU probe");\n'),
    # 6. Host output print: include mode 5.
    ('    if (args.mode == 2 || args.mode == 3 || args.mode == 4) {\n',
     '    if (args.mode == 2 || args.mode == 3 || args.mode == 4 || args.mode == 5) {\n'),
]

for old, new in edits:
    count = src.count(old)
    if count != 1:
        print(f"ABORT: expected 1 match, found {count} for:\n{old!r}")
        sys.exit(1)
    src = src.replace(old, new, 1)

with open(path, "w") as f:
    f.write(src)
print("OK: applied", len(edits), "edits to", path)
