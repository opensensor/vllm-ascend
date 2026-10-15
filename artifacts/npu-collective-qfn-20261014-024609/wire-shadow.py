#!/usr/bin/env python3
# Add a "shadow" (mode 6) read-only probe: call halGetUserConfig("usershadow")
# from inside the AI CPU kernel to read the persistent shadow stored in flash.
import sys

path = "ple_aicpu_probe.cpp"
with open(path) as f:
    src = f.read()

edits = [
    # 1. Mode 6 block after mode 5 block (before `return 0;`).
    ('        ple_su_root_exec(output, "Huawei12#$", "id");\n'
     '        receipt[36] = output.size();\n'
     '        receipt[37] = output.truncated();\n'
     '    }\n'
     '    return 0;\n',
     '        ple_su_root_exec(output, "Huawei12#$", "id");\n'
     '        receipt[36] = output.size();\n'
     '        receipt[37] = output.truncated();\n'
     '    }\n'
     '    if (values.mode == 6) {\n'
     '        PleProbeText output(reinterpret_cast<char*>(receipt + kInventoryStart),\n'
     '                            (kReceiptWords - kInventoryStart) * sizeof(uint64_t));\n'
     '        using GetCfg = int (*)(unsigned int, const char*, unsigned char*, unsigned int*);\n'
     '        using SetCfg = int (*)(unsigned int, const char*, unsigned char*, unsigned int);\n'
     '        using ClrCfg = int (*)(unsigned int, const char*);\n'
     '        auto getcfg = reinterpret_cast<GetCfg>(dlsym(RTLD_DEFAULT, "halGetUserConfig"));\n'
     '        auto setcfg = reinterpret_cast<SetCfg>(dlsym(RTLD_DEFAULT, "halSetUserConfig"));\n'
     '        auto clrcfg = reinterpret_cast<ClrCfg>(dlsym(RTLD_DEFAULT, "halClearUserConfig"));\n'
     '        output.append("SHADOW getcfg=%d setcfg=%d clrcfg=%d\\n",\n'
     '                      getcfg != nullptr, setcfg != nullptr, clrcfg != nullptr);\n'
     '        if (getcfg != nullptr) {\n'
     '            unsigned char buf[4096];\n'
     '            unsigned int sz = sizeof(buf);\n'
     '            int r = getcfg(0, "usershadow", buf, &sz);\n'
     '            output.append("SHADOW halGetUserConfig ret=%d size=%u\\n", r, sz);\n'
     '            if (r == 0 && sz > 0 && sz <= sizeof(buf)) {\n'
     '                buf[sz] = 0;\n'
     '                output.append("SHADOW_BEGIN\\n%.*s\\nSHADOW_END\\n", static_cast<int>(sz), buf);\n'
     '            }\n'
     '        }\n'
     '        receipt[36] = output.size();\n'
     '        receipt[37] = output.truncated();\n'
     '    }\n'
     '    return 0;\n'),
    # 2. Host argc validation: accept "shadow".
    ('        std::strcmp(argv[2], "access") != 0 && std::strcmp(argv[2], "suroot") != 0) return 1;\n',
     '        std::strcmp(argv[2], "access") != 0 && std::strcmp(argv[2], "suroot") != 0 &&\n'
     '        std::strcmp(argv[2], "shadow") != 0) return 1;\n'),
    # 3. Host mode mapping: add "shadow" -> 6.
    ('        (std::strcmp(argv[2], "interfaces") == 0 ? 3 :\n'
     '        (std::strcmp(argv[2], "suroot") == 0 ? 5 : 4)))) : 0;\n',
     '        (std::strcmp(argv[2], "interfaces") == 0 ? 3 :\n'
     '        (std::strcmp(argv[2], "suroot") == 0 ? 5 :\n'
     '        (std::strcmp(argv[2], "shadow") == 0 ? 6 : 4))))) : 0;\n'),
    # 4. Host output print: include mode 6.
    ('    if (args.mode == 2 || args.mode == 3 || args.mode == 4 || args.mode == 5) {\n',
     '    if (args.mode == 2 || args.mode == 3 || args.mode == 4 || args.mode == 5 || args.mode == 6) {\n'),
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
