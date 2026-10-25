import json
import time

import torch

from tools.minimax_m3.resident_int8 import reorder_codes

torch.npu.set_device(5)
torch.npu.set_compile_mode(jit_compile=False)
torch.npu.set_per_process_memory_fraction(0.02, 5)
g = torch.Generator().manual_seed(71)
x = torch.randint(-128, 128, (8, 3072, 3072), dtype=torch.int8, generator=g).npu()
torch.npu.synchronize()
t = time.monotonic()
y = reorder_codes(x, chunk=3)
z = reorder_codes(y, inverse=True, chunk=3)
print(json.dumps({"passed": bool(torch.equal(x, z)), "bytes": x.numel(), "seconds": time.monotonic() - t}), flush=True)
