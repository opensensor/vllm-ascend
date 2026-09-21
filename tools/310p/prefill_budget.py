"""Split a Qwen3.8-27B prefill step into its parts, on the real shapes.

Each section runs in its OWN process: a CANN inner error poisons the device
context, and every op after it returns without executing, which shows up as
impossible throughput rather than as a failure. Rates above the hardware peak
are therefore treated as a failed measurement, not a result.

    python3 prefill_budget.py                       # every section
    python3 prefill_budget.py --only mlp_int8       # one section
    torchrun --nproc_per_node=4 prefill_budget.py --collective
"""
import argparse, os, subprocess, sys, time

HIDDEN, INTER, TP, LAYERS = 5120, 17408, 4, 64
GDN_LAYERS = 48
T = 8192

# Atlas 300I Duo: 2x 310P per card, 140 TOPS INT8 / 70 TFLOPS FP16 per CARD.
PEAK_INT8_CHIP = 70e12
PEAK_FP16_CHIP = 35e12

SECTIONS = ["mlp_int8", "attn_int8", "dynquant", "out_proj_fp16", "wy", "ut"]


def bench(fn, warmup=3, iters=10):
    import torch
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.npu.synchronize()
    return (time.perf_counter() - t0) / iters


def emit(label, seconds, count, flops=None, byts=None, peak=None):
    note = ""
    if flops is not None:
        rate = flops / seconds
        note = "  %7.2f TOP/s" % (rate / 1e12)
        if peak and rate > peak * 1.15:
            note += "  <-- IMPLAUSIBLE (>peak), op did not run"
    elif byts is not None:
        note = "  %7.1f GB/s" % (byts / seconds / 1e9)
    print("%-32s %8.3f ms x%4d = %6.3f s/step%s" % (label, seconds * 1e3, count, seconds * count, note),
          flush=True)


# vllm_ascend.utils.ACL_FORMAT_FRACTAL_NZ. Inlined rather than imported:
# importing vllm_ascend here re-runs its CANN env bootstrap after torch_npu has
# already initialised, which fails in SetPrecisionMode.
ACL_FORMAT_FRACTAL_NZ = 29


def quant_weight(out_features, in_features, device):
    """Lay an INT8 weight out the way the runtime does: NZ, then transposed."""
    import torch, torch_npu
    w = torch.randint(-127, 127, (out_features, in_features), dtype=torch.int8, device=device)
    return torch_npu.npu_format_cast(w, ACL_FORMAT_FRACTAL_NZ).transpose(0, 1)


def run_section(name):
    import torch, torch_npu  # noqa: F401
    torch.npu.set_device(0)
    dev = "npu"

    if name in ("mlp_int8", "attn_int8"):
        shapes = {
            "mlp_int8": [("MLP gate+up", T, HIDDEN, 2 * INTER // TP, LAYERS),
                         ("MLP down", T, INTER // TP, HIDDEN, LAYERS)],
            "attn_int8": [("GDN in_proj_qkv", T, HIDDEN, 10240 // TP, GDN_LAYERS),
                          ("GDN in_proj_z", T, HIDDEN, 6144 // TP, GDN_LAYERS)],
        }[name]
        for label, m, k, n, count in shapes:
            a = torch.randint(-127, 127, (m, k), dtype=torch.int8, device=dev)
            w = quant_weight(n, k, dev)
            ws = torch.ones(n, dtype=torch.float32, device=dev)
            ps = torch.ones(m, dtype=torch.float32, device=dev)
            dt = bench(lambda: torch_npu.npu_quant_matmul(a, w, ws, pertoken_scale=ps,
                                                          output_dtype=torch.float16))
            emit("%s int8" % label, dt, count, flops=2 * m * k * n, peak=PEAK_INT8_CHIP)

    elif name == "dynquant":
        for label, shape, count in (("[T x hidden]", (T, HIDDEN), 4 * LAYERS),
                                    ("[T x inter/TP]", (T, INTER // TP), LAYERS)):
            x = torch.randn(*shape, dtype=torch.float16, device=dev)
            emit("dynamic_quant %s" % label, bench(lambda: torch_npu.npu_dynamic_quant(x)),
                 count, byts=x.numel() * 3)

    elif name == "out_proj_fp16":
        m, k, n = T, 6144 // TP, HIDDEN
        a = torch.randn(m, k, dtype=torch.float16, device=dev)
        b = torch.randn(k, n, dtype=torch.float16, device=dev)
        emit("GDN out_proj fp16", bench(lambda: torch.mm(a, b)), GDN_LAYERS,
             flops=2 * m * k * n, peak=PEAK_FP16_CHIP)

    elif name in ("wy", "ut"):
        HV, CH = 48 // TP, 64
        NC = T // CH
        key = torch.randn(1, HV, NC, CH, 128, dtype=torch.float32, device=dev)
        beta = torch.rand(1, HV, NC, CH, dtype=torch.float32, device=dev)
        g = torch.randn(1, HV, NC, CH, dtype=torch.float32, device=dev).cumsum(-1)

        def wy():
            d = (g.unsqueeze(-1) - g.unsqueeze(-2)).tril(-1).exp().tril(-1)
            return -(key * beta.unsqueeze(-1) @ key.transpose(-1, -2) * d)

        if name == "wy":
            emit("WY attn build (fp32)", bench(wy, warmup=2, iters=5), GDN_LAYERS)
            return

        attn = wy()
        eye = torch.eye(CH, dtype=torch.float32, device=dev)

        def inv(mm, block=8):
            nn = mm.shape[-1]
            if nn <= block:
                out = torch.zeros_like(mm)
                idx = torch.arange(nn, device=mm.device)
                out[..., idx, idx] = 1
                for i in range(1, nn):
                    out[..., i, :i] = -(mm[..., i, :i].unsqueeze(-1) * out[..., :i, :i]).sum(-2)
                return out
            h = nn // 2
            ai, bi = inv(mm[..., :h, :h], block), inv(mm[..., h:, h:], block)
            out = torch.zeros_like(mm)
            out[..., :h, :h] = ai
            out[..., h:, h:] = bi
            out[..., h:, :h] = -bi @ mm[..., h:, :h] @ ai
            return out

        def loop():
            a = attn.clone()
            for i in range(1, CH):
                row = a[..., i, :i].clone()
                sub = a[..., :i, :i].clone()
                a[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
            return a + eye

        emit("UT blocked inverse", bench(lambda: inv(eye - attn), warmup=2, iters=5), GDN_LAYERS)
        emit("UT row substitution", bench(loop, warmup=2, iters=5), GDN_LAYERS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--collective", action="store_true")
    ap.add_argument("--only")
    args = ap.parse_args()

    if args.collective:
        import torch, torch_npu  # noqa: F401
        import torch.distributed as dist
        dist.init_process_group(backend="hccl")
        rank, world = dist.get_rank(), dist.get_world_size()
        torch.npu.set_device(rank)
        for label, shape in (("[T x hidden] fp16", (T, HIDDEN)),
                             ("[T/2 x hidden] fp16", (T // 2, HIDDEN))):
            x = torch.randn(*shape, dtype=torch.float16, device="npu")
            dt = bench(lambda: dist.all_reduce(x), warmup=5, iters=20)
            if rank == 0:
                nb = x.numel() * x.element_size()
                emit("all_reduce %s" % label, dt, 2 * LAYERS, byts=nb)
                print("    %.1f MB/call, ring moves ~1.5x over links" % (nb / 1e6), flush=True)
        dist.destroy_process_group()
        return

    if args.only:
        run_section(args.only)
        return

    # Append every section to a file as it finishes, and flush stdout each time.
    # Sections can take minutes (CANN compiles uncached shapes on first call),
    # so a run that gets interrupted must still leave behind what it had -- the
    # first attempt at this buffered everything in the parent and lost the lot.
    results = os.path.expanduser("~/prefill_budget_results.txt")
    header = "--- per-chip prefill budget, TP=%d, T=%d ---" % (TP, T)
    print(header, flush=True)
    with open(results, "w") as fh:
        fh.write(header + "\n")
        fh.flush()
    for s in SECTIONS:
        r = subprocess.run([sys.executable, __file__, "--only", s],
                           capture_output=True, text=True, env=os.environ)
        out = "".join(l for l in r.stdout.splitlines(keepends=True)
                      if not l.startswith(("INFO", "WARNING", "Warning", "[W")))
        if r.returncode != 0:
            first = [l for l in r.stderr.splitlines() if "rror" in l]
            out += "%-32s FAILED: %s\n" % (s, (first[0] if first else "")[:110])
        sys.stdout.write(out)
        sys.stdout.flush()
        with open(results, "a") as fh:
            fh.write(out)
            fh.flush()
    print("\nalso written to %s" % results, flush=True)


main()
