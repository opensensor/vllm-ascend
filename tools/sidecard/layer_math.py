# SPDX-License-Identifier: Apache-2.0
"""Explicit GLM text layer math for the isolated heterogeneous runtime.

This eager implementation is independent of vLLM's scheduler. Dense DSA is
exact within 2048 tokens; longer sequences use learned causal compressed-pool
selection with stage-owned cache and persistent incomplete pool tails.
Device imports occur only inside the actor that owns their context.
"""

MAX_CONTEXT = 1048576
INDEX_TOP_K = 2048
HIDDEN = 4096
HEADS = 64
HEAD_DIM = 128
STREAMS = 4
TOP_K = 8
LOWER_BOUND = -5.0
L2_EPS = 1e-6
FRACTAL_NZ = 29


def prepare_projection_layout(weights, format_weight):
    """Format only 2D FP16 linear weights, retaining absorbed/cache operands.

    Decode's skinny GEMM otherwise reformats the large ND weight each call.
    kv_b and indexer weights have separate view/batched-matmul consumers.
    """
    for name, weight in list(weights.items()):
        if (
            name == "self_attn.in_proj"
            or name == "self_attn.kv_a_proj_with_mqa.weight"
            or (name.endswith("proj.weight") and "indexer" not in name and "kv_b" not in name)
        ):
            weights[name] = format_weight(weight, FRACTAL_NZ)


def validate_config(config, max_context):
    expected = {
        "hidden_size": HIDDEN,
        "num_hidden_layers": 45,
        "n_routed_experts": 288,
        "num_experts_per_tok": TOP_K,
        "hc_mult": STREAMS,
        "qk_rope_head_dim": 0,
        "n_group": 1,
        "scoring_func": "sigmoid",
        "index_topk": INDEX_TOP_K,
        "index_kpool": 4,
        "swiglu_limit": 10.0,
    }
    if any(config.get(key) != value for key, value in expected.items()):
        raise ValueError("checkpoint geometry or attention semantics differ from supported GLM text profile")
    if type(max_context) is not int or not 1 <= max_context <= MAX_CONTEXT:
        raise ValueError("context exceeds the trained model's supported position range")


def rms_norm(x, weight, eps, torch):
    xf = x.float()
    return (xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + eps) * weight.float()).to(weight.dtype)


def round_state(x, torch):
    """Preserve BF16 residual range using FP32 storage on both architectures."""
    return x.to(torch.bfloat16).float()


def mhc_post(x, residual, post, comb, torch):
    if residual.device.type == "cuda":
        from .cuda_layer_ops import mhc_post as cuda_mhc_post

        return cuda_mhc_post(x, residual, post, comb)
    # comb[input_stream, output_stream]; elementwise reduction avoids 310P's
    # FP32 matmul rounding to FP16 and makes the accumulator policy explicit.
    mixed = (residual[:, :, None, :] * comb[:, :, :, None]).sum(1)
    return round_state(mixed + post * x[:, None, :].float(), torch)


def mhc_pre(residual, weights, kind, norm_name, config, torch, native_sinkhorn=None):
    if residual.device.type == "cuda":
        from .cuda_layer_ops import mhc_pre as cuda_mhc_pre

        return cuda_mhc_pre(
            residual,
            weights[f"hc_{kind}_fn"],
            weights[f"hc_{kind}_scale"],
            weights[f"hc_{kind}_base"],
            weights[norm_name],
            config["rms_norm_eps"],
            config["hc_eps"],
            config["hc_sinkhorn_iters"],
        )
    flat = residual.flatten(1).float()
    fn = weights[f"hc_{kind}_fn"].float()
    if residual.device.type == "npu":
        mixes = (flat[:, None, :] * fn[None, :, :]).sum(-1)
    else:
        mixes = flat @ fn.t()
    mixes = mixes * torch.rsqrt(flat.square().mean(-1, keepdim=True) + config["rms_norm_eps"])
    scale, base = weights[f"hc_{kind}_scale"].float(), weights[f"hc_{kind}_base"].float()
    pre = torch.sigmoid(mixes[:, :STREAMS] * scale[0] + base[:STREAMS]) + config["hc_eps"]
    post = 2.0 * torch.sigmoid(mixes[:, STREAMS : 2 * STREAMS] * scale[1] + base[STREAMS : 2 * STREAMS])
    logits = (mixes[:, 2 * STREAMS :] * scale[2] + base[2 * STREAMS :]).reshape(-1, STREAMS, STREAMS)
    if native_sinkhorn is not None:
        comb = native_sinkhorn(logits.contiguous(), config["hc_sinkhorn_iters"], config["hc_eps"])
    else:
        comb = torch.softmax(logits, -1) + config["hc_eps"]
        comb = comb / (comb.sum(-2, keepdim=True) + config["hc_eps"])
        for _ in range(config["hc_sinkhorn_iters"] - 1):
            comb = comb / (comb.sum(-1, keepdim=True) + config["hc_eps"])
            comb = comb / (comb.sum(-2, keepdim=True) + config["hc_eps"])
    layer_input = round_state((pre[..., None] * residual).sum(1), torch)
    return (
        round_state(post[..., None], torch),
        round_state(comb, torch),
        rms_norm(layer_input, weights[norm_name], config["rms_norm_eps"], torch),
    )


def recurrent_kda(q, k, v, gate, beta, state, torch):
    """FP32 reference recurrence, state indexed [head, value, key]."""
    result = []
    for row in range(q.shape[0]):
        qr, kr, vr = q[row].float(), k[row].float(), v[row].float()
        qr = qr * torch.rsqrt(qr.square().sum(-1, keepdim=True) + L2_EPS) * (q.shape[-1] ** -0.5)
        kr = kr * torch.rsqrt(kr.square().sum(-1, keepdim=True) + L2_EPS)
        state.mul_(torch.exp(gate[row])[:, None, :])
        innovation = (vr - (state * kr[:, None, :]).sum(-1)) * beta[row, :, None]
        state.add_(innovation[:, :, None] * kr[:, None, :])
        result.append((state * qr[:, None, :]).sum(-1).to(q.dtype))
    return torch.stack(result)


def prefill_convolution(history, kernel, torch):
    """Reduce each four-tap window in FP32, round to FP16, then apply SiLU."""
    windows = history.unfold(-1, kernel.shape[-1], 1)
    convolved = (windows.float() * kernel[:, None, :]).sum(-1)
    return torch.nn.functional.silu(convolved.to(history.dtype)).t().contiguous()


class TextLayer:
    """One owner-local attention cache and complete decoder-layer computation."""

    def __init__(
        self,
        index,
        weights,
        config,
        device,
        torch,
        experts,
        max_context,
        *,
        native_sinkhorn=None,
        tp_rank=None,
        vectorized_prefill_conv=False,
    ):
        self.index, self.weights, self.config = index, weights, config
        self.device, self.torch, self.experts = device, torch, experts
        self.max_context, self.native_sinkhorn = max_context, native_sinkhorn
        self.tp_rank = tp_rank
        self.vectorized_prefill_conv = vectorized_prefill_conv
        self.heads = HEADS // 2 if tp_rank is not None else HEADS
        self.linear = config["layer_types"][index] == "linear_attention"
        self.state, self.conv, self.cache = None, None, None
        self.length = 0
        self.prepare()

    def prepare(self):
        t, w = self.torch, self.weights
        for kind in ("attn", "ffn"):
            for suffix in ("fn", "base", "scale"):
                name = f"hc_{kind}_{suffix}"
                w[name] = w[name].float()
        if self.linear:
            names = ("q_proj", "k_proj", "v_proj", "b_proj", "f_a_proj", "g_a_proj")
            w["self_attn.in_proj"] = t.cat([w.pop(f"self_attn.{name}.weight") for name in names]).contiguous()
            w["self_attn.conv"] = t.cat([w.pop(f"self_attn.{name}_conv1d.weight") for name in ("q", "k", "v")])
            self.state = t.zeros((self.heads, HEAD_DIM, HEAD_DIM), dtype=t.float32, device=self.device)
            self.conv = t.zeros((3 * self.heads * HEAD_DIM, 3), dtype=t.float16, device=self.device)
        else:
            from .layer_indexer import PoolIndexer

            self.cache = t.empty((self.max_context, self.config["kv_lora_rank"]), dtype=t.float16, device=self.device)
            self.indexer = PoolIndexer(w, self.config, self.device, t, self.max_context)
            self.sparse_enabled = self.max_context > INDEX_TOP_K
        if self.index >= 3:
            w["mlp.gate.weight"] = w["mlp.gate.weight"].float()
            w["mlp.gate.e_score_correction_bias"] = w["mlp.gate.e_score_correction_bias"].float()
        if self.device.startswith("npu"):
            import torch_npu

            prepare_projection_layout(w, torch_npu.npu_format_cast)

    def reset(self):
        self.length = 0
        if self.linear:
            self.state.zero_()
            self.conv.zero_()
        else:
            self.indexer.reset()

    def project(self, x, name):
        return self.torch.nn.functional.linear(x, self.weights[name])

    def kda(self, x):
        t, w = self.torch, self.weights
        projected = self.project(x, "self_attn.in_proj")
        heads = self.heads
        qkv, b, fa, ga = projected.split((3 * heads * HEAD_DIM, heads, HEAD_DIM, HEAD_DIM), -1)
        history = t.cat((self.conv, qkv.t()), -1)
        kernel = w["self_attn.conv"].view(3 * heads * HEAD_DIM, 4).float()
        if self.vectorized_prefill_conv and self.device == "cuda" and x.shape[0] > 1:
            convolved_rows = prefill_convolution(history, kernel, t)
        else:
            rows = []
            for offset in range(x.shape[0]):
                convolved = (history[:, offset : offset + 4].float() * kernel).sum(-1)
                # Conv kernels round the accumulation before SiLU.
                rows.append(t.nn.functional.silu(convolved.to(x.dtype)))
            convolved_rows = t.stack(rows)
        self.conv.copy_(history[:, -3:])
        q, k, v = convolved_rows.reshape(-1, 3, heads, HEAD_DIM).unbind(1)
        raw_gate = self.project(fa, "self_attn.f_b_proj.weight").reshape(-1, heads, HEAD_DIM).float()
        gate = LOWER_BOUND * t.sigmoid(
            t.exp(w["self_attn.A_log"].float())[None, :, None]
            * (raw_gate + w["self_attn.dt_bias"].float().view(heads, HEAD_DIM))
        )
        if self.device == "cuda":
            from .cuda_layer_ops import recurrent

            output = recurrent(q, k, v, gate, t.sigmoid(b.float()), self.state)
        else:
            output = recurrent_kda(q, k, v, gate, t.sigmoid(b.float()), self.state, t)
        g = self.project(ga, "self_attn.g_b_proj.weight").reshape(-1, heads, HEAD_DIM)
        output = (
            rms_norm(output, w["self_attn.o_norm.weight"], self.config["rms_norm_eps"], t).float()
            * t.sigmoid(g.float())
        ).to(x.dtype)
        return self.project(output.flatten(1), "self_attn.o_proj.weight")

    def dsa_prepare(self, x):
        t, w, c = self.torch, self.weights, self.config
        heads = (
            c["num_attention_heads"] // 2 if getattr(self, "tp_rank", None) is not None else c["num_attention_heads"]
        )
        count = x.shape[0]
        q = rms_norm(
            self.project(x, "self_attn.q_a_proj.weight"), w["self_attn.q_a_layernorm.weight"], c["rms_norm_eps"], t
        )
        qr = q
        q = self.project(q, "self_attn.q_b_proj.weight").view(count, heads, c["qk_nope_head_dim"])
        kv = rms_norm(
            self.project(x, "self_attn.kv_a_proj_with_mqa.weight"),
            w["self_attn.kv_a_layernorm.weight"],
            c["rms_norm_eps"],
            t,
        )
        # Absorb K/V projections without expanding persistent KV into all heads.
        wb = w["self_attn.kv_b_proj.weight"].view(heads, c["qk_nope_head_dim"] + c["v_head_dim"], c["kv_lora_rank"])
        wk, _ = wb.split((c["qk_nope_head_dim"], c["v_head_dim"]), 1)
        qlatent = t.bmm(q.transpose(0, 1), wk).transpose(0, 1)
        return qlatent, qr, kv

    def dsa_attend(self, x, qlatent, qr, kv):
        t, c = self.torch, self.config
        count = x.shape[0]
        if self.length + count > self.max_context:
            raise ValueError("owner-local DSA cache capacity exceeded")
        self.cache[self.length : self.length + count].copy_(kv)
        selected = self.indexer.select(x, qr, self.length) if getattr(self, "sparse_enabled", False) else None
        if selected is None:
            keys = self.cache[: self.length + count]
            scores = t.matmul(qlatent.float(), keys.float().t()) * (c["qk_nope_head_dim"] ** -0.5)
            positions = t.arange(count, device=self.device) + self.length
            causal = t.arange(keys.shape[0], device=self.device)[None, :] > positions[:, None]
            scores.masked_fill_(causal[:, None, :], -float("inf"))
        else:
            keys = self.cache[selected.clamp_min(0)]
            scores = t.bmm(qlatent.float(), keys.float().transpose(1, 2)) * (c["qk_nope_head_dim"] ** -0.5)
            scores.masked_fill_((selected < 0)[:, None, :], -float("inf"))
        probabilities = t.softmax(scores, -1).to(x.dtype)
        latent = t.matmul(probabilities, keys) if selected is None else t.bmm(probabilities, keys)
        self.length += count
        return latent

    def dsa_project(self, latent):
        t, w, c = self.torch, self.weights, self.config
        heads = (
            c["num_attention_heads"] // 2 if getattr(self, "tp_rank", None) is not None else c["num_attention_heads"]
        )
        wb = w["self_attn.kv_b_proj.weight"].view(heads, c["qk_nope_head_dim"] + c["v_head_dim"], c["kv_lora_rank"])
        _, wv = wb.split((c["qk_nope_head_dim"], c["v_head_dim"]), 1)
        output = t.bmm(latent.transpose(0, 1), wv.transpose(1, 2)).transpose(0, 1).flatten(1)
        return self.project(output, "self_attn.o_proj.weight")

    def dsa(self, x):
        if self.length + x.shape[0] > self.max_context:
            raise ValueError("owner-local DSA cache capacity exceeded")
        return self.dsa_project(self.dsa_attend(x, *self.dsa_prepare(x)))

    def dense_mlp(self, x, prefix):
        t = self.torch
        gate = self.project(x, f"{prefix}.gate_proj.weight").clamp(max=self.config["swiglu_limit"])
        up = self.project(x, f"{prefix}.up_proj.weight").clamp(
            -self.config["swiglu_limit"], self.config["swiglu_limit"]
        )
        return self.project((t.nn.functional.silu(gate.float()) * up.float()).to(x.dtype), f"{prefix}.down_proj.weight")

    def prepare_attention(self, hidden, residual=None, post=None, comb=None):
        t, w, c = self.torch, self.weights, self.config
        if post is None:
            if self.index != 0:
                raise ValueError("noninitial layer requires complete deferred mHC state")
            residual = hidden[:, None, :].expand(-1, STREAMS, -1).float().contiguous()
        else:
            residual = mhc_post(hidden, residual, post, comb, t)
        post, comb, x = mhc_pre(residual, w, "attn", "input_layernorm.weight", c, t, self.native_sinkhorn)
        return x, residual, post, comb

    def finish_attention(self, x, residual, post, comb):
        t, w, c = self.torch, self.weights, self.config
        if self.tp_rank is not None:
            x = x.float()
            t.distributed.all_reduce(x)
            x = x.to(t.float16)
        residual = mhc_post(x, residual, post, comb, t)
        post, comb, x = mhc_pre(residual, w, "ffn", "post_attention_layernorm.weight", c, t, self.native_sinkhorn)
        return x, residual, post, comb

    def prepare_forward(self, hidden, residual=None, post=None, comb=None):
        x, residual, post, comb = self.prepare_attention(hidden, residual, post, comb)
        x = self.kda(x) if self.linear else self.dsa(x)
        return self.finish_attention(x, residual, post, comb)

    def route(self, x):
        t, w, c = self.torch, self.weights, self.config
        # Keep routing FP32; 310P requires explicit operand reduction.
        gate = w["mlp.gate.weight"]
        logits = (
            (x.float()[:, None, :] * gate[None, :, :]).sum(-1)
            if self.device.startswith("npu")
            else x.float() @ gate.t()
        )
        scores = t.sigmoid(logits)
        ids = t.topk(scores + w["mlp.gate.e_score_correction_bias"], TOP_K, -1).indices
        route_weights = scores.gather(-1, ids)
        route_weights = route_weights / route_weights.sum(-1, keepdim=True) * c["routed_scaling_factor"]
        return ids, route_weights

    def finish_forward(self, x, residual, post, comb, ids=None, route_weights=None):
        if self.index < 3:
            x = self.dense_mlp(x, "mlp")
        else:
            if ids is None:
                ids, route_weights = self.route(x)
            if hasattr(self.experts, "with_shared"):
                routed, shared = self.experts.with_shared(self, x, ids, route_weights)
            else:
                routed = self.experts(self.index, x, ids, route_weights)
                shared = self.dense_mlp(x, "mlp.shared_experts")
            return self.finish_experts(x, residual, post, comb, routed, shared)
        return self.finish_mlp(x, residual, post, comb)

    def finish_experts(self, x, residual, post, comb, routed, shared):
        combined = routed.float() + shared.float()
        if self.tp_rank is not None:
            self.torch.distributed.all_reduce(combined)
        return self.finish_mlp(combined.to(x.dtype), residual, post, comb)

    def finish_mlp(self, x, residual, post, comb):
        t, c = self.torch, self.config
        if self.index == c["num_hidden_layers"] - 1:
            # Normalize the contracted residual in FP32 before converting the
            # bounded normalized result to the head dtype. FP16 here would
            # overflow valid BF16-range residuals before RMSNorm can act.
            return mhc_post(x, residual, post, comb, t).mean(1), None, None, None
        return x, residual, post, comb

    def forward(self, hidden, residual=None, post=None, comb=None):
        return self.finish_forward(*self.prepare_forward(hidden, residual, post, comb))
