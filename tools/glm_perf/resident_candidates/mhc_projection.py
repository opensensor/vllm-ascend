# SPDX-License-Identifier: Apache-2.0
"""Prepare mHC weights while idle and gate the complete vector epilogue."""

import torch

MIN_ROWS = 640
REAL_GATE_ROWS = (640, 1280)
REAL_GATE_SCALES = (1.0, 16.0)
FP16_ULP_LIMIT = 2


class PreparedProjection:
    def __init__(self, original, projection):
        self.original = original
        self.projection = projection
        self.accepted = set()
        self.records = []
        self.calls = {}

    def __getattr__(self, name):
        return getattr(self.original, name)

    def prepare_model(self, runner):
        roots = [runner.model]
        draft = getattr(getattr(runner, "drafter", None), "model", None)
        if draft is not None:
            roots.append(draft)
        branches = {}
        for root in roots:
            root = getattr(root, "runnable", root)
            for layer in root.modules():
                if not hasattr(layer, "mhc_pre_op"):
                    continue
                for name, norm in (("attn", layer.input_layernorm), ("ffn", layer.post_attention_layernorm)):
                    fn = getattr(layer, "hc_" + name + "_fn")
                    params = (
                        layer.mhc_sinkhorn_iterations,
                        layer.rms_norm_eps,
                        layer.hc_eps,
                        layer.hc_eps,
                        layer.mhc_post_mult_value,
                        norm.variance_epsilon,
                    )
                    branches[fn.data_ptr()] = (layer, name, norm, fn, params)
        if not branches:
            raise ValueError("no qualified mHC branches found for prepared projections")
        self.projection.prepare([values[3] for values in branches.values()], range(MIN_ROWS, 1281))
        with torch.inference_mode():
            for layer, name, norm, fn, params in branches.values():
                allowed = True
                scale = getattr(layer, "hc_" + name + "_scale")
                base = getattr(layer, "hc_" + name + "_base")
                for rows in REAL_GATE_ROWS:
                    for magnitude in REAL_GATE_SCALES:
                        generator = torch.Generator(device=fn.device).manual_seed(1011 + rows)
                        residual = (
                            (torch.randn(rows, 4, 4096, device=fn.device, generator=generator) * magnitude)
                            .half()
                            .float()
                        )
                        expected = self.original(residual, fn, scale, base, norm.weight, params)
                        mixes = self.projection(residual, fn)
                        actual = self.original.epilogue(residual, mixes, scale, base, norm.weight, params)
                        metrics = []
                        for a, b in zip(actual, expected):
                            ac, bc = a.half().cpu(), b.half().cpu()
                            delta = (ac.float() - bc.float()).abs()
                            distance = (ac.view(torch.int16).int() - bc.view(torch.int16).int()).abs()
                            valid = bool(torch.isfinite(ac).all()) and not bool(
                                ((distance > FP16_ULP_LIMIT) & (delta > 1e-7)).any()
                            )
                            allowed = allowed and valid
                            metrics.append(dict(passed=valid, max_abs=float(delta.max()), max_ulp=int(distance.max())))
                        self.records.append(
                            dict(layer=layer.layer_idx, branch=name, rows=rows, magnitude=magnitude, outputs=metrics)
                        )
                if allowed:
                    self.accepted.add(fn.data_ptr())
        if not self.accepted:
            raise ValueError("no prepared projection passed the loaded-weight epilogue gate")

    def __call__(self, residual, fn, scale, base, gamma, parameters):
        rows = residual.shape[0]
        if fn.data_ptr() in self.accepted and rows in self.projection.configs and rows >= MIN_ROWS:
            self.calls[rows] = self.calls.get(rows, 0) + 1
            mixes = self.projection(residual, fn)
            return self.original.epilogue(residual, mixes, scale, base, gamma, parameters)
        return self.original(residual, fn, scale, base, gamma, parameters)


def extend_replacements(parent_replacements, native_resources):
    projection = native_resources["mhc_projection_v1011"]
    proxy = PreparedProjection(native_resources["mhc_pre_vector_v1005_corrected_reference"], projection)
    selected = dict(native_resources, mhc_pre_vector_v1005_corrected_reference=proxy)
    changes = parent_replacements(selected)
    prefix = "vllm_ascend._310p.worker_310p:NPUWorker310."
    capture = changes[prefix + "resident_capture"]
    parent_status = changes[prefix + "resident_status"]

    def capture_model(self):
        try:
            if not proxy.records:
                proxy.prepare_model(self.model_runner)
                torch.npu.empty_cache()
            receipt = capture(self)
            if "error" not in receipt:
                torch.npu.reset_peak_memory_stats()
            return receipt
        except Exception as error:
            self._resident_session().graphs_dirty = True
            return self._resident_error(error)

    def status(self):
        row = parent_status(self)
        row["prepared_mhc_projection"] = dict(
            accepted_weights=len(proxy.accepted),
            prepared_weights=len(projection.weights),
            prepared_bytes=sum(t.untyped_storage().nbytes() for t in projection.weights.values()),
            tile_k=projection.tile_k,
            calls_by_rows=dict(proxy.calls),
            cases=proxy.records,
            scope=(
                "FP16-rounded input and weights, FP32 Cube accumulation/output; "
                "loaded-weight epilogue gate <=2 FP16 ULP, not language quality"
            ),
        )
        return row

    result = dict(changes)
    result[prefix + "resident_capture"] = capture_model
    result[prefix + "resident_status"] = status
    return result
