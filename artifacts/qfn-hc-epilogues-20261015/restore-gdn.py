"""Reattach the previously qualified output binary after the unrelated engine loss."""


def operation(wrapper, check_only=False):
    worker = getattr(wrapper, "worker", wrapper)
    try:
        import hashlib
        import importlib
        import json
        from pathlib import Path

        import torch

        from tools.qwen4exp.gdn_output_native import GDNOutputNative, build_forward

        module = importlib.import_module("vllm_ascend._310p.ops.fla.chunk_gated_delta_rule")
        root = Path("/srv/ai/storage/artifacts/qfn-high-end-20261015")
        binary = root / "gdn-output-build-v1/native-gdn-output.bin"
        owner = getattr(worker, "_qfn_gdn_output_adapter", None)
        if owner is None:
            prior = json.loads((root / f"gdn-output-qualified-v1/rank{worker.rank}-gate.json").read_text())
            assert prior["passed"]
            assert hashlib.sha256(binary.read_bytes()).hexdigest() == (
                "85787bd1f5494939deb9eb25a1106f050dc8f31e6029e72989560951824068f4"
            )
            kernel = torch.classes.qwen_checkpoint_preparation_v1.Kernel(str(binary), "qwen_gdn_output_aligned_v1")
            worker._qfn_restored_gdn_kernel = kernel
            owner = GDNOutputNative(kernel, torch.ops._C_ascend.chunk_fwd_o_vllm, worker.device)
            worker._qfn_gdn_output_adapter = owner
            owner.original_function = module.chunk_gated_delta_rule_310
            owner.replacement = build_forward(owner.original_function, owner, prior["source_sha256"])
            with torch.no_grad():
                q = torch.zeros(1, 4, 64, 128, dtype=torch.float16, device=worker.device)
                v = torch.zeros(1, 12, 64, 128, dtype=torch.float16, device=worker.device)
                h = torch.zeros(1, 12, 1, 128, 128, dtype=torch.float16, device=worker.device)
                g = torch.zeros(1, 12, 64, device=worker.device)
                passed = torch.equal(owner(q, q, v, h, 128**-0.5, g=g), v)
            owner.receipt = dict(
                passed=passed,
                source_sha256=prior["source_sha256"],
                prior_full_qualification=str(root / "gdn-output-qualified-v1"),
                scope="same binary/source restored; fresh zero-input launch smoke",
            )
        assert owner.receipt["passed"]
        assert module.chunk_gated_delta_rule_310 is owner.original_function
        if not check_only:
            module.chunk_gated_delta_rule_310 = owner.replacement
        return dict(rank=worker.rank, passed=True, receipt=owner.receipt, status=owner.status())
    except Exception as error:
        return dict(rank=worker.rank, passed=False, error=repr(error))


def admission(worker):
    return operation(worker, check_only=True)


class Candidate:
    def validate(self, engine):
        assert engine._pause_complete() and not engine.batch_queue
        assert not engine.scheduler.get_num_unfinished_requests()
        if not hasattr(self, "rows"):
            self.rows = engine.model_executor.collective_rpc(admission)
        return dict(passed=len(self.rows) == 4 and all(row["passed"] for row in self.rows), workers=self.rows)

    def apply(self, engine):
        self.validate(engine)
        rows = engine.model_executor.collective_rpc(operation)
        return dict(passed=len(rows) == 4 and all(row["passed"] for row in rows), workers=rows)


def prepare(engine):
    return Candidate()
