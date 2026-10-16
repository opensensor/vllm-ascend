import ast
import unittest
from pathlib import Path


class NativeGraphTest(unittest.TestCase):
    def test_native_graphs_skip_acl_task_updates_and_acl_graphs_delegate(self):
        source = Path(__file__).resolve().parents[3] / "vllm_ascend/_310p/model_runner_310p.py"
        tree = ast.parse(source.read_text())
        runner = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "NPUModelRunner310")
        method = next(
            node
            for node in runner.body
            if isinstance(node, ast.FunctionDef) and node.name == "_update_full_graph_params_if_needed"
        )

        class Parent:
            def _update_full_graph_params_if_needed(self, context, tokens):
                self.calls.append((context, tokens))

        # Execute the actual override with a CPU-only parent. Importing the
        # complete runner would require NPU/vLLM even for this dispatch check.
        definition = ast.ClassDef(
            name="Runner",
            bases=[ast.Name(id="Parent", ctx=ast.Load())],
            keywords=[],
            body=[method],
            decorator_list=[],
        )
        module = ast.fix_missing_locations(ast.Module(body=[definition], type_ignores=[]))
        namespace = {"Parent": Parent, "ForwardContext": object}
        exec(compile(module, str(source), "exec"), namespace)
        instance = namespace["Runner"]()
        instance.calls = []
        context = object()
        instance.use_aclgraph = False
        instance._update_full_graph_params_if_needed(context, 4)
        self.assertEqual(instance.calls, [])
        instance.use_aclgraph = True
        instance._update_full_graph_params_if_needed(context, 4)
        self.assertEqual(instance.calls, [(context, 4)])
