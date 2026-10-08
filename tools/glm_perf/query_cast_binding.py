# SPDX-License-Identifier: Apache-2.0
"""Rebuild only Indexer's query cast from its authoritative class source."""

import ast


def bounded_converter(converter, max_elements):
    """Keep large prefills on their measured faster conversion path."""
    if type(max_elements) is not int or max_elements <= 0:
        raise ValueError("query cast limit must be a positive element count")

    def convert(value, dtype):
        if value.numel() > max_elements:
            return value.to(dtype)
        return converter(value, dtype)

    return convert


def wrap_forward(original, converter, source):
    tree = ast.parse(source)
    owner = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Indexer")
    method = next(node for node in owner.body if isinstance(node, ast.FunctionDef) and node.name == "forward")

    class QueryCast(ast.NodeTransformer):
        def __init__(self):
            self.count = 0

        def visit_Assign(self, node):
            node = self.generic_visit(node)
            if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name) or node.targets[0].id != "q":
                return node
            call = node.value
            if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Attribute) or call.func.attr != "to":
                return node
            if not isinstance(call.func.value, ast.Name) or call.func.value.id != "q":
                return node
            dtype = call.args[0] if len(call.args) == 1 and not call.keywords else None
            if not call.args and len(call.keywords) == 1 and call.keywords[0].arg == "dtype":
                dtype = call.keywords[0].value
            if (
                not isinstance(dtype, ast.Attribute)
                or dtype.attr != "bfloat16"
                or not isinstance(dtype.value, ast.Name)
                or dtype.value.id != "torch"
            ):
                raise ValueError("indexer query cast no longer converts only to BF16")
            self.count += 1
            node.value = ast.copy_location(
                ast.Call(ast.Name("_aicore_convert", ast.Load()), [ast.Name("q", ast.Load()), dtype], []), call
            )
            return node

    rewrite = QueryCast()
    method = rewrite.visit(method)
    if rewrite.count != 1:
        raise ValueError("indexer source must contain exactly one query BF16 cast")
    # The permanent indexer also replaces its RoPE casts. Rebuilding the
    # authoritative query method must retain those bound conversions instead
    # of silently returning them to torch's emulated BF16 path on 310P.
    permanent = original.__globals__.get("_aicore_convert")
    if permanent is not None:
        if not callable(permanent):
            raise ValueError("permanent indexer converter must be callable")

        class PermanentCasts(ast.NodeTransformer):
            def visit_Assign(self, node):
                node = self.generic_visit(node)
                if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name) or node.targets[0].id != "k_pe":
                    return node
                call = node.value
                if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Attribute):
                    return node
                source_value = call.func.value
                if (
                    call.func.attr == "to"
                    and isinstance(source_value, ast.Name)
                    and source_value.id == "k_pe"
                    and len(call.args) == 1
                    and not call.keywords
                    and ast.unparse(call.args[0]) == "torch.bfloat16"
                ):
                    dtype = call.args[0]
                elif (
                    call.func.attr == "float"
                    and isinstance(source_value, ast.Call)
                    and isinstance(source_value.func, ast.Attribute)
                    and source_value.func.attr == "reshape"
                    and isinstance(source_value.func.value, ast.Name)
                    and source_value.func.value.id == "k_pe"
                    and not call.args
                    and not call.keywords
                ):
                    dtype = ast.Attribute(ast.Name("torch", ast.Load()), "float32", ast.Load())
                else:
                    return node
                node.value = ast.copy_location(
                    ast.Call(ast.Name("_aicore_permanent_convert", ast.Load()), [source_value, dtype], []), call
                )
                return node

        method = PermanentCasts().visit(method)

    # Compile the function alone, avoiding class decorators or prior wrapper
    # metadata whose source locations may refer to another installed forward.
    module = ast.Module(body=[method], type_ignores=[])
    namespace = dict(original.__globals__, _aicore_convert=converter, _aicore_permanent_convert=permanent)
    exec(compile(ast.fix_missing_locations(module), "<glm-bound-query-cast>", "exec"), namespace)
    return namespace["forward"]
