import ast
import json
from pathlib import Path

from tools.minimax_m3.tool_probe import chat_request, check_calls, tool

out = Path("/workspace/m31-int8-short-checks")
out.mkdir(exist_ok=True)
common = {
    "model": "minimax-m31-flash",
    "temperature": 0,
    "max_tokens": 96,
    "stream": True,
    "stream_options": {"include_usage": True},
}
payload = {
    **common,
    "messages": [{"role": "user", "content": "Call lookup_code for city Paris. Do not guess the tool result."}],
    "tools": [tool("lookup_code", {"city": {"type": "string"}})],
    "tool_choice": "auto",
}
r = chat_request("http://127.0.0.1:8000", payload)
r["passed"] = check_calls(r, [("lookup_code", {"city": "Paris"})])
(out / "tool.json").write_text(json.dumps(r, indent=2) + "\n")
print("tool", r["passed"], r["message"], flush=True)
payload = {
    **common,
    "messages": [
        {"role": "user", "content": "Return only Python code for a function add(a, b) that returns their sum."}
    ],
}
r = chat_request("http://127.0.0.1:8000", payload)
code = r["message"].get("content", "").strip()
code = code.removeprefix("```python").removeprefix("```").removesuffix("```").strip()
try:
    tree = ast.parse(code)
    r["passed"] = any(
        isinstance(n, ast.FunctionDef)
        and n.name == "add"
        and any(
            isinstance(z, ast.Return) and isinstance(z.value, ast.BinOp) and isinstance(z.value.op, ast.Add)
            for z in ast.walk(n)
        )
        for n in ast.walk(tree)
    )
except SyntaxError:
    r["passed"] = False
(out / "code.json").write_text(json.dumps(r, indent=2) + "\n")
print("code", r["passed"], r["message"], flush=True)
