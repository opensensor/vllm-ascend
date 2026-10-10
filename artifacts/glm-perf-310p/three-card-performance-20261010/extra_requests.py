"""Exercise concurrent serving and a longer generated Python function."""

import ast
import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def main():
    root = Path("artifacts/glm-perf-310p/three-card-performance-20261010/variant")

    def call(text, max_tokens=64):
        data = {
            "model": "three-card-w3",
            "messages": [{"role": "user", "content": text}],
            "temperature": 0,
            "max_tokens": max_tokens,
            "thinking_token_budget": 0,
        }
        q = urllib.request.Request(
            "http://127.0.0.1:18701/v1/chat/completions",
            data=json.dumps(data).encode(),
            headers={"Content-Type": "application/json"},
        )
        t = time.monotonic()
        with urllib.request.urlopen(q, timeout=300) as r:
            v = json.loads(r.read())
        assert v["usage"]["completion_tokens_details"]["reasoning_tokens"] == 0
        return {"elapsed_s": time.monotonic() - t, "response": v}

    values = ["391", "527", "684", "913"]
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda x: call("Output exactly the integer " + x + " and nothing else."), values))
    for expected, row in zip(values, results):
        assert row["response"]["choices"][0]["message"]["content"].strip() == expected, row
    (root / "concurrent-answer-gate.json").write_text(
        json.dumps({"expected": values, "results": results}, indent=2) + "\n"
    )
    print("FOUR_CONCURRENT_DISTINCT_ANSWERS_PASS", flush=True)
    result = call(
        "Write a Python function is_prime(n: int) -> bool. Return False below 2 and handle 2 correctly. "
        "Use trial division up to the square root. Output only the function, no explanation.",
        256,
    )
    text = result["response"]["choices"][0]["message"]["content"].strip()
    code = text.split("```python", 1)[1].split("```", 1)[0] if "```python" in text else text
    if code.startswith("```"):
        code = code.split("```", 2)[1]
    tree = ast.parse(code)
    assert len(tree.body) == 1 and isinstance(tree.body[0], ast.FunctionDef) and tree.body[0].name == "is_prime"
    assert not any(isinstance(x, (ast.Import, ast.ImportFrom)) for x in ast.walk(tree))
    # Exercise the simple generated function only after its structure is checked.
    scope = {"__builtins__": {"int": int, "bool": bool, "range": range}}
    exec(compile(tree, "<glm-prime-gate>", "exec"), scope)
    expected = {
        -1: False,
        0: False,
        1: False,
        2: True,
        3: True,
        4: False,
        9: False,
        17: True,
        23: True,
        25: False,
        49: False,
        97: True,
    }
    for value, answer in expected.items():
        assert scope["is_prime"](value) == answer, (value, answer)
    result["functional_cases"] = expected
    (root / "coding-prime-gate.json").write_text(json.dumps(result, indent=2) + "\n")
    print("CODING_PRIME_FUNCTION_12_CASES_PASS", flush=True)


if __name__ == "__main__":
    main()
