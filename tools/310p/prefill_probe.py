import json, time, urllib.request, sys

URL = "http://127.0.0.1:8080/v1/chat/completions"

def run(prompt, max_tokens=1):
    b = json.dumps({"model": "qwen27b",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": max_tokens, "temperature": 0}).encode()
    r = urllib.request.Request(URL, b, {"Content-Type": "application/json"})
    t0 = time.time()
    d = json.load(urllib.request.urlopen(r, timeout=900))
    return time.time() - t0, d["usage"]["prompt_tokens"], d["usage"]["completion_tokens"]

SNIP = ("def compute_slot_mapping(block_table, positions, block_size):\n"
        "    return block_table[positions // block_size] * block_size + positions % block_size\n")

# warm up the engine (short prompt, unrelated content so nothing is cached)
run("Say OK.", 1)

for reps in (180, 360):
    body = SNIP * reps
    prompt = "Here is a file:\n```python\n" + body + "```\nReply with only the word OK."
    # unique prefix each run to defeat prefix caching
    for trial in range(2):
        p = ("# run %d-%d\n" % (reps, trial)) + prompt
        dt, n, c = run(p, 1)
        print("reps=%-4d trial=%d  prompt=%6d tok  %6.2fs  ->  %7.1f tok/s prefill"
              % (reps, trial, n, dt, n / dt), flush=True)
