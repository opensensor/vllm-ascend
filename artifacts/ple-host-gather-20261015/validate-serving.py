import concurrent.futures
import json
import time
import urllib.request
from pathlib import Path

from tools.sidecard.qfn_resident_server import QfnResidentClient

ROOT = Path('/srv/ai/storage/artifacts/ple-host-gather-20261015')
client = QfnResidentClient('http://127.0.0.1:8001', ROOT / 'serving-v1/control.token', timeout=900)


def request(prompt, tokens=80):
    started = time.monotonic()
    payload = dict(model='qwen38-flash-next', messages=[dict(role='user', content=prompt)],
                   max_tokens=tokens, temperature=0, seed=0, chat_template_kwargs={'enable_thinking': False})
    req = urllib.request.Request('http://127.0.0.1:8001/v1/chat/completions',
                                 data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=900) as response:
        data = json.load(response)
    return dict(prompt=prompt, seconds=time.monotonic()-started, response=data)


result = {'before': client.rpc('resident_status')}
cases = [
    'What is the capital of France? Answer in one sentence.',
    'Calculate 17 times 23. Give the number and a brief explanation.',
    'Explain why the sky appears blue in two sentences.',
    'Write a Python function that returns the largest number in a nonempty list.',
]
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
    result['concurrent_verified'] = list(executor.map(request, cases))
result['verified'] = client.rpc('resident_status')
(ROOT / 'serving-validation-v1.json').write_text(json.dumps(result, indent=2))
for case in result['concurrent_verified']:
    print(json.dumps(case), flush=True)
client.request('/pause?mode=wait&clear_cache=false')
result['verification_disabled'] = client.rpc('qfn_cpu_profile_stop', 'native_ple_verify_off')
client.request('/resume')
result['warm'] = request('What is the capital of France? Answer in one sentence.')
result['unverified'] = [request('Explain how a CPU and an accelerator cooperate during neural network inference.', 128)]
result['after'] = client.rpc('resident_status')
assert all(not worker['native_ple']['failed'] and worker['native_ple']['checked'] > 4 for worker in result['after'])
assert len({worker['native_ple']['table_inode'] for worker in result['after']}) == 1
(ROOT / 'serving-validation-v1.json').write_text(json.dumps(result, indent=2))
print('Verified four-rank native serving, concurrent chat, and unverified warm request', flush=True)
for worker in result['after']:
    print(json.dumps({'rank': worker['rank'], **worker['native_ple']}), flush=True)
