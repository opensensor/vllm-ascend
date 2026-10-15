import json
import pathlib
import subprocess

ROOT = pathlib.Path('/srv/ai/storage/artifacts/npu-collective-qfn-20261014-024609')
rows = json.loads((ROOT/'ple-driver-deployment-v2/modules.json').read_text())
result = {'boot_id': pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip(), 'modules': []}
for row in rows:
    expected = subprocess.check_output(['modinfo', '-F', 'srcversion', row['source']], text=True).strip()
    actual = pathlib.Path('/sys/module', row['name'], 'srcversion').read_text().strip()
    result['modules'].append({'name': row['name'], 'expected': expected, 'loaded': actual, 'matched': actual == expected})
result['passed'] = all(row['matched'] for row in result['modules']) and len(rows) == 31
(ROOT/'ple-driver-loaded-modules-v2.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps({'passed': result['passed'], 'modules': len(rows), 'boot_id': result['boot_id']}))
assert result['passed']
