import argparse
import json
import subprocess
import time
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('directory', type=Path)
args = parser.parse_args()
with (args.directory / 'npu-memory.jsonl').open('x') as output:
    while not (args.directory / 'stop-memory').exists():
        result = subprocess.run(['npu-smi', 'info'], capture_output=True, text=True, timeout=20)
        output.write(json.dumps({'time': time.time(), 'returncode': result.returncode, 'output': result.stdout}) + '\n')
        output.flush()
        time.sleep(10)
