"""Validate file-relative row descriptors against safetensors, without NPU work."""
import json
import os
import random
from pathlib import Path
from safetensors import safe_open
from tools.qwen4exp.ple_storage import register
ROOT=Path('/srv/ai/storage/artifacts/npu-collective-qfn-20261014-024609')
MODEL=Path('/srv/ai/models/Qwen3.8-Flash-Next-W4A16-G128-300i')
output=ROOT/'ple-storage-registration-v1.json'
if output.exists():raise FileExistsError(output)
index='model.safetensors.index.json'
if not (MODEL/index).exists():index='quant_model_weights.safetensors.index.json'
weights=json.loads((MODEL/index).read_text())['weight_map']
storage=register(MODEL,index_filename=index)
keys=[next(k for k in weights if k.endswith(f'ngram_embedding.shard_{s}.weight')) for s in range(128)]
rng=random.Random(14102026)
row_count=storage.shards[0].rows
batches=[list(range(s*row_count,(s*row_count)+2))+[(s+1)*row_count-1] for s in range(128)]
batches += [[rng.randrange(storage.rows) for _ in range(count)] for count in (16,48,96,192) for _ in range(4)]
checked=useful=fetched=0
for ids in batches:
 plan=storage.plan(ids)
 rows=[bytearray(storage.shards[0].row_bytes) for _ in ids]
 reads=[]
 for read in plan.reads:
  with open(storage.files[read.file].path,'rb',buffering=0) as handle:
   reads.append(os.pread(handle.fileno(),read.length,read.offset))
 for item in plan.slices:
  rows[item.destination_row][item.destination_offset:item.destination_offset+item.length]=reads[item.read][item.source_offset:item.source_offset+item.length]
 for row,actual in zip(ids,rows):
  shard,local=divmod(row,row_count)
  with safe_open(MODEL/weights[keys[shard]],framework='numpy') as file:
   reference=file.get_slice(keys[shard])[local:local+1].tobytes()
  assert actual==reference,(shard,local)
  checked+=1
 useful+=plan.useful_bytes;fetched+=plan.fetched_bytes
for file in storage.files:file.verify()
result=dict(passed=True,checked_rows=checked,useful_bytes=useful,page_read_bytes=fetched,page_read_amplification=fetched/useful,manifest=storage.manifest(),device_io_submitted=False,server_modified=False)
output.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='manifest'},indent=2))
