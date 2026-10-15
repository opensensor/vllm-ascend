import hashlib,json,os,pathlib,shutil,subprocess,sys,time
R=pathlib.Path('/srv/ai/storage/artifacts/npu-collective-qfn-20261014-024609')
D=R/'ple-driver-deployment-v1'
mode=sys.argv[1]
def run(*args,**kw): return subprocess.run(args,check=True,**kw)
if mode=='prepare':
 D.mkdir(mode=0o755)
 (D/'candidate').mkdir()
 run('tar','-xzf',str(R/'ple-driver-dynamic-v1.tar.gz'),'-C',str(D/'candidate'))
 manifest=json.loads((D/'candidate/validation.json').read_text())
 assert os.uname().release==manifest['kernel_release']
 records=[]
 for row in manifest['modules']:
  source=D/'candidate'/row['path']; assert hashlib.sha256(source.read_bytes()).hexdigest()==row['sha256']
  name=source.stem
  target=pathlib.Path(subprocess.check_output(['modinfo','-n',name],text=True).strip())
  assert str(target).startswith('/lib/modules/'+os.uname().release+'/updates/dkms/')
  backup=D/'backup'/target.name; backup.parent.mkdir(exist_ok=True)
  shutil.copy2(target,backup)
  records.append(dict(name=name,source=str(source),target=str(target),backup=str(backup),original_sha256=hashlib.sha256(backup.read_bytes()).hexdigest(),candidate_sha256=row['sha256']))
 (D/'modules.json').write_text(json.dumps(records,indent=2)+'\n')
 (D/'before-lsmod.txt').write_text(subprocess.check_output(['lsmod'],text=True))
 (D/'before-dmesg.txt').write_text(subprocess.check_output(['dmesg','-T'],text=True))
 shutil.copy2('/usr/local/sbin/ascend-310p-bringup',D/'bringup.sh')
 print('Prepared and verified rollback for',len(records),'modules',flush=True)
elif mode in ('install','rollback'):
 rows=json.loads((D/'modules.json').read_text())
 for row in rows:
  target=pathlib.Path(row['target']); tmp=target.with_name(target.name+'.new')
  if mode=='rollback':
   assert hashlib.sha256(pathlib.Path(row['backup']).read_bytes()).hexdigest()==row['original_sha256']
   shutil.copy2(row['backup'],tmp)
  else:
   assert hashlib.sha256(pathlib.Path(row['source']).read_bytes()).hexdigest()==row['candidate_sha256']
   assert target.suffix=='.zst'
   run('zstd','-q','-f',row['source'],'-o',str(tmp))
   tmp.chmod(0o644)
  os.replace(tmp,target)
 run('depmod','-a',os.uname().release)
 if mode=='rollback':
  run('tar','-xzf',str(D/'original-source-453e106.tar.gz'),'-C','/srv/ai/src/ascend-310p-host-driver','--no-same-owner')
  run('update-initramfs','-u','-k',os.uname().release)
 print(mode,'complete; modules on disk only',flush=True)
elif mode=='unload':
 names={r['name'] for r in json.loads((D/'modules.json').read_text())}
 while True:
  loaded={n for n in names if pathlib.Path('/sys/module',n).exists()}
  if not loaded: break
  leaves=[n for n in sorted(loaded) if not list(pathlib.Path('/sys/module',n,'holders').iterdir())]
  if not leaves: raise RuntimeError('No unloadable module: '+repr(loaded))
  for n in leaves:
   print('unloading',n,flush=True)
   run('modprobe','-r',n,timeout=40)
 print('All Ascend modules unloaded',flush=True)
else: raise ValueError(mode)
