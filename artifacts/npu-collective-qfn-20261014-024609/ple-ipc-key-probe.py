import ctypes as c,hashlib,json,sys
import torch,torch_npu
rank=int(sys.argv[1]);torch.npu.set_device(rank);torch.empty(1,device='npu')
acl=c.CDLL('/usr/local/Ascend/ascend-toolkit/latest/lib64/libascendcl.so')
ptr=c.c_void_p;out=c.POINTER(ptr);size=c.c_size_t
for name,args in [('aclrtMalloc',[out,size,c.c_int]),('aclrtCreateNotify',[out,c.c_uint64]),('aclrtIpcMemGetExportKey',[ptr,size,ptr,size,c.c_uint64]),('aclrtNotifyGetExportKey',[ptr,ptr,size,c.c_uint64])]:
 fn=getattr(acl,name);fn.argtypes=args;fn.restype=c.c_int
keys={};owners=[]
for name in ('inbox0','inbox1','ready0','ready1'):
 p=ptr();key=c.create_string_buffer(65)
 if name.startswith('inbox'):
  assert acl.aclrtMalloc(c.byref(p),5120,3)==0
  assert acl.aclrtIpcMemGetExportKey(p,5120,key,65,0)==0
 else:
  assert acl.aclrtCreateNotify(c.byref(p),0)==0
  assert acl.aclrtNotifyGetExportKey(p,key,65,0)==0
 owners.append(p);keys[name]=hashlib.sha256(key.value).hexdigest()
print(json.dumps(dict(rank=rank,key_hashes=keys,aliases=[[a,b] for i,a in enumerate(keys) for b in list(keys)[i+1:] if keys[a]==keys[b]])),flush=True)
