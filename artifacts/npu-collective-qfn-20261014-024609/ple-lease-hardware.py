import dataclasses,errno,fcntl,json,os,pathlib,sys
from ple_driver import DeviceDMALease,ACQUIRE,QUERY,RELEASE,LEASE_STRUCT,ABI_VERSION
out=pathlib.Path(sys.argv[1]); result={'passed':False,'devices':[]}
def rejected(fn,expected):
 try:fn()
 except OSError as e:
  assert e.errno==expected,(e.errno,expected)
  return e.errno
 raise AssertionError('unexpected success')
def request(dev,version=1):return bytearray(LEASE_STRUCT.pack(version,dev,0,0,0,0,0,0,0,0,0))
def snapshot():
 paths=sorted(pathlib.Path('/sys/bus/pci/devices').glob('*/msg/device_dma'))
 assert len(paths)==2, paths
 rows=[]
 for path in paths:
  selector=path.parent.parent/'davinci_dev_id'
  original=selector.read_text().strip()
  try:
   for bank in (0,1):
    selector.write_text(str(bank))
    rows.append(path.read_text().strip())
  finally:selector.write_text(original)
 return rows
leases=[]
try:
 result['before']=snapshot()
 assert all('reserved=0' in x for x in result['before'])
 fd=os.open('/dev/ascend_device_dma',os.O_RDWR|os.O_CLOEXEC)
 try:
  result['bad_version_errno']=rejected(lambda:fcntl.ioctl(fd,ACQUIRE,request(0,99),True),errno.EINVAL)
  result['missing_query_errno']=rejected(lambda:fcntl.ioctl(fd,QUERY,request(0),True),errno.ENODEV)
 finally:os.close(fd)
 for rank in range(4):
  lease=DeviceDMALease.acquire(rank);leases.append(lease)
  assert lease.query()==lease.info and lease.info.capabilities==0
  row=dataclasses.asdict(lease.info)
  row['duplicate_errno']=rejected(lambda:DeviceDMALease.acquire(rank),errno.EBUSY)
  stale=bytearray(LEASE_STRUCT.pack(1,rank,0,lease.info.channel,lease.info.generation+1,0,lease.info.sq_depth,lease.info.cq_depth,0,0,0))
  row['stale_errno']=rejected(lambda:fcntl.ioctl(lease._fd,RELEASE,stale,True),errno.ESTALE)
  assert lease.query()==lease.info
  result['devices'].append(row)
 result['simultaneous_leases']=len(leases)
 result['reserved']=snapshot()
 assert all('reserved=1' in x for x in result['reserved'])
 for lease,row in zip(leases,result['devices']):
  before=lease.info.generation;lease.release()
  with DeviceDMALease.acquire(row['device_id']) as replacement:
   assert replacement.info.generation>before
   row['reacquired_generation']=replacement.query().generation
  with DeviceDMALease.acquire(row['device_id']) as closed:
   row['close_cleanup_generation']=closed.query().generation
 result['after']=snapshot()
 assert all('reserved=0' in x for x in result['after'])
 result['passed']=True
finally:
 for lease in leases:lease.close()
 out.write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps(result,indent=2),flush=True)
