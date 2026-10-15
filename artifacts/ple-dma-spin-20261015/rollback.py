#!/usr/bin/env python3
"""Restore the pre-profiling host modules/initramfs; reboot separately afterward."""
from pathlib import Path
import json, os, shutil, subprocess
root=Path(__file__).resolve().parent
meta=json.loads((root/'installation.json').read_text())
assert os.geteuid()==0, 'Run with sudo'
backup=root/'rollback'
cache=Path('/var/lib/dkms/davinci_ascend/1.0')/meta['kernel']/'x86_64/module'
for record in meta['modules']:
    installed=Path(record['installed'])
    for dest in [installed,cache/installed.name]:
        saved=backup/str(dest).lstrip('/')
        assert saved.is_file(),saved
for record in meta['modules']:
    installed=Path(record['installed'])
    for dest in [installed,cache/installed.name]:
        shutil.copy2(backup/str(dest).lstrip('/'),dest)
shutil.copy2(backup/'Module.symvers',cache/'Module.symvers')
source=Path('/usr/src/davinci_ascend-1.0')
temp=source.with_name(source.name+'.rollback')
temp.symlink_to(meta['old_dkms_source'])
os.replace(temp,source)
shutil.copy2(backup/('initrd.img-'+meta['kernel']),'/boot/initrd.img-'+meta['kernel'])
subprocess.run(['depmod','-a',meta['kernel']],check=True)
print('Original files restored. Reboot to activate them.')
