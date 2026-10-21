"""Bounded data-only transfer for declared files in live Docker tmpfs mounts."""
import base64
import hashlib
import json
import os
import stat
import subprocess
from pathlib import Path

LIMIT = 512 * 1024
READ_CODE = '''import os,stat,sys,json,base64,hashlib
p=sys.argv[1];fd=os.open(p,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
try:
 s=os.fstat(fd)
 if not stat.S_ISREG(s.st_mode) or s.st_size>524288:raise ValueError('Artifact must be a bounded regular file')
 data=b''
 while len(data)<=524288:
  b=os.read(fd,min(65536,524289-len(data)))
  if not b:break
  data+=b
 if len(data)>524288:raise ValueError('Artifact size exceeded')
 after=os.fstat(fd);now=os.lstat(p)
 if (s.st_ino,s.st_size,s.st_mtime_ns)!=(after.st_ino,after.st_size,after.st_mtime_ns) or now.st_ino!=s.st_ino:raise ValueError('Artifact changed during collection')
 print(json.dumps({'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),'data':base64.b64encode(data).decode()}))
finally:os.close(fd)
'''
WRITE_CODE = '''import os,stat,sys,json,base64,hashlib,uuid
m=json.load(sys.stdin);p=sys.argv[1];data=base64.b64decode(m['data'],validate=True)
if len(data)>524288 or len(data)!=m['bytes'] or hashlib.sha256(data).hexdigest()!=m['sha256']:raise ValueError('Artifact transport binding mismatch')
parent=os.path.dirname(p)
if not stat.S_ISDIR(os.lstat(parent).st_mode):raise ValueError('Artifact parent is not a real directory')
pending=p+'.transfer-'+uuid.uuid4().hex
fd=os.open(pending,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o644)
try:
 offset=0
 while offset<len(data):offset+=os.write(fd,data[offset:])
 os.fsync(fd)
finally:os.close(fd)
os.replace(pending,p)
print(json.dumps({'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}))
'''


def envelope(data):
    if len(data) > LIMIT:
        raise ValueError('Artifact exceeds the declared transfer budget')
    return {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest(),
            'data': base64.b64encode(data).decode()}


def read_host_regular(path):
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_size > LIMIT:
            raise ValueError('Source must be a bounded regular artifact')
        data = bytearray()
        while len(data) <= LIMIT:
            chunk = os.read(fd, min(65536, LIMIT + 1 - len(data)))
            if not chunk:
                break
            data.extend(chunk)
        return envelope(bytes(data))
    finally:
        os.close(fd)


def exec_json(docker, container, code, path, user, payload=None):
    command = [docker, 'exec', '-i', '--user', user, '--workdir', '/',
               container, 'python', '-I', '-c', code, path]
    result = subprocess.run(command, input=json.dumps(payload or {}).encode(),
                            capture_output=True, timeout=15)
    if result.returncode:
        raise RuntimeError('Bounded artifact transfer failed: ' + result.stderr.decode(errors='replace')[-1000:])
    if len(result.stdout) > 1024 * 1024:
        raise ValueError('Artifact transport response exceeded its budget')
    return json.loads(result.stdout)


def download(docker, container, source, target):
    result = exec_json(docker, container, READ_CODE, source, '1000:1000')
    data = base64.b64decode(result.pop('data'), validate=True)
    if len(data) > LIMIT or len(data) != result['bytes'] or hashlib.sha256(data).hexdigest() != result['sha256']:
        raise ValueError('Downloaded artifact binding mismatch')
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    pending = target.with_name(target.name + '.transfer-pending')
    with pending.open('xb') as handle:
        handle.write(data)
    pending.replace(target)
    return result


def upload(docker, container, source, target, user='1000:1000'):
    payload = read_host_regular(source)
    result = exec_json(docker, container, WRITE_CODE, target, user, payload)
    if result != {k: payload[k] for k in ['bytes', 'sha256']}:
        raise ValueError('Uploaded artifact binding mismatch')
    return result
