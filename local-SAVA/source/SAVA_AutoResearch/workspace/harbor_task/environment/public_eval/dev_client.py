"""Request protected Dev scoring without network or privilege escalation."""
import json,os,pathlib,sys,time,uuid
if len(sys.argv)!=1:raise SystemExit('FORMAT: evaluator accepts no arguments')
root=pathlib.Path('/tmp/.public-valuation');nonce=uuid.uuid4().hex
startup_deadline=time.monotonic()+15
while not (root/'ready').is_file():
 if time.monotonic()>startup_deadline:raise TimeoutError('Protected evaluator service did not become ready')
 time.sleep(.05)
request=root/(nonce+'.request');response=root/(nonce+'.response')
fd=os.open(request,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o644)
with os.fdopen(fd,'w') as f:f.write('{"evaluate":true}')
deadline=time.monotonic()+7200
while not response.exists():
 if time.monotonic()>deadline:raise TimeoutError('Protected evaluator exceeded its evaluation budget')
 time.sleep(.1)
response_fd=os.open(response,os.O_RDONLY|os.O_NOFOLLOW)
with os.fdopen(response_fd,'r') as f:
 info=os.fstat(f.fileno())
 # Native user namespaces map the outside root owner to overflow UID 65534.
 # The unprivileged Actor cannot create or chown files to either accepted UID.
 if info.st_uid not in [0,65534] or info.st_mode & 0o022:raise PermissionError('Response is not owned by the protected service')
 value=json.loads(f.read())
print(json.dumps(value,indent=2),flush=True)
(root/(nonce+'.ack')).write_text('')
if value.get('status')!='ok':raise SystemExit(1)
