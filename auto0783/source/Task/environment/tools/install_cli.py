import pathlib,json,hashlib,tarfile,io,os
root=pathlib.Path(__file__).resolve().parent;m=json.loads((root/'manifest.json').read_text());buf=io.BytesIO();whole=hashlib.sha256()
for spec in m['chunks']:
 raw=(root/spec['path']).read_bytes()
 if len(raw)!=spec['bytes'] or hashlib.sha256(raw).hexdigest()!=spec['sha256']:raise ValueError('Public CLI chunk differs')
 buf.write(raw);whole.update(raw)
if buf.tell()!=m['archive_bytes'] or whole.hexdigest()!=m['archive_sha256']:raise ValueError('Public CLI compressed archive differs')
buf.seek(0)
with tarfile.open(fileobj=buf,mode='r:xz') as archive:
 if set(archive.getnames())!=set(m['binaries']):raise ValueError('CLI archive member set differs')
 for name,spec in m['binaries'].items():
  member=archive.getmember(name)
  if not member.isfile() or member.size!=spec['bytes']:raise ValueError('Public CLI member type or size differs')
  stream=archive.extractfile(member);target=pathlib.Path('/usr/local/bin')/name;pending=target.with_suffix('.pending');h=hashlib.sha256()
  with pending.open('xb') as out:
   while True:
    raw=stream.read(1024*1024)
    if not raw:break
    h.update(raw);out.write(raw)
  if h.hexdigest()!=spec['sha256']:raise ValueError('Public CLI executable identity differs')
  pending.chmod(0o755);pending.replace(target)
print('PUBLIC_CLI_ALL_THREE_ORIGINAL_EXECUTABLE_SHA_CLOSED')
