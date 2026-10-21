import pathlib,json,hashlib,os
root=pathlib.Path(__file__).resolve().parent/'hidden-chunks';m=json.loads((root/'manifest.json').read_text());target=pathlib.Path('/tests/hidden/hidden-decoder-inputs-v1.safetensors');target.parent.mkdir(parents=True,exist_ok=True);tmp=target.with_suffix('.pending');whole=hashlib.sha256();size=0
with tmp.open('xb') as output:
 for spec in m['chunks']:
  source=root/spec['path'];raw=source.read_bytes()
  if len(raw)!=spec['bytes'] or hashlib.sha256(raw).hexdigest()!=spec['sha256']:raise ValueError('Hidden chunk differs')
  output.write(raw);whole.update(raw);size+=len(raw)
if size!=m['bytes'] or whole.hexdigest()!=m['sha256']:raise ValueError('Hidden combined identity differs')
tmp.chmod(0o644);tmp.replace(target)
print('VERIFIER_HIDDEN_BYTES_AND_SHA_CLOSED')
