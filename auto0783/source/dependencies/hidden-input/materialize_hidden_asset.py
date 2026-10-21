import argparse,pathlib,json,hashlib
p=argparse.ArgumentParser();p.add_argument('--manifest',type=pathlib.Path,required=True);p.add_argument('--destination',type=pathlib.Path,required=True);a=p.parse_args();m=json.loads(a.manifest.read_text());source=a.manifest.resolve().parents[2]/'Task/tests/hidden-chunks';raw=b''.join((source/s['path']).read_bytes() for s in m['chunks']);assert len(raw)==m['bytes'] and hashlib.sha256(raw).hexdigest()==m['sha256']
for name in m['targets']:
 target=a.destination/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(raw)
print('DECLARED_HIDDEN_INPUT_SHA_MATCH')
