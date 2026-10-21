import pathlib,argparse,json,hashlib,shutil
p=argparse.ArgumentParser();p.add_argument('--package',type=pathlib.Path,required=True);p.add_argument('--destination',type=pathlib.Path,required=True);p.add_argument('--evidence',type=pathlib.Path);p.add_argument('--verify-only',action='store_true');a=p.parse_args();m=json.loads((a.package/'dependencies/task/task-files.json').read_text())
def verify(root):
 files={f.relative_to(root).as_posix():f for f in root.rglob('*') if f.is_file()}
 if set(files)!=set(m['files']) or any(f.is_symlink() for f in root.rglob('*')):raise ValueError('Task file set differs')
 for n,f in files.items():
  raw=f.read_bytes();spec=m['files'][n]
  if len(raw)!=spec['bytes'] or hashlib.sha256(raw).hexdigest()!=spec['sha256']:raise ValueError('Task file differs')
 hashes={n:spec['sha256'] for n,spec in m['files'].items()}
 if hashlib.sha256(json.dumps(hashes,sort_keys=True,separators=(',',':')).encode()).hexdigest()!=m['materialized_task_manifest_sha256']:raise ValueError('Task manifest differs')
if a.verify_only:verify(a.destination)
else:
 verify(a.package/'Task')
 if a.destination.exists():raise ValueError('Use a new destination')
 shutil.copytree(a.package/'Task',a.destination);verify(a.destination)
result={'status':'EXACT_BUILD_READY_TASK_SOURCE_SHA_MATCH','files':m['task_file_count'],'task_manifest_sha256':m['materialized_task_manifest_sha256'],'candidate_executed':False}
if a.evidence:
 a.evidence.mkdir(parents=True,exist_ok=False);(a.evidence/'closure.json').write_text(json.dumps(result)+'\n')
print(json.dumps(result))
