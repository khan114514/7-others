"""Restore checksum-bound trusted worker files; never import or start the worker."""
import argparse
import hashlib
import json
import os
import stat
from pathlib import Path,PurePosixPath

MANIFEST_SHA = '3cb2b03642c02d0c0076e858aa2bd44c3d62290b5bf29b401f7f3147cb2cf358'
MANIFEST_TENSOR_SHA = '61d7fabc0298197c99c63317cfc906360d3dfd06e78741cdf4ac728ecf5d94fe'
CODE = {'numerical_worker.py','checkpoint_loader.py','full_decoder_evaluator.py','tensor_dag.py','bounded_graph_compiler.py','baseline_ops.py','configuration_qwen2_5_vl.py'}
TARGETS = CODE | {'independent-audit/recompute_numerical_result.py','baseline-program.json','full-scout-inputs.safetensors','hidden-decoder-inputs-v1.safetensors'}


def regular_open(root,relative):
    if root.is_symlink() or not root.is_dir():raise ValueError('Input root must be a real directory')
    rel=PurePosixPath(relative)
    if rel.is_absolute() or not rel.parts or any(part in ['.','..'] for part in rel.parts):raise ValueError('Invalid declared relative path')
    current=root
    for part in rel.parts[:-1]:
        current=current/part
        if not stat.S_ISDIR(current.lstat().st_mode):raise ValueError('Declared parent must be a real directory')
    fd=os.open(root/relative,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
    info=os.fstat(fd)
    if not stat.S_ISREG(info.st_mode):os.close(fd);raise ValueError('Declared input must be regular')
    return fd,info


def check_file(root,relative,specification,output=None):
    fd,before=regular_open(root,relative)
    try:
        if before.st_size!=specification['bytes']:raise ValueError('Declared input byte count differs')
        sha=hashlib.sha256();count=0
        while True:
            chunk=os.read(fd,1024*1024)
            if not chunk:break
            sha.update(chunk);count+=len(chunk)
            if output is not None:output.write(chunk)
        after=os.fstat(fd);present=(root/relative).lstat()
        if ((before.st_ino,before.st_size,before.st_mtime_ns)!=(after.st_ino,after.st_size,after.st_mtime_ns)
                or present.st_ino!=before.st_ino or not stat.S_ISREG(present.st_mode)
                or count!=specification['bytes'] or sha.hexdigest()!=specification['sha256']):
            raise ValueError('Declared input SHA or read stability differs')
    finally:os.close(fd)


def read_descriptor(path):
    if path.is_symlink() or not path.is_file():raise ValueError('Descriptor must be regular')
    encoded=path.read_bytes()
    if len(encoded)>65536 or hashlib.sha256(encoded).hexdigest()!=MANIFEST_SHA:raise ValueError('Frozen deployment descriptor differs')
    value=json.loads(encoded)
    if (value['schema']!='bifrost-trusted-numerical-deployment/2' or value['manifest_sha256']!=MANIFEST_TENSOR_SHA
            or value['tensor_count']!=677 or set(value['files'])!=TARGETS or value['worker_code_file_count']!=7):
        raise ValueError('Trusted numerical source set differs')
    for name,item in value['files'].items():
        source=PurePosixPath(item['source'])
        if source.is_absolute() or '..' in source.parts or not isinstance(item['bytes'],int) or item['bytes']<=0:
            raise ValueError('Invalid declared source')
    return value


def verify(directory,descriptor):
    for name,item in descriptor['files'].items():check_file(directory,name,item)
    if {path.name for path in directory.glob('*.py')}!=CODE:raise ValueError('Worker code root has an unexpected Python module')
    return {'verified_files':len(descriptor['files']),'worker_code_files':len(CODE),'descriptor_sha256':MANIFEST_SHA}


def materialize(task,destination,descriptor):
    if destination.exists() or destination.is_symlink():raise ValueError('Deployment requires a new destination; use verify-only for an existing one')
    # Validate the entire declaration before creating any output directory.
    for item in descriptor['files'].values():check_file(task,item['source'],item)
    destination.mkdir(parents=True,exist_ok=False)
    for name,item in descriptor['files'].items():
        target=destination/name;target.parent.mkdir(parents=True,exist_ok=True)
        with target.open('xb') as output:check_file(task,item['source'],item,output)
    return verify(destination,descriptor)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True);parser.add_argument('--destination',type=Path,required=True)
    parser.add_argument('--task',type=Path);parser.add_argument('--verify-only',action='store_true')
    args=parser.parse_args();descriptor=read_descriptor(args.manifest)
    if args.verify_only:result=verify(args.destination,descriptor)
    else:
        if args.task is None:parser.error('--task is required for materialization')
        result=materialize(args.task,args.destination,descriptor)
    print(json.dumps({'status':'DECLARED_FILES_SHA_MATCH',**result,'worker_started':False,'model_or_GPU_called':False}))

if __name__=='__main__':main()
