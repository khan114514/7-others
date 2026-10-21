"""Materialize the exact tested Linux ARM64 vendor binaries for an operator.

Binaries are supplied separately by an authorized operator. This command does
not redistribute, download, execute, or establish a license for vendor assets.
Input locations and matched bytes never appear in the receipt or errors.
"""
import argparse
import hashlib
import json
import os
import shutil
import stat
from pathlib import Path

HERE=Path(__file__).resolve().parent


def provision(source,output):
    manifest=json.loads((HERE/'binary_manifest.json').read_text())
    dockerfile=(HERE/'Dockerfile').read_bytes()
    if hashlib.sha256(dockerfile).hexdigest()!=manifest['dockerfile_sha256']:
        raise ValueError('Frozen Dockerfile digest mismatch')
    if output.exists() or output.is_symlink():raise ValueError('Fresh private output directory required')
    if source.is_symlink() or not source.is_dir():raise ValueError('Trusted source directory required')
    verified=[]
    for item in manifest['external_binaries']:
        name=item['name']
        if name not in {'codex','codex-code-mode-host','rg'}:raise ValueError('Unexpected binary manifest entry')
        path=source/name;info=path.lstat()
        if not stat.S_ISREG(info.st_mode) or info.st_uid!=os.getuid():raise ValueError('Operator-owned regular vendor binary required')
        if info.st_size!=item['bytes']:raise ValueError('Vendor binary size mismatch')
        fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
        with os.fdopen(fd,'rb') as stream:
            opened=os.fstat(stream.fileno())
            if (opened.st_dev,opened.st_ino)!=(info.st_dev,info.st_ino):raise ValueError('Vendor source changed during validation')
            head=stream.read(64);stream.seek(0);digest=hashlib.sha256()
            while chunk:=stream.read(1024*1024):digest.update(chunk)
            final=os.fstat(stream.fileno())
        if (opened.st_size,opened.st_mtime_ns)!=(final.st_size,final.st_mtime_ns):raise ValueError('Vendor source changed during validation')
        if head[:4]!=b'\x7fELF' or head[4:6]!=b'\x02\x01' or int.from_bytes(head[18:20],'little')!=183:
            raise ValueError('Linux ARM64 ELF64 binary required')
        if digest.hexdigest()!=item['sha256']:raise ValueError('Vendor binary digest mismatch')
        verified.append(item)
    if {v['name'] for v in verified}!={'codex','codex-code-mode-host','rg'}:raise ValueError('Complete three-binary set required')
    output.mkdir(mode=0o700,parents=False)
    try:
        for item in verified:
            destination=output/item['name']
            # Verify the copied bytes independently before they are executable.
            with destination.open('xb') as target, (source/item['name']).open('rb') as origin:
                shutil.copyfileobj(origin,target,1024*1024)
            destination.chmod(0o600);digest=hashlib.sha256()
            with destination.open('rb') as stream:
                while chunk:=stream.read(1024*1024):digest.update(chunk)
            if destination.stat().st_size!=item['bytes'] or digest.hexdigest()!=item['sha256']:
                raise ValueError('Materialized vendor digest mismatch')
            destination.chmod(0o755)
        (output/'Dockerfile').write_bytes(dockerfile)
        result={'phase':'EXACT_OPERATOR_VENDOR_BINARIES_MATERIALIZED_BUILD_PENDING',
                'architecture':'Linux ARM64','codex_version':manifest['codex_version'],
                'files':verified,'dockerfile_sha256':manifest['dockerfile_sha256'],
                'binary_execution_occurred':False,'new_machine_reproduction_claim':False,
                'redistribution_permission_claim':False,'model_research_seconds':0}
        (output/'provision-receipt.json').write_text(json.dumps(result,indent=2)+'\n')
        return result
    except Exception:
        # Partial materialization remains private for diagnosis, not a release.
        (output/'INCOMPLETE').write_text('Provisioning incomplete; inspect original private invocation receipt.\n')
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir',required=True,type=Path)
    parser.add_argument('--output-dir',required=True,type=Path)
    args=parser.parse_args()
    try:value=provision(args.source_dir,args.output_dir)
    except Exception as error:
        print(json.dumps({'phase':'FAILED_CLOSED','error_type':type(error).__name__,
                          'input_locations_withheld':True,'model_research_seconds':0}))
        return 1
    print(json.dumps(value));return 0


if __name__=='__main__':raise SystemExit(main())
