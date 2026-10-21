"""Run a current-source synthetic GPU/isolation probe under existing GPU locks."""
import argparse,hashlib,json,os,shutil,subprocess,time,uuid
from pathlib import Path

p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--trace',action='store_true');a=p.parse_args()
root=a.root.resolve();base=root/'private/gpu-jail-runtime-v2'
build=json.loads((root/'private/gpu-jail-runtime-v2-build.json').read_text())
active=root/'private/gpu-jail-runtime-v2-active-entry.json'
if active.exists():build['entry_sha256']=json.loads(active.read_text())['entry_sha256']
assert hashlib.sha256((base/'runner.py').read_bytes()).hexdigest()==build['entry_sha256']
identifier=uuid.uuid4().hex;folder=root/'private/gpu-jails'/identifier
folder.parent.mkdir(mode=0o700,exist_ok=True)
shutil.copytree(base,folder,copy_function=os.link,symlinks=True)
uid=54000+int(identifier[:4],16)%9000
argv=[str(root/'preparation/remote_venv/bin/python'),str(root/'preparation/gpu_jail_launcher.py'),
 '--jail',str(folder),'--root',str(root),'--uid',str(uid),'--mode','probe']
out=root/'private/gpu-jail-probes'/identifier;out.mkdir(parents=True,mode=0o700)
if a.trace:
    argv=['/usr/bin/strace','-D','-f','-e','trace=openat,access,readlink,ioctl,mmap,statfs,fstatfs','-o',str(out/'syscalls.original.log'),*argv]
start=time.monotonic()
def own_proc_snapshot(pid):
    # Only this subprocess's maps and safe device/runtime descriptors are exposed.
    # No live proc mount, environ, cwd/root magic links, peer PIDs or host fd paths.
    proc=folder/'proc';selfdir=proc/'self';selfdir.mkdir(exist_ok=True)
    numbered=proc/str(pid);numbered.mkdir(exist_ok=True)
    source=Path('/proc')/str(pid)
    if not source.exists():return
    maps=(source/'maps').read_text().replace(str(folder),'')
    for dest in [selfdir/'maps',numbered/'maps']:
        tmp=dest.with_suffix('.pending');tmp.write_text(maps);tmp.chmod(0o644);tmp.replace(dest)
    fdroot=selfdir/'fd';fdroot.mkdir(exist_ok=True)
    for f in (source/'fd').iterdir():
        try:target=os.readlink(f)
        except FileNotFoundError:continue
        if target.startswith(str(folder)):
            target=target[len(str(folder)):]
        if not (target.startswith('/dev/nvidia') or target.startswith('/opt/') or target=='/dev/null'):continue
        dst=fdroot/f.name
        if not dst.exists() and not dst.is_symlink():dst.symlink_to(target)
with (out/'stdout.original.log').open('xb') as stdout,(out/'stderr.original.log').open('xb') as stderr:
    proc=subprocess.Popen(argv,env={'PATH':'/usr/bin:/bin','LANG':'C.UTF-8'},stdin=subprocess.DEVNULL,
      stdout=stdout,stderr=stderr,close_fds=True,start_new_session=True)
    while proc.poll() is None:
        if time.monotonic()-start>120:
            proc.kill();proc.wait();raise TimeoutError('Synthetic GPU probe timeout')
        try:own_proc_snapshot(proc.pid)
        except (FileNotFoundError,ProcessLookupError):pass
        time.sleep(.01)
report={'probe_id':identifier,'phase':'GPU_JAIL_SYNTHETIC_PROBE_PASSED' if proc.returncode==0 else 'GPU_JAIL_SYNTHETIC_PROBE_FAILED',
 'exit_code':proc.returncode,'elapsed_seconds':time.monotonic()-start,'uid':uid,
 'entry_sha256':build['entry_sha256'],'candidate_code_imported_on_host':False,'formal_research_credit':0}
if proc.returncode==0:
    rows=(out/'stdout.original.log').read_text().splitlines();report['actual']=json.loads(rows[-1])
(out/'receipt.json').write_text(json.dumps(report,indent=2))
(root/'private/gpu-jail-probe-latest.json').write_text(json.dumps({'probe_dir':str(out),'receipt':report},indent=2))
print(json.dumps(report,indent=2));raise SystemExit(proc.returncode)
