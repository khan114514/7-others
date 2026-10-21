"""Build a task-owned runtime snapshot; no actor code or credentials copied."""
import argparse,hashlib,json,os,re,shutil,stat,subprocess,sys,time
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--version',default='v1',choices=['v1','v2']);a=p.parse_args()
    root=a.root.resolve();assert root.name=='save_autoresearch_20261009'
    base=root/('private/gpu-jail-runtime-'+a.version)
    if base.exists():raise RuntimeError('Use existing immutable runtime receipt; never overwrite')
    base.mkdir(mode=0o700)
    venv=root/'preparation/remote_venv'
    info=json.loads(subprocess.check_output([str(venv/'bin/python'),'-c',
       'import sys,json;print(json.dumps({"base":sys.base_prefix}))'],text=True))
    python=Path(info['base'])
    count=0;size=0
    def link_or_copy(src,dst):
        nonlocal count,size
        s=src.stat();assert stat.S_ISREG(s.st_mode)
        dst.parent.mkdir(parents=True,exist_ok=True)
        if s.st_uid==0 and not s.st_mode&0o022:
            try:os.link(src,dst)
            except OSError:shutil.copyfile(src,dst);dst.chmod(0o755 if s.st_mode&0o111 else 0o644)
        else:
            shutil.copyfile(src,dst)
            dst.chmod(0o755 if s.st_mode&0o111 else 0o644)
        assert dst.stat().st_uid==0 and not dst.stat().st_mode&0o022
        count+=1;size+=s.st_size
    def tree(src,dst):
        dst.mkdir(parents=True,exist_ok=True)
        for f in src.iterdir():
            if f.name=='__pycache__' or f.suffix=='.pyc':continue
            q=dst/f.name
            if f.is_symlink():
                target=os.readlink(f)
                if target.startswith('/'):raise RuntimeError('Unexpected absolute runtime symlink: '+str(f))
                q.symlink_to(target)
            elif f.is_dir():tree(f,q)
            else:link_or_copy(f,q)
    tree(venv/'lib',base/'opt/runtime/lib')
    tree(python/'lib',base/'opt/python/lib')
    tree(python/'bin',base/'opt/python/bin')
    bindir=base/'opt/runtime/bin';bindir.mkdir(parents=True)
    for name in ['python','python3','python3.10']:(bindir/name).symlink_to('/opt/python/bin/python3.10')
    (base/'opt/runtime/pyvenv.cfg').write_text('home = /opt/python/bin\ninclude-system-site-packages = false\nversion = 3.10.21\n')
    # ELF dependencies are resolved only from installed trusted libraries.
    seeds=[python/'bin/python3.10',Path('/lib/x86_64-linux-gnu/libseccomp.so.2')]
    seeds+=list((venv/'lib/python3.10/site-packages/numpy').glob('core/*.so'))
    seeds+=list((python/'lib/python3.10/lib-dynload').glob('*.so'))
    seeds+=[venv/'lib/python3.10/site-packages/torch/lib/libtorch_cpu.so',
            venv/'lib/python3.10/site-packages/torch/lib/libtorch_cuda.so']
    for directory in [Path('/lib/x86_64-linux-gnu'),Path('/usr/lib/x86_64-linux-gnu')]:
        for pattern in ['libcuda.so*','libnvidia-*.so*']:
            seeds+=list(directory.glob(pattern))
    seen=set();copied=[]
    while seeds:
        f=seeds.pop();key=str(f)
        if key in seen or not f.is_file():continue
        seen.add(key)
        if not (str(f).startswith(str(venv)) or str(f).startswith(str(python))):
            dst=base/str(f).lstrip('/')
            if not dst.exists():link_or_copy(f,dst);copied.append(str(f))
        r=subprocess.run(['/usr/bin/ldd',str(f)],capture_output=True,text=True,env={'PATH':'/usr/bin:/bin','LANG':'C.UTF-8'})
        for name in re.findall(r'(?:=>\s+)?(/[^\s]+)\s+\(',r.stdout):seeds.append(Path(name))
    shutil.copyfile(root/'preparation/gpu_jail_entry.py',base/'runner.py')
    for name in ['tmp','dev','proc','workspace','data']:(base/name).mkdir(exist_ok=True)
    for f in Path('/dev').glob('nvidia*'):
        if not f.is_char_device() or f.name=='nvidia-modeset':continue
        s=f.stat();os.mknod(base/'dev'/f.name,stat.S_IFCHR|0o666,s.st_rdev);(base/'dev'/f.name).chmod(0o666)
    for name in ['null','zero','random','urandom']:
        s=(Path('/dev')/name).stat();os.mknod(base/'dev'/name,stat.S_IFCHR|0o666,s.st_rdev);(base/'dev'/name).chmod(0o666)
    # Hardware snapshots only. No live proc tree, process metadata or environ.
    for name in ['cpuinfo','meminfo']:
        (base/'proc'/name).write_bytes((Path('/proc')/name).read_bytes())
    for f in base.rglob('*'):
        if f.is_dir():f.chmod(0o755)
    base.chmod(0o755)
    receipt={'phase':'TASK_OWNED_GPU_JAIL_RUNTIME_BUILT_NOT_YET_VERIFIED','files_linked_or_copied':count,
      'logical_runtime_bytes':size,'elf_dependencies':copied,'api_keys_or_ssh_material_copied':False,
      'candidate_or_hidden_target_files_copied':False,'runtime_python':'/opt/runtime/bin/python',
      'entry_sha256':hashlib.sha256((base/'runner.py').read_bytes()).hexdigest(),'completed_epoch':time.time()}
    (root/('private/gpu-jail-runtime-'+a.version+'-build.json')).write_text(json.dumps(receipt,indent=2));print(json.dumps({k:v for k,v in receipt.items() if k!='elf_dependencies'}))

if __name__=='__main__':main()
