"""Trusted CUDA runtime preload followed by UID drop and seccomp enforcement."""
import argparse,ctypes,errno,json,os,sys,time,hashlib,importlib.util
from pathlib import Path

p=argparse.ArgumentParser();p.add_argument('--uid',type=int,required=True);p.add_argument('--mode',choices=['probe','run'],default='probe');p.add_argument('--jail',type=Path,required=True)
p.add_argument('--seed',type=int,default=17);p.add_argument('--reload',action='store_true')
p.add_argument('--checkpoint-fd',type=int);p.add_argument('--prediction-fd',type=int);p.add_argument('--receipt-fd',type=int);a=p.parse_args()
start=time.monotonic()
import numpy as np
import scanpy as sc
import torch
from sklearn.preprocessing import MaxAbsScaler
import threading
from tqdm import tqdm
tqdm.set_lock(threading.RLock())

torch.set_num_threads(4)
# Only fixed public synthetic kernels are run before entering the jail. No
# candidate module, private data, pretrained model or credential is loaded.
assert not torch.cuda.is_initialized()
assert torch.cuda.is_available()
warm=torch.randn(16,16,device='cuda',requires_grad=True)
(warm@warm.T).square().mean().backward();torch.cuda.synchronize()
warm_optimizer=torch.optim.AdamW([warm],lr=0.0001)
warm_optimizer.step()
import torch._dynamo
del warm;torch.cuda.empty_cache()
old_prefix=sys.prefix;old_base=sys.base_prefix
def guest_path(value):
    if not isinstance(value,str):return value
    for original,replacement in [(old_prefix,'/opt/runtime'),(old_base,'/opt/python'),(str(a.jail),'')]:
        if value.startswith(original):return replacement+value[len(original):]
    return value
for module in list(sys.modules.values()):
    if module is None:continue
    for name in ['__file__','__cached__']:
        value=module.__dict__.get(name)
        if isinstance(value,str):module.__dict__[name]=guest_path(value)
    value=module.__dict__.get('__path__')
    if value is not None:module.__dict__['__path__']=[guest_path(x) for x in value]
    spec=module.__dict__.get('__spec__')
    if spec is not None:
        spec.origin=guest_path(spec.origin)
        if spec.submodule_search_locations is not None:spec.submodule_search_locations=[guest_path(x) for x in spec.submodule_search_locations]
        if hasattr(spec.loader,'path'):spec.loader.path=guest_path(spec.loader.path)
sys.path=[guest_path(x) for x in sys.path]
sys.prefix=sys.exec_prefix='/opt/runtime';sys.base_prefix=sys.base_exec_prefix='/opt/python'
# Never carry an external directory handle through chroot.
import stat
for fd in range(3,256):
    try:
        mode=os.fstat(fd).st_mode
        if stat.S_ISDIR(mode) or stat.S_ISSOCK(mode):os.close(fd)
    except OSError:pass
os.chroot(a.jail);os.chdir('/workspace')
os.environ.update(HOME='/tmp',TMPDIR='/tmp',MPLCONFIGDIR='/tmp/mpl',NUMBA_CACHE_DIR='/tmp/numba',XDG_CACHE_HOME='/tmp/cache')
import tempfile
tempfile.tempdir='/tmp'
os.setgroups([]);os.setresgid(a.uid,a.uid,a.uid);os.setresuid(a.uid,a.uid,a.uid)
libc=ctypes.CDLL(None,use_errno=True)
assert libc.prctl(38,1,0,0,0)==0
assert os.getresuid()==(a.uid,)*3 and os.getresgid()==(a.uid,)*3
assert torch.cuda.is_available(),'CUDA unavailable inside unprivileged task jail'
torch.cuda.init();first=torch.arange(16,device='cuda',dtype=torch.float32).reshape(4,4)
before=(first@first).flatten().cpu().tolist();assert before[:4]==[56,62,68,74]
del first;torch.cuda.empty_cache()

lib=ctypes.CDLL('/lib/x86_64-linux-gnu/libseccomp.so.2',use_errno=True)
class Cmp(ctypes.Structure):
    _fields_=[('arg',ctypes.c_uint),('op',ctypes.c_int),('datum_a',ctypes.c_uint64),('datum_b',ctypes.c_uint64)]
lib.seccomp_init.argtypes=[ctypes.c_uint32];lib.seccomp_init.restype=ctypes.c_void_p
lib.seccomp_syscall_resolve_name.argtypes=[ctypes.c_char_p];lib.seccomp_syscall_resolve_name.restype=ctypes.c_int
lib.seccomp_rule_add_array.argtypes=[ctypes.c_void_p,ctypes.c_uint32,ctypes.c_int,ctypes.c_uint,ctypes.POINTER(Cmp)]
lib.seccomp_load.argtypes=[ctypes.c_void_p];lib.seccomp_release.argtypes=[ctypes.c_void_p]
lib.seccomp_attr_set.argtypes=[ctypes.c_void_p,ctypes.c_uint,ctypes.c_uint32]
ctx=lib.seccomp_init(0x7fff0000);assert ctx
assert lib.seccomp_attr_set(ctx,4,1)==0,'All-thread seccomp synchronization is required'
def rule(name,code=errno.EPERM,cmp=None):
    number=lib.seccomp_syscall_resolve_name(name.encode())
    if number>=0:
        arr=(Cmp*len(cmp))(*cmp) if cmp else None
        assert lib.seccomp_rule_add_array(ctx,0x00050000|code,number,len(cmp or []),arr)==0,name
denied='''socket socketpair connect bind listen accept accept4 sendto recvfrom sendmsg recvmsg sendmmsg recvmmsg
shutdown setsockopt getsockopt fork vfork execve execveat ptrace process_vm_readv process_vm_writev
kill tkill pidfd_open pidfd_getfd pidfd_send_signal mount umount2 pivot_root chroot unshare setns
open_by_handle_at name_to_handle_at bpf perf_event_open keyctl add_key request_key io_uring_setup
io_uring_enter io_uring_register setuid setgid setreuid setregid setresuid setresgid setfsuid setfsgid
setgroups capset reboot kexec_load kexec_file_load init_module finit_module delete_module swapon swapoff
userfaultfd mknod mknodat quotactl fanotify_init sched_setaffinity creat mkdir mkdirat rmdir unlink unlinkat
rename renameat renameat2 link linkat symlink symlinkat chmod fchmod fchmodat chown fchown lchown fchownat'''.split()
for name in denied:rule(name)
# CUDA and BLAS may create threads, but cannot create another process.
rule('clone',cmp=[Cmp(0,7,0x10000,0)])
rule('clone3',code=errno.ENOSYS)
rule('openat2',code=errno.ENOSYS)
for name,arg in [('open',1),('openat',2)]:
    rule(name,cmp=[Cmp(arg,7,os.O_CREAT,os.O_CREAT)])
assert lib.seccomp_load(ctx)==0;lib.seccomp_release(ctx)

checks={}
for label,fn in [
 ('network_denied',lambda:__import__('socket').socket()),
 ('process_creation_denied',lambda:os.fork()),
 ('filesystem_creation_denied',lambda:Path('/tmp/candidate-write').write_text('x')),
 ('affinity_change_denied',lambda:os.sched_setaffinity(0,{min(os.sched_getaffinity(0))}))]:
    try:fn();checks[label]=False
    except PermissionError:checks[label]=True
checks['host_proc_environ_absent']=not Path('/proc/self/environ').exists()
checks['host_private_root_absent']=not Path('/root').exists()
checks['no_supplementary_groups']=os.getgroups()==[]
torch.manual_seed(17);x=torch.randn(16,16,device='cuda',requires_grad=True)
loss=(x@x.T).square().mean();loss.backward();torch.cuda.synchronize()
checks['cuda_forward_backward_finite']=bool(torch.isfinite(loss)) and bool(torch.isfinite(x.grad).all())
checks['cpu_four_affinity']=len(os.sched_getaffinity(0))==4
assert all(checks.values()),checks
print(json.dumps({'phase':'UNPRIVILEGED_CHROOT_SECCOMP_CUDA_PROBE_PASSED','checks':checks,
 'uid':os.getuid(),'gid':os.getgid(),'torch':torch.__version__,'gpu_name':torch.cuda.get_device_name(0),
 'elapsed_seconds':time.monotonic()-start,'candidate_imported_by_host':False,'formal_research_seconds':0}),flush=True)

if a.mode=='run':
    # The immutable TensorDataset is read synchronously. This does not change
    # batch size, sample order, optimizer updates or the 200-epoch recipe;
    # equivalence is checked against the already completed expert runs.
    OriginalDataLoader=torch.utils.data.DataLoader
    class SingleProcessDataLoader(OriginalDataLoader):
        def __init__(self,*args,**kwargs):
            args=list(args)
            if len(args)>5:args[5]=0;kwargs.pop('num_workers',None)
            else:kwargs['num_workers']=0
            if len(args)>11:args[11]=None;kwargs.pop('multiprocessing_context',None)
            else:kwargs['multiprocessing_context']=None
            kwargs['persistent_workers']=False;kwargs['prefetch_factor']=None
            super().__init__(*args,**kwargs)
    torch.utils.data.DataLoader=SingleProcessDataLoader
    # No target labels exist in this process or jail. Only these two fixed inputs.
    torch.manual_seed(a.seed);np.random.seed(a.seed)
    torch.cuda.reset_peak_memory_stats()
    train=sc.read_h5ad('/data/train.h5ad');controls=sc.read_h5ad('/data/controls.h5ad')
    assert list(train.var_names)==list(controls.var_names)
    scaler=MaxAbsScaler();train.X=scaler.fit_transform(train.X);controls.X=scaler.transform(controls.X)
    sys.path.insert(0,'/opt/backend')
    spec=importlib.util.spec_from_file_location('candidate_method','/workspace/method.py')
    method=importlib.util.module_from_spec(spec);spec.loader.exec_module(method)
    for name in ['build_model','train_model','predict']:
        if not callable(getattr(method,name,None)):raise ValueError('Missing method callable: '+name)
    model=method.build_model(train,torch.device('cuda:0'),a.seed)
    before=time.monotonic()
    if a.reload:
        model.model.load_state_dict(torch.load('/inputs/model.pt',map_location='cuda:0',weights_only=True),strict=True)
        model.is_trained=True
    else:method.train_model(model,a.seed)
    training_seconds=0.0 if a.reload else time.monotonic()-before
    with os.fdopen(a.checkpoint_fd,'wb') as f:
        f.seek(0);f.truncate();torch.save(model.model.state_dict(),f)
    prediction=np.asarray(method.predict(model,controls,scaler,a.seed+10000),dtype=np.float64)
    if prediction.shape!=controls.shape or not np.isfinite(prediction).all() or prediction.min()<0:
        raise ValueError('Hard gate: expected finite nonnegative control-cells x genes prediction')
    # Output destinations are pre-opened by the trusted supervisor; no filesystem
    # creation permission is granted to candidate code or the runtime wrapper.
    with os.fdopen(a.prediction_fd,'wb') as f:
        f.seek(0);f.truncate();np.save(f,prediction,allow_pickle=False)
    receipt={'phase':'SANDBOX_CANDIDATE_EXECUTION_FINISHED','mode':'reload' if a.reload else 'fresh_training',
      'seed':a.seed,'inference_seed':a.seed+10000,'method_sha256':hashlib.sha256(Path('/workspace/method.py').read_bytes()).hexdigest(),
      'training_shape':list(train.shape),'prediction_shape':list(prediction.shape),
      'training_seconds':training_seconds,'elapsed_seconds':time.monotonic()-start,
      'device':'cuda:0','torch':torch.__version__,'gpu_name':torch.cuda.get_device_name(0),
      'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(),'uid':os.getuid(),
      'candidate_imported_by_host':False,'isolation_checks':checks}
    with os.fdopen(a.receipt_fd,'w') as f:f.seek(0);f.truncate();json.dump(receipt,f,indent=2)
