"""Trusted fixed-input candidate supervisor; untrusted Python stays in its jail."""
import argparse,fcntl,hashlib,json,os,resource,shutil,signal,stat,subprocess,time,uuid
from pathlib import Path

def own_proc_snapshot(jail,pid):
    source=Path('/proc')/str(pid)
    if not source.exists():return
    for directory in [jail/'proc/self',jail/'proc'/str(pid)]:
        directory.mkdir(exist_ok=True)
        data=(source/'maps').read_text().replace(str(jail),'')
        temp=directory/'maps.pending';temp.write_text(data);temp.chmod(0o644);temp.replace(directory/'maps')
    fdroot=jail/'proc/self/fd';fdroot.mkdir(exist_ok=True)
    for f in (source/'fd').iterdir():
        try:target=os.readlink(f)
        except FileNotFoundError:continue
        if target.startswith(str(jail)):target=target[len(str(jail)):]
        if not (target.startswith('/dev/nvidia') or target.startswith('/opt/') or target=='/dev/null'):continue
        dest=fdroot/f.name
        if not dest.exists() and not dest.is_symlink():dest.symlink_to(target)

def copy_readonly(source,target):
    assert source.is_file() and not source.is_symlink()
    target.parent.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(source,target);target.chmod(0o644)

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--method',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--seed',type=int,required=True);p.add_argument('--split',choices=['dev','hidden'],default='dev')
    p.add_argument('--reload-checkpoint',type=Path);p.add_argument('--stage',required=True);a=p.parse_args()
    root=a.root.resolve();assert root.name=='save_autoresearch_20261009'
    assert 0<=a.seed<2**31 and a.method.is_file() and not a.method.is_symlink() and a.method.stat().st_size<=1024**2
    a.out.resolve().relative_to(root/'private');a.out.mkdir(parents=True,exist_ok=False)
    base=root/'private/gpu-jail-runtime-v2';actual=hashlib.sha256((base/'runner.py').read_bytes()).hexdigest()
    active=root/'private/gpu-jail-runtime-v2-active-entry.json'
    expected=json.loads((active if active.exists() else root/'private/gpu-jail-runtime-v2-build.json').read_text())['entry_sha256']
    assert actual==expected
    identifier=uuid.uuid4().hex;jail=root/'private/gpu-jails'/identifier
    jail.parent.mkdir(mode=0o700,exist_ok=True);shutil.copytree(base,jail,copy_function=os.link,symlinks=True)
    copy_readonly(a.method,jail/'workspace/method.py')
    task=root/'submission/workspace/harbor_task'
    shutil.copytree(task/'environment/starter',jail/'opt/backend',ignore=shutil.ignore_patterns('__pycache__'))
    train=task/'environment/public_assets/train.h5ad'
    controls=task/'environment/public_assets/dev_controls.h5ad' if a.split=='dev' else root/'private/hidden_calibration_assets/controls.h5ad'
    copy_readonly(train,jail/'data/train.h5ad');copy_readonly(controls,jail/'data/controls.h5ad')
    if a.reload_checkpoint:copy_readonly(a.reload_checkpoint,jail/'inputs/model.pt')
    uid_root=root/'private/gpu-uid-locks';uid_root.mkdir(mode=0o700,exist_ok=True)
    uid_handle=None
    for offset in range(9000):
        uid=54000+(int(identifier[:4],16)+offset)%9000
        candidate=(uid_root/str(uid)).open('a')
        try:fcntl.flock(candidate,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:candidate.close();continue
        occupied=False
        for proc_status in Path('/proc').glob('[0-9]*/status'):
            try:
                for line in proc_status.read_text().splitlines():
                    if line.startswith('Uid:') and uid in map(int,line.split()[1:]):occupied=True;break
            except (PermissionError,FileNotFoundError,ProcessLookupError):continue
            if occupied:break
        if occupied:candidate.close();continue
        uid_handle=candidate;break
    if uid_handle is None:raise RuntimeError('No exclusive unprivileged identity is available')
    outputs={name:os.open(a.out/name,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_CLOEXEC,0o600)
             for name in ['model.pt','prediction.npy','execution_receipt.json']}
    argv=[str(root/'preparation/remote_venv/bin/python'),str(root/'preparation/gpu_jail_launcher.py'),
      '--jail',str(jail),'--root',str(root),'--uid',str(uid),'--mode','run','--seed',str(a.seed),
      '--checkpoint-fd',str(outputs['model.pt']),'--prediction-fd',str(outputs['prediction.npy']),
      '--receipt-fd',str(outputs['execution_receipt.json'])]
    if a.reload_checkpoint:argv+=['--reload']
    start=time.monotonic();started_epoch=time.time();peak_rss=0;timed_out=False;memory_exceeded=False
    try:
        with (a.out/'stdout.original.log').open('xb') as stdout,(a.out/'stderr.original.log').open('xb') as stderr:
            proc=subprocess.Popen(argv,stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr,
              close_fds=True,pass_fds=tuple(outputs.values()),start_new_session=True,
              env={'PATH':'/usr/bin:/bin','LANG':'C.UTF-8'})
            def stop_owned_child(signum,frame):
                if proc.poll() is None:os.killpg(proc.pid,signal.SIGKILL)
                raise SystemExit(128+signum)
            signal.signal(signal.SIGTERM,stop_owned_child)
            signal.signal(signal.SIGHUP,stop_owned_child)
            for fd in outputs.values():os.close(fd)
            outputs.clear()
            while proc.poll() is None:
                duration=time.monotonic()-start
                if duration>(120 if a.reload_checkpoint else 900):timed_out=True;os.killpg(proc.pid,signal.SIGKILL);break
                try:
                    status=(Path('/proc')/str(proc.pid)/'status').read_text()
                    for line in status.splitlines():
                        if line.startswith('VmHWM:'):peak_rss=max(peak_rss,int(line.split()[1])*1024)
                    if peak_rss>32*1024**3:memory_exceeded=True;os.killpg(proc.pid,signal.SIGKILL);break
                    own_proc_snapshot(jail,proc.pid)
                except (FileNotFoundError,ProcessLookupError):pass
                time.sleep(.05)
            code=proc.wait(timeout=10)
        # This supervisor has only its one numerical child; kernel usage is not
        # taken from candidate metadata and catches peaks between polling samples.
        peak_rss=max(peak_rss,int(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss)*1024)
        memory_exceeded|=peak_rss>32*1024**3
        report={'stage':a.stage,'split':a.split,'seed':a.seed,'run_id':identifier,'exit_code':code,
          'status':'TIMEOUT' if timed_out else 'RESOURCE_LIMIT' if memory_exceeded else 'COMPLETE' if code==0 else 'CANDIDATE_ERROR',
          'started_epoch':started_epoch,'completed_epoch':time.time(),'elapsed_seconds':time.monotonic()-start,
          'actual_child_pid':proc.pid,'uid':uid,'actual_peak_rss_bytes':peak_rss,
          'method_sha256':hashlib.sha256(a.method.read_bytes()).hexdigest(),'entry_sha256':actual,
          'candidate_imported_on_host':False,'target_labels_in_candidate':False,'formal_model_research_credit':0,
          'artifact_sha256':{name:hashlib.sha256((a.out/name).read_bytes()).hexdigest() for name in ['model.pt','prediction.npy','execution_receipt.json']}}
        if report['status']=='COMPLETE':
            import numpy as np
            from independent_metrics import expression_mean_metrics
            import scanpy as sc
            prediction_file=a.out/'prediction.npy'
            with prediction_file.open('rb') as f:
                version=np.lib.format.read_magic(f)
                if version==(1,0):shape,order,dtype=np.lib.format.read_array_header_1_0(f,max_header_size=4096)
                elif version==(2,0):shape,order,dtype=np.lib.format.read_array_header_2_0(f,max_header_size=4096)
                else:raise ValueError('Unexpected prediction array format')
                expected_cells=811 if a.split=='dev' else 855
                assert shape==(expected_cells,2000) and not order and dtype==np.dtype('float64')
                assert prediction_file.stat().st_size==f.tell()+expected_cells*2000*8
            pred=np.load(prediction_file,allow_pickle=False,max_header_size=4096)
            if not np.isfinite(pred).all() or pred.min()<0:raise ValueError('Hard gate: prediction must be finite and nonnegative')
            target=task/'environment/public_assets/dev_targets.h5ad' if a.split=='dev' else root/'private/hidden_calibration_assets/targets.h5ad'
            t=sc.read_h5ad(target);truth=t.X.toarray() if hasattr(t.X,'toarray') else np.asarray(t.X)
            report['metrics']=expression_mean_metrics(truth,pred)
            report['checkpoint_bytes']=(a.out/'model.pt').stat().st_size
            assert report['checkpoint_bytes']<=256*1024**2
        (a.out/'supervisor_receipt.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
        raise SystemExit(0 if report['status']=='COMPLETE' else 1)
    finally:
        for fd in outputs.values():os.close(fd)
        uid_handle.close()
        shutil.rmtree(jail)

if __name__=='__main__':main()
