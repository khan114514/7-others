"""Persistent single candidate execution, fixed protocol and external evidence.

No strategy lives here. The method snapshot is the only candidate-specific code.
This control process has no model/network traffic and runs each method stage
under the filesystem and socket guard installed by that stage's entry point.
"""
import argparse
from dataclasses import asdict
from datetime import datetime
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time
from zoneinfo import ZoneInfo

ROOT = Path('historical-codex-workspace')
OUT = ROOT/'agent-output'
WORK = Path('historical-codex-models')
FRAMEWORK = WORK/'framework'
REPO = Path('historical-resources/autoresearch_tasks/repos/distilled_decoding/VAR')
PY = 'historical-resources/venvs/dd1/bin/python'
TZ = ZoneInfo('Asia/Shanghai')


def now():
    return datetime.now(TZ).isoformat()


def save(path, value):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    temporary.replace(path)


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda:source.read(8<<20),b''):
            digest.update(block)
    return digest.hexdigest()


def event(value):
    with (OUT/'run_events.jsonl').open('a') as f:
        f.write(json.dumps({'recorded_at':now(),**value},ensure_ascii=False)+'\n')


def state(status, **kwargs):
    previous=json.loads((OUT/'resume_state.json').read_text())
    previous.update(status=status,recorded_at=now(),**kwargs)
    save(OUT/'resume_state.json',previous)


def gpu():
    return subprocess.run(['nvidia-smi','--query-gpu=timestamp,memory.used,memory.free,utilization.gpu',
                           '--format=csv,noheader,nounits'],capture_output=True,text=True,check=True).stdout.strip()


def integrity_checks():
    result=[]
    manifest=ROOT/'harbor_task/environment/PROTECTED_SHA256.txt'
    for line in manifest.read_text().splitlines():
        if not line.strip() or line.lstrip().startswith('#'):
            continue
        expected,relative,*_=line.split()
        path=Path('historical-resources')/relative
        actual=sha(path)
        result.append({'path':str(path),'expected':expected,'actual':actual,'matches':expected==actual})
    if not all(r['matches'] for r in result):
        raise RuntimeError('Protected asset integrity failure')
    return result


def run_stage(command, stage, directory, environment, round_number, deadline=None):
    metadata={'stage':stage,'command':command,'cwd':str(REPO),'started_at':now(),
              'finished_at':None,'exit_code':None,'deadline_unix':deadline}
    file=directory/(stage+'_execution.json')
    save(file,metadata)
    start=time.monotonic()
    with (directory/'run.log').open('a') as logfile:
        logfile.write('\n'+json.dumps({'stage':stage,'started_at':metadata['started_at'],'command':command})+'\n')
        logfile.flush()
        child=subprocess.Popen(command,cwd=REPO,env=environment,stdin=subprocess.DEVNULL,
                               stdout=logfile,stderr=subprocess.STDOUT,start_new_session=True)
        metadata['pid']=child.pid
        save(file,metadata)
        state('candidate_'+stage,active_experiment={'round':round_number,'runner_pid':os.getpid(),
              'child_pid':child.pid,'stage':stage,'evidence':str(directory)})
        event({'kind':'effective_process_segment_start','round':round_number,'stage':stage,
               'pid':child.pid,'execution':str(file)})
        termination=None
        peak=None
        with (directory/(stage+'_gpu.jsonl')).open('a') as monitor:
            while child.poll() is None:
                memory=subprocess.run(['nvidia-smi','--query-compute-apps=pid,used_memory',
                    '--format=csv,noheader,nounits'],capture_output=True,text=True)
                own=[]
                for line in memory.stdout.splitlines():
                    parts=line.split(',')
                    if parts[0].strip()==str(child.pid):
                        own.append(line)
                        try:
                            measured=float(parts[1])
                            if measured>0:
                                peak=max(peak or 0,measured)
                        except ValueError:
                            pass
                monitor.write(json.dumps({'at':now(),'pid':child.pid,'own_gpu':own,
                    'global_gpu':gpu(),'observed_process_peak_mib':peak,
                    'zero_process_memory_means':'unavailable measurement, not zero usage'})+'\n')
                monitor.flush()
                if peak is not None and peak>19000:
                    termination='memory_limit'
                if deadline is not None and time.time()>=deadline:
                    termination='budget_limit'
                if termination:
                    os.killpg(child.pid,signal.SIGTERM)
                    try:
                        child.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid,signal.SIGKILL)
                    break
                time.sleep(10)
        code=child.wait()
    metadata.update(finished_at=now(),exit_code=code,elapsed_seconds=time.monotonic()-start,
                    termination=termination,peak_observed_gpu_mib=peak)
    save(file,metadata)
    event({'kind':'effective_process_segment_end','round':round_number,'stage':stage,
           **{k:metadata[k] for k in ('pid','started_at','finished_at','exit_code','elapsed_seconds','termination')},
           'eligibility':'requires result and infrastructure review'})
    if code or termination:
        raise RuntimeError(f'{stage} failed: exit={code}, termination={termination}')
    return metadata


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--round',type=int,required=True)
    parser.add_argument('--method',required=True)
    parser.add_argument('--training-hours',type=float,default=3.8)
    parser.add_argument('--diagnostic-samples',type=int,default=0)
    args=parser.parse_args()
    if not 0<args.training_hours<=4.:
        raise ValueError('Frozen budget exceeded')
    lines=(OUT/'trajectory.jsonl').read_text().splitlines()
    rounds=[json.loads(line) for line in lines]
    if not rounds or rounds[0]['round']!=1 or rounds[0]['status']!='ok':
        raise RuntimeError('Phase one must be reviewed and closed before a candidate')
    if args.round!=len(rounds)+1:
        raise RuntimeError('Next candidate must preserve contiguous round numbers')
    lock=(OUT/'.candidate_runner.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    baseline_lock=(OUT/'.baseline_runner.lock').open('a')
    fcntl.flock(baseline_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    directory=OUT/'rounds'/f'round{args.round:02d}'
    directory.mkdir(parents=True,exist_ok=True)
    if (directory/'runner.json').exists():
        raise RuntimeError('Existing candidate execution requires recovery review')
    method=directory/'method.py'
    shutil.copyfile(args.method,method)
    large=WORK/f'round{args.round:02d}'
    large.mkdir(parents=True,exist_ok=True)
    save(directory/'runner.json',{'pid':os.getpid(),'started_at':now(),'round':args.round,
         'method_sha256':sha(method),'framework_sha256':{str(p.relative_to(FRAMEWORK)):sha(p)
         for p in FRAMEWORK.rglob('*.py')},'large_work_area':str(large),'training_hours':args.training_hours})
    save(directory/'protected_before.json',integrity_checks())
    for name in ('common.py','generate_pairs.py','train_student.py','rectified_categorical.py'):
        shutil.copyfile(FRAMEWORK/name,directory/name)
    shutil.copyfile(ROOT/'runtime_guard.py',directory/'runtime_guard.py')
    shutil.copyfile(ROOT/'official_entry.py',directory/'official_entry.py')
    shutil.copyfile(ROOT/'candidate_runner.py',directory/'candidate_runner.py')
    shutil.copyfile(REPO/'sample.py',directory/'official_sample.py')
    scorer_hash=sha(REPO/'evaluation/fid_score.py')
    event({'kind':'gpu_queue_begin','round':args.round,'counts_as_effective':False,'runner_pid':os.getpid()})
    while True:
        status=gpu()
        with (directory/'gpu_queue.jsonl').open('a') as f:
            f.write(json.dumps({'at':now(),'gpu':status,'counts_as_effective':False})+'\n')
        if float(status.split(',')[2])>=19500:
            break
        time.sleep(30)
    event({'kind':'gpu_queue_end','round':args.round,'counts_as_effective':False})
    environment={k:v for k,v in os.environ.items() if not any(x in k.upper() for x in ('API_KEY','TOKEN','SECRET','PASSWORD'))}
    environment.update(PYTHONDONTWRITEBYTECODE='1',PYTHONHASHSEED='0',PYTHONFAULTHANDLER='1',
        HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',
        TMPDIR=str(WORK/'runtime/tmp'),CUDA_CACHE_PATH=str(WORK/'runtime/cuda_cache'),
        PYTORCH_KERNEL_CACHE_PATH=str(WORK/'runtime/torch_kernels'),PYTORCH_CUDA_ALLOC_CONF='max_split_size_mb:128')
    # Include process startup, teacher generation, training, checkpoints and
    # stage transitions in the same <=4h clock; evaluation starts after it.
    begin=time.time()
    deadline=begin+args.training_hours*3600
    evidence={'start_unix':begin,'started_at':now(),'deadline_unix':deadline,'limit_gpu_hours':4.,
              'reserved_engineering_hours':4.-args.training_hours,'teacher_reuse_generation_charge':'Full actual generation cost included; no cross-candidate cache discounts.'}
    save(directory/'budget.json',evidence)
    try:
        run_stage([PY,'-B','-u',str(FRAMEWORK/'generate_pairs.py'),'--method',str(method),
            '--out',str(large/'pairs'),'--deadline',str(deadline)],'teacher_generation',directory,environment,args.round,deadline)
        run_stage([PY,'-B','-u',str(FRAMEWORK/'train_student.py'),'--method',str(method),
            '--pairs',str(large/'pairs'),'--out',str(large/'training'),'--deadline',str(deadline)],
            'student_training',directory,environment,args.round,deadline)
        ended=time.time()
        evidence.update(end_unix=ended,finished_at=now(),total_gpu_wall_seconds=ended-begin,
                        eligible_budget=ended-begin<=4.*3600)
        save(directory/'budget.json',evidence)
        if not evidence['eligible_budget']:
            raise RuntimeError('Actual generation+training exceeds the frozen4h budget')
        checkpoint=large/'training/checkpoint.pth'
        samples=args.diagnostic_samples or 50000
        if not 1<=samples<=50000:
            raise ValueError('Invalid sample count')
        sample_root=large/'evaluation'
        environment['DD_CUDA_MEMORY_FRACTION']='.30'
        command=[PY,'-B','-u',str(ROOT/'official_entry.py'),'--ckpt_path',str(checkpoint),
            '--vae_ckpt',str(VAE_PATH),'--sample_dir',str(sample_root),'--sample_num',str(samples),
            '--t_list','0','--loss_split_t','5','--depth','16','--bs','8','--seed','0',
            '--fid_statistic','historical-resources/models/dd1/fid_stats',
            '--local_out_dir_path',str(large/'official_runtime')]
        run_stage(command,'evaluation',directory,environment,args.round)
        save(directory/'protected_after.json',integrity_checks())
        if sha(REPO/'evaluation/fid_score.py')!=scorer_hash:
            raise RuntimeError('Scorer changed during candidate execution')
        text=(sample_root/'fid.txt').read_text()
        match=re.search(r'FID:\s*([-+0-9.eE]+)',text)
        fid=float(match[1]) if match else float('nan')
        if not math.isfinite(fid) or fid<0 or fid==114514:
            raise RuntimeError('Invalid scorer result')
        images=sample_root/'images'
        paths=sorted(images.glob('*.png'),key=lambda p:int(p.stem))
        if len(paths)!=samples or [int(p.stem) for p in paths]!=list(range(samples)):
            raise RuntimeError('Evaluation sample count mismatch')
        from PIL import Image,ImageStat
        means=set()
        with (directory/'sample_manifest.jsonl').open('w') as f:
            for index,path in enumerate(paths):
                with Image.open(path) as image:
                    if image.size!=(256,256):
                        raise RuntimeError('Invalid evaluation image size')
                    if index<256:
                        means.add(tuple(round(v,4) for v in ImageStat.Stat(image).mean))
                f.write(json.dumps({'file':path.name,'bytes':path.stat().st_size,'sha256':sha(path)})+'\n')
        if len(means)<2:
            raise RuntimeError('Collapsed evaluation images')
        grid=Image.new('RGB',(8*256,8*256))
        for index,path in enumerate(paths[:64]):
            with Image.open(path) as image:
                grid.paste(image.convert('RGB'),((index%8)*256,(index//8)*256))
        grid.save(directory/'sample_grid.jpg',quality=90)
        shutil.copyfile(sample_root/'fid.txt',directory/'fid.txt')
        result={'protocol_version':'dd-selfproto-v1.0','sample_count':samples,'seed_set':[0],
                'batch_size':8,'fid_diagnostic' if samples<50000 else 'fid_50k':fid,
                'score':fid if samples==50000 else None,'checkpoint_path':str(checkpoint),
                'checkpoint_sha256':sha(checkpoint),'method_sha256':sha(method),
                'sample_manifest_sha256':sha(directory/'sample_manifest.jsonl'),
                'training_budget_seconds':evidence['total_gpu_wall_seconds'],
                'visual_review_required':True,'quality_gate_passed':False}
        save(directory/'result.json',result)
        state('candidate_finished_pending_review',active_experiment=None,pending_round=args.round,
              pending_result=str(directory/'result.json'))
        event({'kind':'candidate_finished_pending_visual_review','round':args.round,'result':str(directory/'result.json')})
    except Exception as error:
        evidence.update(failed_at=now(),failure_reason=str(error),total_wall_seconds_so_far=time.time()-begin)
        save(directory/'failure.json',evidence)
        state('candidate_failed_requires_diagnosis',active_experiment=None,pending_round=args.round)
        event({'kind':'candidate_failure','round':args.round,'reason':str(error),'closed_round':False})
        raise


VAE_PATH='historical-resources/models/dd1/VAR/vae_ch160v4096z32.pth'

if __name__=='__main__':
    main()
