"""Finish the existing round3 from its last complete saved training snapshot.

Original exit code, damaged temporary and all consumed training time remain
in evidence. This continuation performs no teacher generation or training.
"""
from datetime import datetime
import fcntl
import json
import math
import os
from pathlib import Path
import re
import shutil
import time

from candidate_runner import (OUT,PY,REPO,ROOT,WORK,event,gpu,integrity_checks,now,
                              run_stage,save,sha,state)


def main():
    directory=OUT/'rounds/round03'
    large=WORK/'round03'
    lock=(OUT/'.candidate_runner.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    baseline=(OUT/'.baseline_runner.lock').open('a')
    fcntl.flock(baseline,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (directory/'evaluation_execution.json').exists() or (directory/'result.json').exists():
        raise RuntimeError('Existing evaluation must be audited before any continuation')
    recovery=json.loads((directory/'student_training_recovery.json').read_text())
    original=json.loads((directory/'student_training_execution.json').read_text())
    if recovery['original_execution_sha256']!=sha(directory/'student_training_execution.json'):
        raise RuntimeError('Original training execution changed')
    if original['exit_code']!=1 or original['termination'] is not None or Path(f"/proc/{original['pid']}").exists():
        raise RuntimeError('Original training must have actually ended with its documented storage fault')
    if not recovery['weights_and_ema_all_finite'] or recovery['checkpoint_update']!=recovery['optimizer_update']:
        raise RuntimeError('Incomplete recovered snapshot')
    checkpoint=large/'training/checkpoint.pth'
    if sha(checkpoint)!=recovery['checkpoint_sha256'] or sha(directory/'method.py')!=recovery['method_sha256']:
        raise RuntimeError('Verified recovered artifact changed')
    budget=json.loads((directory/'budget.json').read_text())
    save(directory/'budget_before_recovery.json',budget)
    ended=datetime.fromisoformat(original['finished_at']).timestamp()
    budget.update(end_unix=ended,finished_at=original['finished_at'],
        total_gpu_wall_seconds=ended-budget['start_unix'],eligible_budget=ended-budget['start_unix']<=14400,
        recovered_after_storage_failure=True,recovery_bookkeeping_at=now(),
        actual_attempted_updates=recovery['last_attempted_update'],actual_scored_checkpoint_updates=recovery['checkpoint_update'],
        budget_clock_restarted=False,new_teacher_generation_seconds=0,new_training_seconds=0)
    if not budget['eligible_budget']:
        raise RuntimeError('Original generation and training exceeded the4h candidate budget')
    save(directory/'budget.json',budget)
    save(directory/'recovery_runner.json',{'pid':os.getpid(),'started_at':now(),
        'original_runner_pid':90785,'scope':'Fixed formal scoring of verified latest complete snapshot only',
        'training_failure_and_partial_retained':True,'recovery_evidence_sha256':sha(directory/'student_training_recovery.json')})
    shutil.copyfile(ROOT/'recover_round03.py',directory/'recover_round03.py')
    save(directory/'protected_at_recovery.json',integrity_checks())
    frozen=json.loads((OUT/'protocol_frozen.json').read_text())
    if sha(REPO/'sample.py')!=frozen['official_sample_sha256'] or sha(REPO/'evaluation/fid_score.py')!=frozen['scorer_sha256']:
        raise RuntimeError('Frozen sampling or FID source changed')
    event({'kind':'candidate_artifact_recovery','round':3,'runner_pid':os.getpid(),
        'checkpoint_update':recovery['checkpoint_update'],'original_exit_code':1,
        'actual_attempted_updates':recovery['last_attempted_update'],'training_gpu_seconds_fully_charged':budget['total_gpu_wall_seconds'],
        'original_failed_temporary_retained':True,'new_teacher_or_training_process':False})
    while float(gpu().split(',')[2])<19500:
        with (directory/'recovery_gpu_queue.jsonl').open('a') as output:
            output.write(json.dumps({'at':now(),'gpu':gpu(),'counts_as_effective':False})+'\n')
        time.sleep(30)
    env={k:v for k,v in os.environ.items() if not any(x in k.upper() for x in ('API_KEY','TOKEN','SECRET','PASSWORD'))}
    env.update(PYTHONDONTWRITEBYTECODE='1',PYTHONHASHSEED='0',PYTHONFAULTHANDLER='1',
        HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',
        TMPDIR=str(WORK/'runtime/tmp'),CUDA_CACHE_PATH=str(WORK/'runtime/cuda_cache'),
        PYTORCH_KERNEL_CACHE_PATH=str(WORK/'runtime/torch_kernels'),PYTORCH_CUDA_ALLOC_CONF='max_split_size_mb:128',
        DD_CUDA_MEMORY_FRACTION='.30')
    sample_root=large/'evaluation'
    if sample_root.exists():
        raise RuntimeError('Unexpected existing sample corpus')
    command=[PY,'-B','-u',str(ROOT/'official_entry.py'),'--ckpt_path',str(checkpoint),
        '--vae_ckpt','historical-resources/models/dd1/VAR/vae_ch160v4096z32.pth','--sample_dir',str(sample_root),
        '--sample_num','50000','--t_list','0','--loss_split_t','5','--depth','16','--bs','8','--seed','0',
        '--fid_statistic','historical-resources/models/dd1/fid_stats','--local_out_dir_path',str(large/'official_runtime')]
    run_stage(command,'evaluation',directory,env,3)
    save(directory/'protected_after.json',integrity_checks())
    if sha(REPO/'sample.py')!=frozen['official_sample_sha256'] or sha(REPO/'evaluation/fid_score.py')!=frozen['scorer_sha256']:
        raise RuntimeError('Frozen sampling or FID source changed during evaluation')
    raw=(sample_root/'fid.txt').read_text()
    match=re.search(r'FID:\s*([-+0-9.eE]+)',raw)
    fid=float(match[1]) if match else float('nan')
    if not math.isfinite(fid) or fid<0 or fid==114514:
        raise RuntimeError('Invalid official score')
    paths=sorted((sample_root/'images').glob('*.png'),key=lambda path:int(path.stem))
    if len(paths)!=50000 or [int(path.stem) for path in paths]!=list(range(50000)):
        raise RuntimeError('Formal sample count mismatch')
    from PIL import Image,ImageStat
    means=set()
    with (directory/'sample_manifest.jsonl').open('w') as manifest:
        for index,path in enumerate(paths):
            with Image.open(path) as image:
                if image.size!=(256,256):
                    raise RuntimeError('Formal dimensions differ')
                if index<256:
                    means.add(tuple(round(v,4) for v in ImageStat.Stat(image).mean))
            manifest.write(json.dumps({'file':path.name,'bytes':path.stat().st_size,'sha256':sha(path)})+'\n')
    if len(means)<2:
        raise RuntimeError('Collapsed identical output corpus')
    grid=Image.new('RGB',(2048,2048))
    for index,path in enumerate(paths[:64]):
        with Image.open(path) as image:
            grid.paste(image.convert('RGB'),((index%8)*256,(index//8)*256))
    grid.save(directory/'sample_grid.jpg',quality=90)
    shutil.copyfile(sample_root/'fid.txt',directory/'fid.txt')
    result={'protocol_version':'dd-selfproto-v1.0','sample_count':50000,'seed_set':[0],'batch_size':8,
        'fid_50k':fid,'score':fid,'checkpoint_path':str(checkpoint),'checkpoint_sha256':sha(checkpoint),
        'method_sha256':sha(directory/'method.py'),'sample_manifest_sha256':sha(directory/'sample_manifest.jsonl'),
        'training_budget_seconds':budget['total_gpu_wall_seconds'],'visual_review_required':True,'quality_gate_passed':False,
        'recovered_training_snapshot':True,'checkpoint_updates':6100,'attempted_updates_billed':6200,
        'original_training_exit_code':1,'recovery_evidence_sha256':sha(directory/'student_training_recovery.json')}
    save(directory/'result.json',result)
    published=directory/'published'
    published.mkdir(exist_ok=True)
    save(published/'result.json',result)
    shutil.copyfile(directory/'fid.txt',published/'fid.txt')
    state('candidate_finished_pending_review',active_experiment=None,pending_round=3,pending_result=str(directory/'result.json'))
    event({'kind':'candidate_finished_pending_visual_review','round':3,'result':str(directory/'result.json'),'recovered_snapshot':True})
    print(f'FID: {fid}',flush=True)


if __name__=='__main__':
    main()
