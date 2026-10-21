"""Expert-only A100 preparation. No model trajectory time is credited here."""
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import fcntl

ROOT = Path('/opt/author_work/rivermind-data/stssdl_auto2912_20261006')
PRIVATE = ROOT / 'private'
SOURCE = ROOT / 'source/ST-SSDL'
PYTHON = '/opt/author_work/heima_20261005_runtime/venv/bin/python'


def atomic(name, value):
    path = PRIVATE / name
    temporary = path.with_suffix('.pending.json')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def command(argv, cwd, tag, timeout):
    started = time.time()
    with (PRIVATE / (tag + '.run.log')).open('x') as log:
        result = subprocess.run(argv, cwd=cwd, env=os.environ.copy(), stdout=log,
                                stderr=subprocess.STDOUT, timeout=timeout)
    receipt = dict(command=argv, cwd=str(cwd), started_epoch=started,
                   completed_epoch=time.time(), exit_code=result.returncode,
                   formal_trajectory=False, credited_research_seconds=0)
    atomic(tag + '.execution.json', receipt)
    if result.returncode:
        raise RuntimeError('Observed command failure; inspect ' + tag + '.run.log')


def lease():
    paths = [Path('/opt/author_work/dd_repair_v5_runs/gpu.lock'),
             Path('/opt/author_work/rivermind-data/dd_repair_v5_runs/gpu.lock'),
             Path('/dev/shm/dd_primary_antithetic_late256_20261005/gpu.lock')]
    paths += list(Path('/tmp').glob('*gpu*lock'))
    unique = {}
    for path in paths:
        if path.is_file():
            stat = path.stat()
            unique[(stat.st_dev, stat.st_ino)] = path
    if not unique:
        raise RuntimeError('Expected shared GPU mutex not available')
    started = time.time()
    atomic('status.json', dict(phase='WAITING_FOR_GPU_EXCLUDED', queued_epoch=started,
                              locks=[str(p) for p in sorted(unique.values())]))
    while time.time() - started < 172800:
        held = []
        try:
            for path in sorted(unique.values()):
                handle = path.open('a+')
                held.append(handle)
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            environment = os.environ.copy()
            environment.pop('LD_PRELOAD', None)
            memory = subprocess.check_output(['/usr/bin/nvidia-smi', '--query-gpu=memory.used',
                        '--format=csv,noheader,nounits'], env=environment, text=True)
            if int(memory.strip().splitlines()[0]) < 256:
                atomic('gpu-lease.json', dict(queued_seconds=time.time()-started,
                        gpu_started_epoch=time.time(), actual_memory_mib=int(memory.strip()),
                        locks=[str(p) for p in sorted(unique.values())]))
                return held
        except BlockingIOError:
            pass
        for handle in held:
            handle.close()
        time.sleep(10)
    raise TimeoutError('Shared A100 unavailable; queue time never counts as research')


def main():
    PRIVATE.mkdir(parents=True, exist_ok=True)
    if (PRIVATE / 'preparation-complete.json').exists():
        raise RuntimeError('Existing preparation evidence must be preserved')
    os.environ['PYTHONPATH'] = str(ROOT / 'runtime-deps')
    os.environ['PYTHONUNBUFFERED'] = '1'
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    # All patches remain expert-side; preserve pristine source and a literal diff.
    original = SOURCE / 'generate_training_data_his_PEMS.py'
    code = original.read_text()
    old = "df_train = data[(data.timestamp >= start_dates[dataset_name]) & (data.timestamp <= train_dates[dataset_name])]"
    assert code.count(old) == 1
    replacement = "training_end_exclusive = 11 + round((len(df_new) - 23) * 0.6) + 12\n    training_end_time = df_new.index[training_end_exclusive - 1]\n    df_train = data[data.timestamp <= training_end_time]"
    code = code.replace(old, replacement)
    # Training 60%, calibration/early-stop 10%, public development 10%, hidden 20%.
    code = code.replace("num_val = num_samples - num_test - num_train", "num_val = round(num_samples * 0.1)\n    num_dev = num_samples - num_test - num_train - num_val")
    code = code.replace("x_test, y_test = x[-num_test:], y[-num_test:]", "x_test, y_test = x[-num_test:], y[-num_test:]\n    x_dev, y_dev = x[num_train+num_val:-num_test], y[num_train+num_val:-num_test]")
    code = code.replace('for cat in ["train", "val", "test"]:', 'for cat in ["train", "val", "dev", "test"]:')
    (SOURCE / 'prepare_train_only.py').write_text(code)
    command([PYTHON, 'prepare_train_only.py', '--dataset', 'PEMS08'], SOURCE,
            'train-only-preprocessing', 1800)
    training = (SOURCE / 'model_STSSDL/train_STSSDL.py').read_text()
    # Official dataset presets keep all original hyperparameters, except restore requested seed.
    marker = "model_name = 'STSSDL'"
    assert training.count(marker) == 1
    training = training.replace(marker, "args.seed = int(os.environ['STSSDL_FIXED_SEED'])\nrandom.seed(args.seed)\n" + marker)
    # Neither public development nor hidden is consulted for checkpoint selection.
    training = training.replace("        test_loss, _, _ = evaluate(model, 'test')", "        # Held-out evaluation deliberately omitted during training.")
    training = training.replace("    test_loss, _, _ = evaluate(model, 'test')", "    test_loss, _, _ = evaluate(model, 'val')")
    (SOURCE / 'model_STSSDL/train_prepared.py').write_text(training)
    import difflib
    diff = ''.join(difflib.unified_diff((SOURCE/'model_STSSDL/train_STSSDL.py').read_text().splitlines(True), training.splitlines(True), fromfile='official/train_STSSDL.py', tofile='expert/train_prepared.py'))
    (PRIVATE / 'training-compatibility.patch').write_text(diff)
    preprocessing_diff = ''.join(difflib.unified_diff(original.read_text().splitlines(True), code.splitlines(True), fromfile='official/generate_training_data_his_PEMS.py', tofile='expert/prepare_train_only.py'))
    (PRIVATE / 'preprocessing-leakage-repair.patch').write_text(preprocessing_diff)
    handles = lease()
    try:
        os.environ['STSSDL_FIXED_SEED'] = '17'
        atomic('status.json', dict(phase='A100_OFFICIAL_MODEL_SMOKE', started_epoch=time.time(),
                formal_trajectory=False, credited_research_seconds=0))
        command([PYTHON, '-u', 'train_prepared.py', '--dataset', 'PEMS08', '--gpu', '0',
                 '--epochs', '3', '--seed', '17'], SOURCE/'model_STSSDL', 'official-model-smoke', 1800)
        atomic('status.json', dict(phase='A100_SHARED_BACKBONE_TRAINING', started_epoch=time.time(),
                seed=17, maximum_epochs=200, early_stop_patience=20,
                formal_trajectory=False, credited_research_seconds=0))
        command([PYTHON, '-u', 'train_prepared.py', '--dataset', 'PEMS08', '--gpu', '0',
                 '--epochs', '200', '--seed', '17'], SOURCE/'model_STSSDL', 'shared-backbone-training', 21600)
        checkpoints = sorted((SOURCE/'save').glob('PEMS08_STSSDL_*/*.pt'), key=lambda p:p.stat().st_mtime)
        checkpoint = checkpoints[-1]
        result = dict(status='SHARED_BACKBONE_TRAINED', checkpoint=str(checkpoint),
                sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(), size_bytes=checkpoint.stat().st_size,
                seed=17, official_model=True, changed_method=False,
                formal_model_trajectory_seconds={'gpt':0,'seed':0},
                next_gate='Export frozen forecasts, calibrate B/R/U, run local Harbor NOP, then independent model CLI trajectories')
        atomic('preparation-complete.json', result)
        atomic('status.json', dict(phase='BACKBONE_READY_AWAITING_TASK_CALIBRATION', **result))
    finally:
        for handle in handles:
            handle.close()


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        atomic('status.json', dict(phase='PREPARATION_FAILED', error_type=type(error).__name__,
                message=str(error), observed_epoch=time.time(), formal_trajectory=False,
                credited_research_seconds=0))
        raise
