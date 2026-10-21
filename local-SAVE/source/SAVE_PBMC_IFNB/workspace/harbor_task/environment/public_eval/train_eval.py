"""Candidate execution: train and control inputs only, never target labels.

Must run in the platform candidate sandbox. Expert B/R executes audited methods.
This script alone does not establish an isolation boundary.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import time
import numpy as np
import scanpy as sc
import torch
from sklearn.preprocessing import MaxAbsScaler

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def load_method(path):
    spec = importlib.util.spec_from_file_location('candidate_method', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for name in ['build_model', 'train_model', 'predict']:
        if not callable(getattr(module, name, None)):
            raise ValueError('Missing method callable: ' + name)
    return module

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--method',type=Path,required=True)
    p.add_argument('--train',type=Path,required=True)
    p.add_argument('--controls',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--seed',type=int,required=True)
    p.add_argument('--reload',action='store_true')
    a=p.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required; CPU fallback is not a formal run')
    if a.out.exists() and not a.reload:
        raise RuntimeError('Fresh independent output directory required')
    a.out.mkdir(parents=True,exist_ok=a.reload)
    start=time.perf_counter()
    torch.set_num_threads(4)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    train=sc.read_h5ad(a.train)
    controls=sc.read_h5ad(a.controls)
    if list(train.var_names)!=list(controls.var_names):
        raise ValueError('Gene order mismatch')
    scaler=MaxAbsScaler()
    train.X=scaler.fit_transform(train.X)
    controls.X=scaler.transform(controls.X)
    method=load_method(a.method)
    model=method.build_model(train,torch.device('cuda:0'),a.seed)
    checkpoint=a.out/'model/model.pt'
    if a.reload:
        prior=json.loads((a.out/'execution_receipt.json').read_text())
        if sha(a.method)!=prior['method_sha256'] or sha(checkpoint)!=prior['checkpoint_sha256']:
            raise RuntimeError('Method or checkpoint changed before reload')
        model.model.load_state_dict(torch.load(checkpoint,map_location='cuda:0',weights_only=True),strict=True)
        model.is_trained=True
        training_seconds=0.0
    else:
        training_start=time.perf_counter()
        method.train_model(model,a.seed)
        training_seconds=time.perf_counter()-training_start
        checkpoint.parent.mkdir()
        torch.save(model.model.state_dict(),checkpoint)
    prediction=np.asarray(method.predict(model,controls,scaler,a.seed+10000),dtype=np.float64)
    if prediction.shape!=controls.shape or not np.isfinite(prediction).all() or prediction.min()<0:
        raise ValueError('Hard gate: expected finite nonnegative control-cells x genes prediction')
    if checkpoint.stat().st_size>256*1024**2:
        raise ValueError('Hard gate: checkpoint exceeds 256 MiB')
    name='reload_prediction.npy' if a.reload else 'prediction.npy'
    np.save(a.out/name,prediction,allow_pickle=False)
    receipt={'mode':'reload' if a.reload else 'fresh_training','seed':a.seed,
             'inference_seed':a.seed+10000,'method_sha256':sha(a.method),
             'train_sha256':sha(a.train),'controls_sha256':sha(a.controls),
             'checkpoint_sha256':sha(checkpoint),'checkpoint_bytes':checkpoint.stat().st_size,
             'prediction_sha256':sha(a.out/name),'train_shape':list(train.shape),
             'control_shape':list(controls.shape),'training_seconds':training_seconds,
             'elapsed_seconds':time.perf_counter()-start,'pid':os.getpid(),
             'gpu':torch.cuda.get_device_name(0),
             'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(0),
             'torch':torch.__version__,'cuda':torch.version.cuda}
    dest=a.out/('reload_receipt.json' if a.reload else 'execution_receipt.json')
    dest.write_text(json.dumps(receipt,indent=2))
    print(json.dumps(receipt,indent=2),flush=True)

if __name__=='__main__':
    main()
