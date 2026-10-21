"""Export disjoint forecast origins from the trained shared backbone, expert-side."""
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path('/opt/author_work/rivermind-data/stssdl_auto2912_20261006')
SOURCE = ROOT/'source/ST-SSDL'
PRIVATE = ROOT/'private'
os.environ['STSSDL_FIXED_SEED'] = '17'
EXPORT_DEVICE=os.environ.get('STSSDL_EXPORT_DEVICE','gpu')
if EXPORT_DEVICE not in ['cpu','gpu']:raise ValueError('Export device must be cpu or gpu')
if EXPORT_DEVICE=='cpu':os.environ['CUDA_VISIBLE_DEVICES']='-1'
os.environ['PYTHONPATH'] = str(ROOT/'runtime-deps')
sys.path.insert(0, str(ROOT/'runtime-deps'))
sys.path.insert(0, str(SOURCE/'model_STSSDL'))
os.chdir(SOURCE/'model_STSSDL')
sys.argv = ['train_prepared.py', '--dataset', 'PEMS08', '--gpu', '0', '--epochs', '0', '--seed', '17']

import numpy as np
import torch
import train_prepared as official
from remote_prepare import lease, atomic

manifest = json.loads((PRIVATE/'preparation-complete.json').read_text())
checkpoint = Path(manifest['checkpoint'])
if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != manifest['sha256']:
    raise RuntimeError('Shared backbone checkpoint hash changed')
output = PRIVATE/'forecast-export'
if (output/'manifest.json').exists():
    prior=json.loads((output/'manifest.json').read_text())
    if prior['backbone']['sha256']!=manifest['sha256']:
        raise RuntimeError('Existing exported backbone differs; preserve evidence')
    for name,row in prior['groups'].items():
        for suffix,key in [('features','features_sha256'),('targets','target_sha256')]:
            if hashlib.sha256((output/f'{name}_{suffix}.npz').read_bytes()).hexdigest()!=row[key]:
                raise RuntimeError('Existing forecast export is corrupted')
    atomic('status.json',dict(phase='BACKBONE_READY_AWAITING_TASK_CALIBRATION',forecast_export_complete=True))
    print('Existing verified export preserved; no duplicate inference')
    sys.exit(0)
if output.exists():
    output.rename(output.with_name('forecast-export-incomplete-preserved-'+str(time.time_ns())))
output.mkdir()
handles = [] if EXPORT_DEVICE=='cpu' else lease()
try:
    if EXPORT_DEVICE=='cpu':torch.set_num_threads(2)
    atomic('status.json',dict(phase='EXPORTING_FROZEN_FORECASTS',export_device=str(official.device)))
    model = official.get_model()
    model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True))
    model.eval()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    records = {}
    for category in ['val', 'dev', 'test']:
        raw = np.load(SOURCE/'PEMS08'/f'{category}his.npz', allow_pickle=False)
        # One non-overlapping horizon per origin. Purge boundary-adjacent windows.
        indices = np.arange(12, len(raw['x']), 12)
        if category == 'val':
            midpoint = len(indices)//2
            groups = {'train':indices[:midpoint], 'pilot':indices[midpoint:]}
        else:
            groups = {'dev' if category == 'dev' else 'hidden':indices}
        for group, selected in groups.items():
            # Exact upstream input preprocessing, used by official training and
            # validation. Empty train-only weekday anchors yield NaN; upstream
            # converts these to zero before scaling, rather than dropping rows.
            missing_x=int(np.isnan(raw['x'][selected]).sum())
            missing_y=int(np.isnan(raw['y'][selected]).sum())
            xs = np.nan_to_num(raw['x'][selected]).astype(np.float32)
            ys = np.nan_to_num(raw['y'][selected]).astype(np.float32)
            forecasts = []
            for first in range(0, len(xs), 16):
                x = xs[first:first+16].copy()
                y = ys[first:first+16]
                x[...,0] = official.scaler.transform(x[...,0])
                x[...,2] = official.scaler.transform(x[...,2])
                tensors = [torch.from_numpy(z).to(official.device) for z in [x[...,0:1],x[...,1:2],x[...,2:3],y[...,1:2]]]
                with torch.inference_mode():
                    forecast = model(*tensors)[0]
                    forecasts.append(official.scaler.inverse_transform(forecast).cpu().numpy()[...,0])
            base = np.concatenate(forecasts).astype(np.float32)
            features = dict(base=base, history=xs[...,0], anchor=ys[...,2], clock=ys[...,1])
            for value in features.values():
                if not np.isfinite(value).all():
                    raise RuntimeError('Nonfinite frozen forecast feature')
            np.savez_compressed(output/f'{group}_features.npz', **features)
            np.savez_compressed(output/f'{group}_targets.npz', target=ys[...,0])
            records[group] = dict(origins=len(selected), shape=list(base.shape), source_category=category,
                    origin_stride=12, leading_origins_purged=12,
                    upstream_preprocessing='numpy.nan_to_num before float32/scaling; same as official train/validation loader',
                    missing_values_converted={'x':missing_x,'y':missing_y},
                    features_sha256=hashlib.sha256((output/f'{group}_features.npz').read_bytes()).hexdigest(),
                    target_sha256=hashlib.sha256((output/f'{group}_targets.npz').read_bytes()).hexdigest())
    np.savez_compressed(output/'graph.npz', adjacency=official.load_adj(str(SOURCE/'PEMS08/adj_PEMS08_distance.pkl'),'symadj')[0].astype(np.float32))
    information = dict(task_id='auto2912', backbone=manifest, groups=records,
            shared_pretrained_initialization=True, independent_neural_training_replicates=False,
            history_estimated_from_backbone_training_only=True,
            backbone_training_device='A100-PCIE-40GB',forecast_export_device=str(official.device),
            public_groups=['train','dev'], expert_only_groups=['pilot','hidden'],
            created_epoch=time.time(), formal_model_research_seconds={'gpt':0,'seed':0})
    (output/'manifest.json').write_text(json.dumps(information,indent=2)+'\n')
    atomic('forecast-export-complete.json', information)
finally:
    for handle in handles:
        handle.close()
    atomic('status.json',dict(phase='BACKBONE_READY_AWAITING_TASK_CALIBRATION',
                             forecast_export_complete=(output/'manifest.json').exists()))
