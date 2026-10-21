"""Frozen child: receive arrays, run one method, return only numerical output."""
import argparse,importlib.util,json,pathlib,sys
import numpy as np

p=argparse.ArgumentParser()
p.add_argument('--method',required=True);p.add_argument('--input',required=True)
p.add_argument('--output',required=True);p.add_argument('--ready',required=True)
a=p.parse_args()
data=np.load(a.input,allow_pickle=False)
spec=importlib.util.spec_from_file_location('candidate_method',a.method)
mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod)
if not callable(getattr(mod,'value',None)):raise ValueError('Missing callable value')
print(a.ready,flush=True)
if sys.stdin.readline().strip()!='GO':raise ValueError('Missing execution signal')
result=mod.value(data['x'],data['y'],data['r'],data['ry'],int(data['seed']))
result=np.asarray(result,dtype=np.float64)
if result.shape!=(len(data['x']),) or not np.isfinite(result).all():
 raise ValueError('Expected one finite value per candidate point')
np.save(a.output,result,allow_pickle=False)
print('DONE_'+a.ready,flush=True)
print(json.dumps({'output_count':len(result)}),flush=True)
