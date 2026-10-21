"""Public/local Docker client; evaluator authority comes from its queue mount."""
import argparse,hashlib,json,os,stat
from pathlib import Path
from queue_client import request_json

def main():
    p=argparse.ArgumentParser();p.add_argument('--split',choices=['dev','hidden'],default='dev')
    p.add_argument('--method',type=Path,default=Path('/workspace/solution/method.py'))
    p.add_argument('--output',type=Path,default=Path('/workspace/solution/dev_result.json'))
    p.add_argument('--reward-dir',type=Path);a=p.parse_args()
    path=a.method.resolve();path.relative_to(Path('/workspace/solution'))
    if a.method.is_symlink():raise ValueError('Submit a regular method file')
    fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW)
    with os.fdopen(fd,'rb') as f:
        s=os.fstat(f.fileno());assert stat.S_ISREG(s.st_mode) and s.st_size<=1024**2
        raw=f.read(1024**2+1)
    if a.reward_dir:
        a.reward_dir.mkdir(parents=True,exist_ok=True)
        for name in ['reward.txt','reward.json']:(a.reward_dir/name).unlink(missing_ok=True)
    # split is documentary here. Only the trusted physical Verifier queue can
    # select hidden; a public lane cannot promote itself using arguments.
    result=request_json('evaluate',{'method_source':raw.decode('utf-8'),
      'method_sha256':hashlib.sha256(raw).hexdigest()},timeout=86400)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    stage=a.output.with_suffix('.pending.json');stage.write_text(json.dumps(result,indent=2));stage.replace(a.output)
    if (result.get('status')=='COMPLETE' and result.get('all_seeds_complete')) or (result.get('status')=='HARD_GATE' and result.get('score')==-1):
        if a.reward_dir:
            for name,value in [('reward.txt',str(result['score'])+'\n'),('reward.json',json.dumps({'reward':result['score']}))]:
                temp=a.reward_dir/(name+'.pending');temp.write_text(value);temp.replace(a.reward_dir/name)
    else:raise RuntimeError('Independent evaluation did not complete; no reward emitted')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
