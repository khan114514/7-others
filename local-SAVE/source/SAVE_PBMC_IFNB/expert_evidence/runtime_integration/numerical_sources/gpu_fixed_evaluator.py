"""Five-seed independent scorer, controlled only by the trusted role relay."""
import argparse,hashlib,json,statistics,subprocess,sys,time,uuid
from pathlib import Path

def unique_pairs(values):
    result={}
    for key,value in values:
        if key in result:raise ValueError('Duplicate JSON key')
        result[key]=value
    return result

def preflight(root,split):
    seeds=json.loads((root/'submission/workspace/harbor_task/tests/protocol.json').read_text())['seeds']
    assert seeds==[17,29,43,71,97]
    task=root/'submission/workspace/harbor_task'
    frozen=json.loads((task/'tests/frozen_assets.json').read_text())
    for relative,digest in frozen.items():
        path=task/'tests'/relative[len('/tests/'):] if relative.startswith('/tests/') else task/'environment'/relative
        if not path.is_file() or path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest()!=digest:
            raise RuntimeError('Protected asset fingerprint mismatch: '+relative)
    contract=json.loads((root/'private/private_verifier_contract_v1.json').read_text())
    if split=='hidden':
        for name,binding in contract['assets'].items():
            path=root/'private/hidden_calibration_assets'/name
            if hashlib.sha256(path.read_bytes()).hexdigest()!=binding['sha256']:raise RuntimeError('Private asset fingerprint mismatch')
    base=root/'private/gpu-jail-runtime-v2'
    expected=json.loads((root/'private/gpu-jail-runtime-v2-active-entry.json').read_text())['entry_sha256']
    assert hashlib.sha256((base/'runner.py').read_bytes()).hexdigest()==expected
    for name in ['gpu_jail_supervisor.py','gpu_jail_launcher.py','independent_metrics.py']:
        assert (root/'preparation'/name).is_file()
    return seeds

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--split',choices=['dev','hidden'],required=True);a=p.parse_args();root=a.root.resolve()
    assert root.name=='save_autoresearch_20261009'
    raw=sys.stdin.buffer.read(2*1024**2+1)
    if len(raw)>2*1024**2:raise ValueError('Request size limit')
    request=json.loads(raw,object_pairs_hook=unique_pairs)
    if set(request)!={'method_source','method_sha256'} or not isinstance(request['method_source'],str):raise ValueError('Invalid source-only request')
    source=request['method_source'].encode()
    if len(source)>1024**2 or hashlib.sha256(source).hexdigest()!=request['method_sha256']:raise ValueError('Source fingerprint or size mismatch')
    identifier=uuid.uuid4().hex;out=root/'private/isolated-evaluations'/identifier;out.mkdir(parents=True,mode=0o700)
    method=out/'method.py';method.write_bytes(source);method.chmod(0o600)
    from freeze_gpu_binding import verify
    runtime_sha=verify(root)
    seeds=preflight(root,a.split)
    start=time.monotonic();rows=[];status='COMPLETE';failure=None;total_budget=3600 if a.split=='dev' else 5700
    for seed in seeds:
        assert verify(root)==runtime_sha
        current=out/f'seed_{seed}'
        command=[sys.executable,str(root/'preparation/gpu_jail_supervisor.py'),'--root',str(root),
          '--method',str(method),'--out',str(current),'--seed',str(seed),'--split',a.split,
          '--stage','FIXED_FIVE_SEED_INDEPENDENT_SCORING']
        with (out/f'seed_{seed}-controller.original.log').open('xb') as log:
            run=subprocess.run(command,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,timeout=min(930,max(1,total_budget-(time.monotonic()-start))))
        if run.returncode:
            status='CANDIDATE_ERROR'
            if (current/'supervisor_receipt.json').exists():status=json.loads((current/'supervisor_receipt.json').read_text())['status']
            details=current/'stderr.original.log'
            if details.exists():
                with details.open('rb') as f:
                    f.seek(max(0,details.stat().st_size-8000));failure=f.read().decode(errors='replace')
                if 'Hard gate:' in failure:status='HARD_GATE'
            break
        fresh=json.loads((current/'supervisor_receipt.json').read_text())
        reloaded=out/f'seed_{seed}_reload'
        command=[*command];command[command.index('--out')+1]=str(reloaded)
        command+=['--reload-checkpoint',str(current/'model.pt')]
        with (out/f'seed_{seed}-reload-controller.original.log').open('xb') as log:
            rerun=subprocess.run(command,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,timeout=min(150,max(1,total_budget-(time.monotonic()-start))))
        if rerun.returncode:status='CANDIDATE_ERROR';break
        loaded=json.loads((reloaded/'supervisor_receipt.json').read_text())
        import numpy as np
        prediction=np.load(current/'prediction.npy',allow_pickle=False)
        re_prediction=np.load(reloaded/'prediction.npy',allow_pickle=False)
        difference=float(np.max(np.abs(prediction-re_prediction)))
        if difference>1e-7 or any((fresh['metrics'][k] is None)!=(loaded['metrics'][k] is None) or (fresh['metrics'][k] is not None and abs(fresh['metrics'][k]-loaded['metrics'][k])>1e-8) for k in ['mse','r2','pcc']):
            status='HARD_GATE';failure='Fresh process reload mismatch';break
        rows.append({'seed':seed,'metrics':fresh['metrics'],'reload_max_abs_difference':difference,
          'fresh_child_pid':fresh['actual_child_pid'],'reload_child_pid':loaded['actual_child_pid'],
          'actual_peak_rss_bytes':fresh['actual_peak_rss_bytes'],'checkpoint_sha256':fresh['artifact_sha256']['model.pt']})
    report={'runtime_binding_sha256':runtime_sha,'run_id':identifier,'status':status,'split':a.split,'seeds':seeds,'method_sha256':request['method_sha256'],
      'rows':rows,'elapsed_seconds':time.monotonic()-start,'scorer_reads_candidate_metrics':False,
      'failure_reason':failure,'candidate_imported_on_host':False,'all_seeds_complete':len(rows)==len(seeds)}
    if status=='COMPLETE' and len(rows)==5:
        mean=statistics.mean(x['metrics']['mse'] for x in rows)
        anchors=json.loads((root/'submission/workspace/harbor_task/tests/anchors.json').read_text())
        if a.split=='hidden':anchors['hidden']=json.loads((root/'private/private_verifier_contract_v1.json').read_text())['hidden_anchor']
        anchor=anchors[a.split];assert anchor['status']=='CALIBRATED' and anchor['baseline']>anchor['upper_bound']
        report.update(raw_metric=mean,score=(anchor['baseline']-mean)/(anchor['baseline']-anchor['upper_bound']),
          quality_gates_passed=True,independent_fresh_reload_verified=True)
    if status=='HARD_GATE':report['score']=-1
    (out/'result.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
    raise SystemExit(0 if report['status']=='COMPLETE' else 1)

if __name__=='__main__':main()
