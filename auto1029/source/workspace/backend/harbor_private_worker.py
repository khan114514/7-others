"""Trusted root worker: fixed full private scorer; candidate source stays data."""
import argparse,hashlib,json,os,runpy,sys,time
from pathlib import Path

def main():
    from deployment_paths import PRIVATE_ROOT,PROJECT_ROOT
    from private_verifier_service import regular,atomic,validate_submission,SEEDS,UUID
    parser=argparse.ArgumentParser();parser.add_argument('--job',required=True)
    parser.add_argument('--expected-worker-sha256',required=True);parser.add_argument('--expected-service-sha256',required=True)
    parser.add_argument('--expected-support-sha256',required=True)
    args=parser.parse_args()
    if not UUID.fullmatch(args.job):raise ValueError('Owned private job UUID required')
    exit_code=1;folder=PRIVATE_ROOT/'private-verifier-jobs'/args.job
    before=time.time()
    try:
        if os.geteuid()!=0:raise PermissionError('Trusted root worker required')
        digest=lambda x:hashlib.sha256(x).hexdigest()
        assert digest(regular(Path(__file__).resolve()))==args.expected_worker_sha256
        assert digest(regular(PRIVATE_ROOT/'private_verifier_service.py'))==args.expected_service_sha256
        request=json.loads(regular(folder/'request.json',4*1024*1024))
        checked=validate_submission(request['role'],{k:request[k] for k in ['job_id','files','method_notes','seeds']}|{'action':'submit'})
        assert checked['policy_sha256']==request['policy_sha256']
        assert checked['method_notes_sha256']==request['method_notes_sha256']
        assert request['worker_sha256']==args.expected_worker_sha256 and request['service_sha256']==args.expected_service_sha256
        support=PRIVATE_ROOT/'trusted_private_backend_support.py'
        assert digest(regular(support))==args.expected_support_sha256
        assert request['backend_support_sha256']==args.expected_support_sha256
        functions=runpy.run_path(str(support),run_name='trusted_backend_support_only')
        verify_backend,injection,wait_for_lease=[functions[k] for k in ['verify_backend','injection','wait_for_lease']]
        canonical,config=verify_backend(PROJECT_ROOT)
        injection(PROJECT_ROOT,canonical)
        private_job=PRIVATE_ROOT/'hidden-validation-jobs'/args.job
        if private_job.exists():raise ValueError('Preserve existing private inference; never overwrite')
        private_job.mkdir(mode=0o700)
        atomic(private_job/'request.json',dict(files=request['files'],method_notes=request['method_notes'],source_kind='native_verified_artifact'))
        state=dict(phase='WAITING_FOR_GPU_EXCLUDED',candidate_imported_by_host=False,model_research_seconds=0)
        def save(**fields):
            state.update(fields,observed_epoch=time.time());atomic(folder/'worker-status.json',state)
        save()
        sys.path.insert(0,config['torchtune_package'])
        import gpu_official_encoder_smoke
        def acquire():
            lease=wait_for_lease(config);save(phase='PRIVATE_HARBOR_GPU_RUNNING',gpu_started_epoch=time.time());return lease
        gpu_official_encoder_smoke.wait_for_idle=acquire
        # Only the immutable trusted scorer is executed here. It constructs the
        # already verified isolated PolicyExecutor for the candidate file data.
        sys.argv=[str(PRIVATE_ROOT/'run_private_hidden_score_v2.py'),'--job',args.job,'--mode','full']
        runpy.run_path(sys.argv[0],run_name='__main__')
        result_path=private_job/'result-full/result.json';result=json.loads(regular(result_path))
        assert result['complete_private_protocol'] is True and result['seeds']==SEEDS and result['samples_per_seed']==1380
        assert result['policy_sha256']==request['policy_sha256'] and result['candidate_imported_by_host'] is False
        keys=['complete_private_protocol','policy_sha256','private_protocol_sha256','development_protocol_sha256',
              'seeds','samples_per_seed','normalized_score','unconditional_normalized_score','all_quality_gates_pass',
              'measured_metric','B','U','scorer_sha256','decoder_sha256']
        summary={k:result[k] for k in keys}
        summary.update(complete_primary_protocol=False,private_verifier_route=True,
            original_private_result_sha256=digest(regular(result_path)),candidate_imported_by_host=False,
            method_notes_sha256=request['method_notes_sha256'],source_kind='native_verified_artifact',
            model_research_seconds=0,official_platform_hidden_claim=False)
        atomic(folder/'verifier-summary.json',summary)
        save(phase='COMPLETE',completed_epoch=time.time());exit_code=0
    except Exception as error:
        atomic(folder/'worker-error.json',dict(error_type=type(error).__name__,message=str(error)[:1000],
            candidate_imported_by_host=False,score_claim=False,completed_epoch=time.time()))
        raise
    finally:
        atomic(folder/'execution.json',dict(exit_code=exit_code,started_epoch=before,completed_epoch=time.time(),model_research_seconds=0))

if __name__=='__main__':main()
