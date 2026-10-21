"""Stock Harbor NOP trial with already-built local images selected in memory.

The task files, Dockerfiles, resource policies and verifier stay unchanged.
This only avoids fetching registry metadata again after a Docker Hub outage.
"""
import asyncio
import hashlib
import json
from pathlib import Path
import subprocess

HERE=Path(__file__).resolve().parent
NOP=HERE/'native-nop-current-version'
DOCKER='<AUTHOR_APPLICATIONS>/Docker.app/Contents/Resources/bin/docker'

async def main():
    from harbor.models.trial.config import TrialConfig
    from harbor.trial.trial import Trial
    from harbor.trial.single_step import SingleStepTrial
    config=TrialConfig.model_validate_json((NOP/'config.json').read_text())
    Trial._resolve_agent_skills(config)
    task,download=await Trial._load_task(config)
    images=json.loads((NOP/'docker-image-inspection.json').read_text())
    for row in images:
        actual=subprocess.check_output([DOCKER,'image','inspect',row['image_id'],'--format','{{.Id}}'],text=True).strip()
        assert actual==row['image_id'] and row['os']=='linux'
    agent=images[0]['image_id'];verifier=images[1]['image_id']
    assert task.config.verifier.environment_mode=='separate'
    task.config.environment.docker_image=agent
    task.config.verifier.environment.docker_image=verifier
    before={str(p.relative_to(task.paths.task_dir)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in task.paths.task_dir.rglob('*') if p.is_file()}
    proof={'scope':'In-memory cached-image selection; stock Harbor SingleStepTrial lifecycle',
        'agent_image_id':agent,'verifier_image_id':verifier,'task_files_modified':False,
        'resource_or_network_policies_modified':False,'verifier_replaced':False,
        'stock_harbor_source_modified':False,'new_training':False,'image_build_evidence':'docker-build-completion.json',
        'task_file_sha256s_before':before,'qa_pass_claimed':False}
    (NOP/'cached-image-adapter-provenance.json').write_text(json.dumps(proof,indent=2)+'\n')
    trial=SingleStepTrial(config,_task=task,_task_download_result=download)
    result=await trial.run()
    after={str(p.relative_to(task.paths.task_dir)):hashlib.sha256(p.read_bytes()).hexdigest()
           for p in task.paths.task_dir.rglob('*') if p.is_file()}
    assert before==after
    (NOP/'cached-image-adapter-terminal.json').write_text(json.dumps({'task_files_unchanged':True,
        'trial_name':config.trial_name,'exception_present':result.exception_info is not None,'qa_pass_claimed':False},indent=2)+'\n')
    print(result.model_dump_json(indent=2))

if __name__=='__main__':asyncio.run(main())
