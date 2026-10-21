"""Actual configured Native build and transport probes; no model or GPU score."""
import argparse
import asyncio
import hashlib
import importlib.metadata
import json
import shlex
import time
import uuid
from pathlib import Path

from harbor.models.trial.config import TrialConfig
from harbor.trial.trial import Trial
from host_paths import PRIVATE_ROOT


async def main(task):
    identifier = uuid.uuid4().hex
    folder = PRIVATE_ROOT / 'portability-probes' / identifier
    folder.mkdir(parents=True)
    state = {'phase': 'ACTUAL_CONFIGURED_NATIVE_PROBE_STARTING', 'started_epoch': time.time(),
             'harbor_version': importlib.metadata.version('harbor'), 'model_research_seconds': 0,
             'model_provider_called': False, 'gpu_score_claim': False, 'stability_12h_claim': False,
             'task_source_manifest': {str(p.relative_to(task)): hashlib.sha256(p.read_bytes()).hexdigest()
                                      for p in task.rglob('*') if p.is_file()},
             'host_source_manifest': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                      for p in Path(__file__).parent.glob('*.py')}}
    result_path = folder / 'result.json'

    def save():
        result_path.write_text(json.dumps(state, indent=2) + '\n')

    save()
    trial = await Trial.create(TrialConfig(
        task={'path': str(task)}, trial_name='heima-portable-' + identifier[:12], trials_dir=folder,
        agent={'name': 'nop'}, environment={'import_path': 'heima_harbor_environment:HeimaDockerEnvironment',
                                          'force_build': True, 'kwargs': {'role': 'gpt'}}))
    env = trial.agent_environment
    try:
        await env.start(True)
        # The normal provider already verifies readonly Dev, real sandbox
        # writing, physical network:none, no bind mounts/devices/capabilities.
        state['native_start_and_dev_probe_pass'] = True
        source = folder / 'transport-sentinel.json'
        source.write_text(json.dumps({'marker': 'TRANSPORT_PROBE_ONLY_NO_SCORE_CLAIM'}) + '\n')
        await env.upload_file(source, '/logs/verifier/transport-sentinel.json')
        probe = await env.exec(command='test ! -r /logs/verifier/transport-sentinel.json && test ! -x /run/heima-verifier/queue',
                               user='1000:1000', timeout_sec=30)
        assert probe.return_code == 0
        await env.download_dir('/logs/verifier', folder / 'actual-export')
        assert (folder / 'actual-export/transport-sentinel.json').read_bytes() == source.read_bytes()
        state['private_namespace_export_and_agent_denial_pass'] = True
        link = await env.exec(command='ln -s /etc/passwd /logs/verifier/ignored-link', user='root', timeout_sec=30)
        assert link.return_code == 0
        await env.download_dir('/logs/verifier', folder / 'safe-export')
        assert not (folder / 'safe-export/ignored-link').exists()
        state['symlink_export_denial_pass'] = True
        writer = "from pathlib import Path;Path('/workspace/solution/large.json').write_text('x'*1048577)"
        probe = await env.exec(command='python -c ' + shlex.quote(writer), user='1000:1000', timeout_sec=30)
        assert probe.return_code == 0
        files = await env.export_candidate()
        assert 'method.py' in files and 'large.json' not in files and 'resume_bundle_manifest.json' in files
        state['bounded_own_snapshot_pass'] = True
        # Exercise the configured SSH bridge through the actual offline Actor
        # queue. Active discovery submits no GPU job and retrieves no key.
        command = "import json,sys;sys.path.insert(0,'/opt/latentcli');from queue_client import request_json;v=request_json('evaluate',{'action':'active'},timeout=60);assert v['role']=='gpt';print(json.dumps(v))"
        probe = await env.exec(command='python -c ' + shlex.quote(command), user='1000:1000', timeout_sec=60)
        (folder / 'actor-owned-active-probe.json').write_text(probe.model_dump_json(indent=2))
        assert probe.return_code == 0
        active = json.loads(probe.stdout)
        assert active['role'] == 'gpt' and active['active'] is None
        state['actual_offline_queue_to_configured_gpu_active_pass'] = True
        from evaluator_bridge import EvaluatorBridge
        other = EvaluatorBridge(folder / 'seed-active-probe')('seed', {'action': 'active'})
        assert other['role'] == 'seed' and other['active'] is None
        state['both_physical_roles_empty_backend_namespace_pass'] = True
        state.update(phase='ACTUAL_CONFIGURED_NATIVE_BUILD_TRANSPORT_PROBES_PASS', completed_epoch=time.time())
        save()
    except Exception as error:
        state.update(phase='FAILED_PRESERVE_EVIDENCE', error_type=type(error).__name__,
                     error=str(error)[:800], completed_epoch=time.time())
        save()
        raise
    finally:
        await env.stop(True)
    print(json.dumps({'folder': str(folder), 'phase': state['phase'], 'model_research_seconds': 0}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', type=Path, required=True)
    args = parser.parse_args()
    asyncio.run(main(args.task.resolve()))
