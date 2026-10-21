"""Dynamic current-image Harbor provider/isolation proof, explicitly not NOP."""
import sys
sys.dont_write_bytecode=True
import asyncio
import hashlib
import json
import time
import uuid
from pathlib import Path
from harbor.environments.factory import EnvironmentFactory
from harbor.models.task.task import Task
from harbor.models.trial.config import EnvironmentConfig
from harbor.models.trial.paths import TrialPaths
from release_source_binding import current_binding
import argparse

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--task',type=Path,required=True)
parser.add_argument('--runtime-state',type=Path,required=True)
parser.add_argument('--numerical-status',type=Path,required=True)
args=parser.parse_args()
TASK_PATH=args.task.resolve()
PRIVATE=args.runtime_state.resolve()
RUNTIME=Path(__file__).resolve().parent
if PRIVATE.is_relative_to(TASK_PATH) or PRIVATE.is_relative_to(RUNTIME):raise ValueError('Evidence state must remain outside Task and Runtime')
TASK = Task(TASK_PATH)
identifier = uuid.uuid4().hex
root = PRIVATE / 'harbor-isolation-preflight' / identifier
root.mkdir(parents=True, exist_ok=False)
evidence = {'run_id': identifier, 'started_epoch': time.time(), 'is_Harbor_NOP': False,
            'is_formal_trajectory': False, 'formal_effective_seconds': 0, 'Model_GPU_called':False,'roles': {}}
source_binding = current_binding(TASK_PATH)
evidence.update(source_binding)


async def main():
    environments = []
    try:
        for role in ['gpt', 'seed', 'verifier']:
            folder = root / role
            for name in ['agent', 'verifier', 'artifacts']:
                path = folder / name; path.mkdir(parents=True, exist_ok=True); path.chmod(0o777)
            config = TASK.config.verifier.environment if role == 'verifier' else TASK.config.environment
            context = TASK.paths.tests_dir if role == 'verifier' else TASK.paths.environment_dir
            mounts = [{'type': 'bind', 'source': str((folder / 'verifier').resolve()), 'target': '/logs/verifier'}]
            if role != 'verifier':
                mounts.extend([{'type': 'bind', 'source': str((folder / name).resolve()), 'target': '/logs/' + name}
                               for name in ['agent', 'artifacts']])
            env = EnvironmentFactory.create_environment_from_config(
                config=EnvironmentConfig(import_path='hybrid_harbor_environment:HybridDockerEnvironment',
                                         kwargs={'lane': 'seed' if role == 'seed' else 'gpt','runtime_root':str(PRIVATE),'numerical_status':str(args.numerical_status.resolve())}),
                environment_dir=context, environment_name='bifrost-preflight-' + role,
                session_id='bifrost-isolation-' + identifier[:12] + '-' + role,
                trial_paths=TrialPaths(trial_dir=folder), task_env_config=config,
                mounts=mounts, network_policy=config.resolve_baseline())
            environments.append(env)
            await asyncio.wait_for(env.start(force_build=False), timeout=180)
            evidence['roles'][role] = {'environment_id': env.identifier,
                'binding': str((env.own_root / 'environment-binding.json').relative_to(PRIVATE)),
                'queue_volume': env.volume, 'default_USER': '1000:1000'}
        images = {role: json.loads((PRIVATE / data['binding']).read_text())['image_id']
                  for role, data in evidence['roles'].items()}
        if images['gpt'] != images['seed'] or images['verifier'] == images['gpt']:
            raise RuntimeError('Agent lanes must share exactly one image digest and Verifier must use a distinct image')
        evidence['image_ids'] = images
        evidence['same_Agent_image_digest'] = True
        transfer_root = root / 'gpt' / 'artifacts' / 'transfer-proof'
        transfer_root.mkdir(parents=True)
        for name in ['program.json', 'method.py']:
            target = transfer_root / name
            await environments[0].download_file('/workspace/solution/' + name, target)
            await environments[2].upload_file(target, '/workspace/solution/' + name)
            code = "from pathlib import Path;import hashlib;print(hashlib.sha256(Path('/workspace/solution/" + name + "').read_bytes()).hexdigest())"
            result = await environments[2].exec(command='python -I -c ' + __import__('shlex').quote(code), user='1000:1000')
            assert result.return_code == 0 and result.stdout.strip() == hashlib.sha256(target.read_bytes()).hexdigest()
        evidence['live_tmpfs_artifact_roundtrip'] = True
        rejected = []
        for kind, mutation in [
            ('symlink', "p.unlink();p.symlink_to('/etc/passwd')"),
            ('oversized', "p.write_bytes(b'X'*524289)"),
            ('FIFO', "p.unlink();os.mkfifo(p)")]:
            code = "from pathlib import Path;import os;p=Path('/workspace/solution/program.json');" + mutation
            result = await environments[0].exec(command='python -I -c ' + __import__('shlex').quote(code), user='1000:1000')
            assert result.return_code == 0, result.stderr or result.stdout
            try:
                await environments[0].download_file('/workspace/solution/program.json', transfer_root / ('rejected-' + kind + '.json'))
            except (RuntimeError, ValueError):
                rejected.append(kind)
            else:
                raise AssertionError('Unsafe artifact was collected: ' + kind)
            await environments[0].upload_file(transfer_root / 'program.json', '/workspace/solution/program.json')
        evidence['unsafe_artifact_collection_rejected'] = rejected
        queues = [env.volume for env in environments]
        assert len(set(queues)) == 3
        for role, env in zip(['gpt', 'seed', 'verifier'], environments):
            code = '''import json,os,socket,hashlib
from pathlib import Path
blocked=['/var/run/docker.sock','/root/.ssh','/q/gpt','/q/seed','/q/verifier','/solution']
path_probe={}
for p in blocked:
 try:
  present=Path(p).exists()
 except PermissionError:
  path_probe[p]='permission_denied'
 else:
  assert not present, 'Forbidden host or peer path is visible: '+p
  path_probe[p]='absent'
assert not any(k in os.environ for k in ['OPENAI_API_KEY','ARK_API_KEY','BIFROST_MODCON_API_KEY','BIFROST_ARK_API_KEY'])
s=socket.socket();s.settimeout(1);code=s.connect_ex(('1.1.1.1',443));assert code!=0
print(json.dumps({'host_and_peer_paths_inaccessible':True,'path_probe':path_probe,'credential_env_absent':True,'egress_connect_result':code,'uid':os.getuid()}))'''
            command = 'python -c ' + __import__('shlex').quote(code)
            result = await env.exec(command=command, user='1000:1000')
            assert result.return_code == 0, result.stderr or result.stdout
            evidence['roles'][role]['negatives'] = json.loads(result.stdout)
            if role != 'verifier':
                code = '''from pathlib import Path
import json,hashlib
assert not Path('/tests').exists()
assert not any('reference' in p.name.lower() for p in Path('/opt/bifrost').rglob('*'))
p=json.loads(Path('/workspace/solution/program.json').read_text())
print(json.dumps({'starter_sha256':hashlib.sha256(json.dumps(p,sort_keys=True,separators=(',',':')).encode()).hexdigest(),'hidden_absent':True,'reference_absent':True}))'''
                result = await env.exec(command='python -c ' + __import__('shlex').quote(code), user='1000:1000')
                assert result.return_code == 0, result.stderr or result.stdout
                evidence['roles'][role]['starter'] = json.loads(result.stdout)
                result = await env.exec(command="python -c 'from pathlib import Path; Path(\"/logs/verifier/candidate-log-shadow-probe\").write_text(\"candidate-owned-shadow\")'", user='1000:1000')
                assert result.return_code == 0, result.stderr or result.stdout
                assert env.verifier_log_shadow is not None
                assert (env.verifier_log_shadow / 'candidate-log-shadow-probe').exists()
                assert not (root / role / 'verifier/candidate-log-shadow-probe').exists()
                evidence['roles'][role]['trusted_verifier_logs_unmounted'] = True
                baseline = json.loads((TASK_PATH / 'environment/public_eval/baseline-program.json').read_text())
                expected = hashlib.sha256(json.dumps(baseline, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
                assert evidence['roles'][role]['starter']['starter_sha256'] == expected
                result = await env.exec(command='env -i PATH=/usr/local/bin:/usr/bin:/bin python /opt/bifrost/public_eval/evaluate.py --help', user='1000:1000')
                assert result.return_code == 0, result.stderr or result.stdout
            else:
                result = await env.exec(command="python -c 'from pathlib import Path; p=Path(\"/logs/verifier/verifier-write-probe\"); p.write_text(\"own-verifier\"); p.unlink()'", user='1000:1000')
                assert result.return_code == 0, result.stderr or result.stdout
                result = await env.exec(command="python -c 'from pathlib import Path; assert Path(\"/tests/hidden/hidden-decoder-inputs-v1.safetensors\").stat().st_size==40912872; print(\"OWN_HIDDEN_PRESENT\")'", user='1000:1000')
                assert result.return_code == 0
            # A role/profile override is rejected before any numerical request,
            # and Verifier cannot call models. These are gateway negatives only.
            code = '''import sys
sys.path.insert(0,'/opt/bifrost')
from queue_client import request_json
try:
 request_json('evaluate',{'program':{},'profile':'formal'},timeout=10)
except RuntimeError:
 print('PROFILE_OVERRIDE_REJECTED')
else:
 raise AssertionError('Profile override was accepted')'''
            result = await env.exec(command='python -c ' + __import__('shlex').quote(code), user='1000:1000')
            assert result.return_code == 0 and 'PROFILE_OVERRIDE_REJECTED' in result.stdout
            assert env.relay.key is None, 'No model credentials should be loaded by this proof'
        if current_binding(TASK_PATH) != source_binding:
            raise RuntimeError('Task or trusted adapter changed during dynamic isolation proof')
        evidence['status'] = 'PASS'
    except Exception as error:
        evidence.update(status='FAIL', error_type=type(error).__name__, error=str(error)[-1800:])
    finally:
        for env in reversed(environments):
            try:
                await env.stop(delete=True)
            except Exception as error:
                evidence.setdefault('cleanup_errors', []).append(type(error).__name__ + ': ' + str(error)[:500])
        if evidence.get('cleanup_errors'):
            evidence['status']='FAIL'
        evidence['completed_epoch'] = time.time()
        (root / 'receipt.json').write_text(json.dumps(evidence, indent=2) + '\n')
        (PRIVATE / 'harbor-isolation-preflight-latest.json').write_text(json.dumps(evidence, indent=2) + '\n')
        print(json.dumps(evidence, indent=2))


if __name__ == '__main__':
    asyncio.run(main())
