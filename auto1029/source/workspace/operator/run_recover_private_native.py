"""Portable real Harbor reproduction; policy files are always treated as data.

Defaults assume workspace/operator/run_native.py in the three-directory bundle.
Trusted configuration and all generated logs stay outside the public bundle.
The installed jailed GPU scorer alone executes submitted candidate modules.
"""
import argparse
import asyncio
import hashlib
import importlib.metadata
import json
import os
import shlex
import stat
import sys
import time
import uuid
from pathlib import Path


def regular(path, limit=4 * 1024 * 1024):
    path = Path(path).absolute()
    if path.resolve() != path:
        raise ValueError('Symlink input forbidden')
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(descriptor, 'rb') as stream:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_size > limit:
            raise ValueError('Bounded regular input required')
        raw = stream.read(limit + 1)
        after = os.fstat(descriptor)
    if len(raw) != info.st_size or (info.st_size, info.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError('Input changed or exceeded its budget')
    return raw


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def manifest(root, pattern='*'):
    if not root.is_dir() or root.resolve() != root:
        raise ValueError('Existing nonsymlink source directory required')
    value = {p.relative_to(root).as_posix(): sha(regular(p, 32 * 1024 * 1024))
             for p in sorted(root.rglob(pattern)) if p.is_file()}
    if not value:
        raise ValueError('Source manifest cannot be empty')
    return value


def method_data(folder, reference=False):
    files = {}
    for path in sorted(folder.rglob('*.py')):
        name = path.relative_to(folder).as_posix()
        if Path(name).is_absolute() or '..' in Path(name).parts:
            raise ValueError('Invalid relative module path')
        files[name] = regular(path, 1024 * 1024).decode('utf-8')
    encoded = json.dumps(files, sort_keys=True, separators=(',', ':')).encode()
    if 'method.py' not in files or not 1 <= len(files) <= 64 or len(encoded) > 2 * 1024 * 1024:
        raise ValueError('Complete bounded method entry and source bundle required')
    notes_path = folder / 'method_notes.json'
    if notes_path.exists():
        notes = json.loads(regular(notes_path, 65536))
    elif reference:
        notes = {'algorithm': 'isolated structural-prefix speculative drafting', 'expert_reproduction': True}
    else:
        raise ValueError('Actual final method metadata is required')
    if not isinstance(notes, dict) or not notes or len(json.dumps(notes).encode()) > 65536:
        raise ValueError('Bounded nonempty method metadata required')
    return {'files': files, 'method_notes': notes, 'policy_sha256': sha(encoded)}


async def main(args):
    task = args.task.absolute()
    adapter = args.adapter.absolute()
    workspace = Path(__file__).resolve().parents[1]
    folder = args.method_dir or (workspace / 'reference' if args.policy == 'reference' else
                                workspace.parent / 'expert_evidence/best_method')
    data = None if args.policy == 'baseline' else method_data(folder.absolute(), args.policy == 'reference')
    checked = {'task_source_manifest': manifest(task), 'host_source_manifest': manifest(adapter, '*.py'),
               'runner_sha256': sha(regular(Path(__file__).absolute())),
               'policy_sha256': data['policy_sha256'] if data else None,
               'model_calls': 0, 'model_research_seconds': 0}
    required_task = {'instruction.md', 'task.toml', 'environment/Dockerfile', 'tests/Dockerfile', 'tests/test.sh'}
    required_host = {'heima_harbor_environment.py', 'cli_provider_relay.py', 'native_codex_agent.py',
                     'evaluator_bridge.py', 'host_paths.py'}
    if (not required_task.issubset(checked['task_source_manifest']) or
            not required_host.issubset(checked['host_source_manifest'])):
        raise ValueError('Complete task and trusted host entry points required')
    if args.preflight_only:
        print(json.dumps({'phase': 'STATIC_PORTABLE_NATIVE_REPRODUCTION_PREFLIGHT_PASS',
                          'task_files': len(checked['task_source_manifest']),
                          'host_python_files': len(checked['host_source_manifest']),
                          'method_python_files': len(data['files']) if data else None,
                          'gpu_jobs_submitted': 0, 'candidate_imported_by_host': False,
                          'actual_native_or_score_claim': False, **checked}), flush=True)
        return 0
    sys.path.insert(0, str(adapter))
    # Only trusted framework/adapter modules are imported on the operator host.
    from harbor.agents.base import BaseAgent
    from harbor.models.trial.config import TrialConfig
    from harbor.trial.trial import Trial
    from host_paths import PRIVATE_ROOT, CONFIG_PATH

    async def capture_view(environment, logs):
        program = '''import os,json,stat,hashlib
from pathlib import Path
assert os.getuid()==1000
rows={}
for root in ['/workspace','/opt/latentcli']:
 for directory,dirs,files in os.walk(root,followlinks=False):
  for name in ['.']+sorted(dirs)+sorted(files):
   p=Path(directory) if name=='.' else Path(directory)/name
   st=p.lstat();row={'path':str(p),'uid':st.st_uid,'gid':st.st_gid,'mode':oct(stat.S_IMODE(st.st_mode))}
   if stat.S_ISLNK(st.st_mode):row.update(type='symlink',readable=False)
   elif stat.S_ISDIR(st.st_mode):row.update(type='directory',readable=os.access(p,os.R_OK|os.X_OK))
   elif stat.S_ISREG(st.st_mode):
    try:raw=p.read_bytes()
    except PermissionError:row.update(type='file',readable=False)
    else:row.update(type='file',readable=True,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
   else:row.update(type='special',readable=False)
   rows[str(p)]=row
denied=[]
for name in ['/tests','/solution','/logs/verifier','/run/heima-verifier/queue']:
 try:list(Path(name).iterdir())
 except PermissionError:denied.append(name)
 else:raise AssertionError('Private directory was visible')
print(json.dumps({'verified_as_uid':os.getuid(),'entries':[rows[n] for n in sorted(rows)],'private_directories_denied':denied}))
'''
        reply = await environment.exec(command='python -I -c ' + shlex.quote(program), user='1000:1000', timeout_sec=30)
        (logs / 'agent-view-probe.original.json').write_text(reply.model_dump_json(indent=2))
        if reply.return_code:
            raise ValueError('Actual UID1000 file inventory or private-directory denial failed')
        view = json.loads(reply.stdout)
        (logs / 'agent-view.json').write_text(json.dumps(view, indent=2) + '\n')
        physical = json.loads(regular(environment.audit / 'container-inspect.json', 1024*1024))
        host = physical['HostConfig']
        image = {'image_digest': physical['Image'], 'network_mode': host['NetworkMode'],
                 'readonly_rootfs': host['ReadonlyRootfs'], 'privileged': host['Privileged'],
                 'cap_drop': host['CapDrop'], 'devices': host.get('Devices'), 'device_requests': host.get('DeviceRequests'),
                 'mounts': [{k: row[k] for k in ['Type','Destination','RW']} for row in physical['Mounts']],
                 'original_container_inspect_sha256': sha(regular(environment.audit / 'container-inspect.json',1024*1024))}
        (logs / 'agent-image.json').write_text(json.dumps(image, indent=2) + '\n')

    class DataInstallAgent(BaseAgent):
        @staticmethod
        def name():
            return 'heima-portable-data-install'

        def version(self):
            return '1.0.0'

        async def setup(self, environment):
            return

        async def run(self, instruction, environment, context):
            files = dict(data['files'])
            files['method_notes.json'] = json.dumps(data['method_notes'], ensure_ascii=False, indent=2) + '\n'
            await environment.restore_candidate(files)
            exported = await environment.export_candidate()
            if any(exported.get(n) != raw for n, raw in files.items()):
                raise ValueError('Actual fresh Actor export differs from supplied method bytes')
            await capture_view(environment, self.logs_dir)
            (self.logs_dir / 'data-install.json').write_text(json.dumps({
                'phase': 'ACTUAL_UID1000_METHOD_DATA_HANDOFF_PASS', 'policy_sha256': data['policy_sha256'],
                'python_source_files': len(data['files']), 'candidate_imported_by_host': False,
                'model_calls': 0, 'model_research_seconds': 0}, indent=2) + '\n')

    # Harbor resolves this trusted installer from the entry point module.
    setattr(sys.modules[__name__], 'DataInstallAgent', DataInstallAgent)
    sys.modules['heima_portable_operator'] = sys.modules[__name__]
    from harbor.trial.single_step import SingleStepTrial
    class NoWallPrivateTrial(SingleStepTrial):
        def _compute_verifier_timeout_sec(self):
            return None

        @classmethod
        async def create(cls, config):
            cls._resolve_agent_skills(config)
            task, download = await cls._load_task(config)
            if task.has_steps or config.source_trial is not None:
                raise ValueError('This verified operator supports the single-step task only')
            trial = cls(config, _task=task, _task_download_result=download)
            await trial._create_bridge()
            if trial._verifier_timeout_sec is not None:
                raise ValueError('Private verifier must have no wall-clock cutoff')
            return trial

    identifier = uuid.uuid4().hex
    output = PRIVATE_ROOT / 'portable-operator-runs' / identifier
    output.mkdir(parents=True, mode=0o700)
    checked['host_config_sha256'] = sha(regular(CONFIG_PATH, 65536))
    state = {**checked, 'phase': 'ACTUAL_PORTABLE_NATIVE_REPRODUCTION_STARTING', 'pid': os.getpid(),
             'id': identifier, 'policy': args.policy, 'probe_only': args.probe_only,
             'started_epoch': time.time(), 'harbor_version': importlib.metadata.version('harbor'),
             'official_hidden_or_platform_acceptance_claim': False, 'private_verifier_no_wall_clock_cutoff': True}

    def save(**fields):
        state.update(fields, observed_epoch=time.time())
        path = output / 'status.pending.json'
        path.write_text(json.dumps(state, indent=2) + '\n'); path.replace(output / 'status.json')

    save()
    agent = {'name': 'nop'} if data is None else {'import_path': 'heima_portable_operator:DataInstallAgent'}
    trial = await NoWallPrivateTrial.create(TrialConfig(task={'path': str(task)},
        trial_name='heima-portable-reproduction-' + identifier[:12], trials_dir=output,
        agent=agent, environment={'import_path': 'heima_harbor_environment:HeimaDockerEnvironment',
            'force_build': True, 'kwargs': {'role': args.role}}))
    save(actual_trial_class=type(trial).__name__, actual_verifier_timeout_sec=trial._verifier_timeout_sec)
    try:
        if args.probe_only:
            await trial.agent_environment.start(True)
            if data is not None:
                await trial.agent.run('', trial.agent_environment, None)
            else:
                await capture_view(trial.agent_environment, trial.paths.agent_dir)
            state['gpu_jobs_submitted'] = 0
            save(phase='ACTUAL_PORTABLE_NATIVE_BUILD_DATA_HANDOFF_PROBE_PASS',
                 completed_epoch=time.time(), complete_scored_trial_claim=False)
        else:
            save(phase='ACTUAL_PORTABLE_FULL_NATIVE_TRIAL_RUNNING')
            result = await trial.run()
            (output / 'native-returned-result.json').write_text(result.model_dump_json(indent=2))
            if result.exception_info or result.verifier_environment_mode != 'separate':
                raise ValueError('Successful real separate Native verifier required')
            score = json.loads(regular(trial.paths.verifier_dir / 'scorer-result.json'))
            reward = float(regular(trial.paths.verifier_dir / 'reward.txt', 1024))
            value = score['result']
            if (value['seeds'] != [17, 29, 43, 71, 101] or value['samples_per_seed'] != 1380 or
                    not value['complete_private_protocol'] or not value['private_verifier_route'] or not value['all_quality_gates_pass'] or
                    
                    value['normalized_score'] != reward or result.verifier_result.rewards['reward'] != reward):
                raise ValueError('Actual complete fixed-five-seed reward differs')
            if data is not None and score['policy_sha256'] != data['policy_sha256']:
                raise ValueError('Actual scored policy differs from supplied method')
            save(phase='ACTUAL_PORTABLE_FULL_NATIVE_REPRODUCTION_PASS', normalized_score=reward,
                 measured_metric=value['measured_metric'], actual_gpu_job_id=score['job_id'],
                 completed_epoch=time.time(), complete_scored_trial_claim=True)
        if (manifest(task) != checked['task_source_manifest'] or manifest(adapter, '*.py') != checked['host_source_manifest']
                or sha(regular(CONFIG_PATH, 65536)) != checked['host_config_sha256'] or
                sha(regular(Path(__file__).absolute())) != checked['runner_sha256']):
            raise ValueError('Reproduction source or operator configuration changed')
        if args.probe_only:
            save(actual_agent_view_relative=str((trial.paths.agent_dir/'agent-view.json').relative_to(output)),
                 actual_agent_image_relative=str((trial.paths.agent_dir/'agent-image.json').relative_to(output)))
            pointer = PRIVATE_ROOT/'portable-operator-runs/latest-probe.pending.json'
            pointer.write_text(json.dumps({'id':identifier,'folder':str(output),'model_research_seconds':0},indent=2)+'\n')
            pointer.replace(pointer.with_name('latest-probe.json'))
    except Exception as error:
        save(phase='FAILED_PRESERVE_ORIGINAL_REPRODUCTION_EVIDENCE', error_type=type(error).__name__,
             reason=str(error)[:1000], completed_epoch=time.time(), complete_scored_trial_claim=False)
        raise
    finally:
        if args.probe_only:
            await trial.agent_environment.stop(True)
    print(json.dumps({'folder': str(output), 'phase': state['phase'], 'model_research_seconds': 0}), flush=True)
    return 0


if __name__ == '__main__':
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument('--policy', choices=['baseline', 'reference', 'best'], required=True)
    parser.add_argument('--role', choices=['gpt', 'seed'], default='gpt')
    parser.add_argument('--task', type=Path, default=root / 'harbor_task')
    parser.add_argument('--adapter', type=Path, default=root / 'host_adapter')
    parser.add_argument('--method-dir', type=Path)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--preflight-only', action='store_true')
    group.add_argument('--probe-only', action='store_true')
    raise SystemExit(asyncio.run(main(parser.parse_args())))
