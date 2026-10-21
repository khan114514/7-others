"""Harbor0.23.0 Docker provider extension for the authorized hybrid topology.

Native Harbor owns task setup, NOP, artifacts, separate Verifier and rewards.
This adapter adds network-none role queues and a trusted numerical/model relay.
It does not declare native container GPU allocation capability.
"""
import asyncio
import hashlib
import json
import subprocess
import shutil
import re
import uuid
from pathlib import Path
from harbor.environments.docker.docker import DockerEnvironment
from harbor.models.task.config import NetworkMode
from harbor.environments.docker.utils import ensure_docker_image_built
from hybrid_relay import HybridRelay
import bounded_artifact_io

BASE = 'python@sha256:02108f5d322dd89f1c9e552442c25acb0543dfdbc455693a5599624f20d9155d'
PRIVATE = Path(__file__).resolve().parent
DOCKER = shutil.which('docker')
if not DOCKER:
    raise RuntimeError('The trusted launcher must expose the Docker CLI on PATH')


class HybridDockerEnvironment(DockerEnvironment):
    @staticmethod
    def _requires_egress_control(**kwargs):
        # All supported policies are no-network. Enforced directly by Docker's
        # network namespace, without an unrelated egress sidecar.
        return False

    @property
    def capabilities(self):
        return super().capabilities.model_copy(update={'disable_internet': True})

    async def _apply_network_policy(self, network_policy):
        if network_policy.network_mode != NetworkMode.NO_NETWORK or network_policy.allowed_hosts:
            raise ValueError('The hybrid provider supports only no-network policies')
        observed = await self._run_docker_compose_command(['ps', '-q', 'main'])
        if observed.stdout and observed.stdout.strip():
            info = json.loads(self._docker('inspect', observed.stdout.strip()).stdout)[0]
            if info['HostConfig']['NetworkMode'] != 'none':
                raise RuntimeError('Actual Docker network namespace is not isolated')

    def _docker(self, *arguments):
        result = subprocess.run([DOCKER, *arguments], capture_output=True, text=True, timeout=60)
        if result.returncode:
            raise RuntimeError('Own Docker operation failed: ' + result.stderr[-1000:])
        return result

    def __init__(self, *args, lane='gpt', numerical_status=None, author_oracle=False,
                 runtime_root=None, credential_provider=None, model_proxy=None, **kwargs):
        environment_dir = Path(kwargs['environment_dir'])
        if environment_dir.name not in ['environment', 'tests'] or lane not in ['gpt', 'seed']:
            raise ValueError('Unknown task context or physical model lane')
        self.role = 'verifier' if environment_dir.name == 'tests' else lane
        if type(author_oracle) is not bool:
            raise ValueError('Author Oracle mode must be an explicit boolean')
        self.author_oracle = author_oracle and self.role != 'verifier'
        self.identifier = uuid.uuid4().hex
        if numerical_status is None or runtime_root is None:
            raise ValueError('Trusted launcher must supply numerical_status and runtime_root')
        self.credential_provider = credential_provider
        self.model_proxy = model_proxy
        self.own_root = Path(runtime_root).resolve() / 'hybrid-harbor-runtime' / self.identifier
        self.own_root.mkdir(parents=True, exist_ok=False)
        self.verifier_log_shadow = None
        mounts = []
        trial_root = Path(kwargs['trial_paths'].trial_dir).resolve()
        for item in kwargs.get('mounts', []) or []:
            raw = item if isinstance(item, dict) else item.model_dump()
            if raw.get('type') == 'bind' and raw.get('target') in ['/logs/agent', '/logs/verifier', '/logs/artifacts']:
                source = Path(raw['source']).resolve()
                if not source.is_relative_to(trial_root):
                    raise ValueError('Log bind is outside this trial')
                if self.role != 'verifier' and raw['target'] == '/logs/verifier':
                    # Native Harbor's default Agent mount includes its verifier
                    # log directory. Give the Agent its own empty shadow so it
                    # cannot observe or overwrite the separate Verifier output.
                    source = self.own_root / 'agent-verifier-log-shadow'
                    self.verifier_log_shadow = source
                    item = ({**raw, 'source': str(source)} if isinstance(item, dict)
                            else item.model_copy(update={'source': str(source)}))
                source.mkdir(parents=True, exist_ok=True)
                source.chmod(0o777)
            mounts.append(item)
        kwargs['mounts'] = mounts
        self.volume = 'bifrost1-harbor-q-' + self.identifier
        self.broker = 'bifrost1-harbor-broker-' + self.identifier
        self.numerical_status = str(Path(numerical_status).resolve())
        self.relay = None; self.volume_created = False; self.broker_created = False
        security = {'network_mode': 'none', 'cap_drop': ['ALL'], 'security_opt': ['no-new-privileges:true'],
                    'user': '1000:1000', 'read_only': self.role != 'verifier',
                    'tmpfs': ['/tmp:rw,nosuid,nodev,size=256m,uid=1000,gid=1000,mode=1777',
                              '/workspace/solution:rw,nosuid,nodev,size=128m,uid=1000,gid=1000,mode=1777',
                              '/home/candidate:rw,nosuid,nodev,size=256m,uid=1000,gid=1000,mode=0700']}
        if self.role == 'verifier':
            security['tmpfs'][-1] = '/home/verifier:rw,nosuid,nodev,size=64m,uid=1000,gid=1000,mode=0700'
        if self.author_oracle:
            # Native Oracle uploads author material only for its own proof.
            # Formal candidate roles never receive this mount or that material.
            security['tmpfs'].append('/solution:rw,nosuid,nodev,size=4m,uid=1000,gid=1000,mode=1777')
        security['volumes'] = [{'type': 'volume', 'source': 'role_queue', 'target': '/run/bifrost/queue'}]
        compose = {'services': {'main': security}, 'volumes': {'role_queue': {'external': True, 'name': self.volume}}}
        self.compose_path = self.own_root / 'role-compose.json'
        self.compose_path.write_text(json.dumps(compose, indent=2) + '\n')
        kwargs['extra_docker_compose'] = [*kwargs.get('extra_docker_compose', []), self.compose_path]
        kwargs['persistent_env'] = {**kwargs.get('persistent_env', {}), **({'BIFROST_LANE': lane} if self.role != 'verifier' else {})}
        super().__init__(*args, **kwargs)
        if self.network_policy.network_mode != NetworkMode.NO_NETWORK:
            raise ValueError('Hybrid task must explicitly declare no-network')

    async def start(self, force_build=False):
        try:
            # Build each source context once through Harbor's content-addressed
            # cache. Both Agent lanes then use the same immutable image ID;
            # native Compose project names cannot cause separate rebuilds.
            image_name = await ensure_docker_image_built(
                docker_name='bifrost1-task-verifier' if self.role == 'verifier' else 'bifrost1-task-agent',
                docker_build_context=self.environment_dir,
                dockerfile_path=self.environment_dir / 'Dockerfile', build_args={},
                platform='linux/arm64', push=False, timeout_sec=120, logger=self.logger)
            image = json.loads(self._docker('image', 'inspect', image_name).stdout)[0]
            pinned_image_id = image['Id']
            self.task_env_config = self.task_env_config.model_copy(update={'docker_image': pinned_image_id})
            self._env_vars.prebuilt_image_name = pinned_image_id
            self._docker('volume', 'create', '--driver', 'local', '--opt', 'type=tmpfs', '--opt', 'device=tmpfs',
                '--opt', 'o=size=128m,uid=1000,gid=1000,mode=0700',
                '--label', 'codex.task=bifrost1-20261005', '--label', 'codex.purpose=harbor-role-queue', self.volume)
            self.volume_created = True
            # A timed-out Docker start can leave a Created container. Its unique
            # name is owned before issuing run, so failure cleanup includes it.
            self.broker_created = True
            self._docker('run', '--pull=never', '-d', '--name', self.broker, '--network', 'none', '--read-only',
                '--user', '1000:1000', '--cap-drop', 'ALL', '--security-opt', 'no-new-privileges',
                '--memory', '512m', '--cpus', '1', '--label', 'codex.task=bifrost1-20261005',
                '--mount', 'type=volume,source=' + self.volume + ',target=/q/' + self.role, BASE, 'sleep', 'infinity')
            code = (PRIVATE / 'runtime-build/queue_broker.py').read_text()
            self.relay = HybridRelay(DOCKER, self.broker, self.role, self.own_root / 'relay', self.numerical_status, code, credential_provider=self.credential_provider, model_proxy=self.model_proxy)
            await super().start(force_build=force_build)
            result = await self._run_docker_compose_command(['ps', '-q', 'main'])
            container = json.loads(self._docker('inspect', result.stdout.strip()).stdout)[0]
            if container['Image'] != pinned_image_id:
                raise RuntimeError('Actual role container does not use the pinned source image')
            (self.own_root / 'container-inspect.json').write_text(json.dumps(container, indent=2) + '\n')
            config = container['HostConfig']
            if config['NetworkMode'] != 'none' or config['CapDrop'] != ['ALL'] or container['Config']['User'] != '1000:1000':
                raise RuntimeError('Actual role sandbox differs from the frozen security policy')
            volume_mounts = [item for item in container['Mounts'] if item['Type'] == 'volume']
            if len(volume_mounts) != 1 or volume_mounts[0]['Name'] != self.volume:
                raise RuntimeError('A peer or unexpected volume entered the role sandbox')
            allowed_logs = {'/logs/agent', '/logs/verifier', '/logs/artifacts'}
            if any(item['Destination'] not in allowed_logs for item in container['Mounts'] if item['Type'] == 'bind'):
                raise RuntimeError('Unexpected host bind entered the role sandbox')
            receipt = {'environment_id': self.identifier, 'physical_role': self.role, 'session_id': self.session_id,
                       'container_id': container['Id'], 'image_id': container['Image'], 'queue_volume': self.volume,
                       'content_addressed_image_name': image_name,
                       'Agent_uses_private_verifier_log_shadow': self.verifier_log_shadow is not None,
                       'adapter_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                       'relay_sha256': hashlib.sha256((PRIVATE / 'hybrid_relay.py').read_bytes()).hexdigest(),
                       'broker_source_sha256': hashlib.sha256(code.encode()).hexdigest(),
                       'author_oracle': self.author_oracle,
                       'native_container_GPU': False, 'remote_trusted_GPU': 'A100 through bounded tensor DAG'}
            (self.own_root / 'environment-binding.json').write_text(json.dumps(receipt, indent=2) + '\n')
        except BaseException:
            try:
                observed = await self._run_docker_compose_command(['ps', '--all', '-q', 'main'])
                if observed.stdout and observed.stdout.strip():
                    container_id = observed.stdout.strip()
                    for operation in ['logs', 'inspect']:
                        diagnostic = subprocess.run([DOCKER, operation, container_id],
                            capture_output=True, text=True, timeout=10)
                        (self.own_root / ('failed-start-' + operation + '.stdout.log')).write_text(diagnostic.stdout)
                        (self.own_root / ('failed-start-' + operation + '.stderr.log')).write_text(diagnostic.stderr)
            except Exception as diagnostic_error:
                (self.own_root / 'failed-start-diagnostic-error.txt').write_text(str(diagnostic_error)[:1000])
            await self.stop(delete=True)
            raise

    async def stop(self, delete=True):
        try:
            await super().stop(delete=delete)
        finally:
            if self.relay:
                await asyncio.to_thread(self.relay.close); self.relay = None
            if self.broker_created:
                removed = subprocess.run([DOCKER, 'rm', '-f', self.broker], capture_output=True, text=True, timeout=30)
                if removed.returncode and 'No such container' not in removed.stderr:
                    raise RuntimeError('Own broker cleanup failed: ' + removed.stderr[-1000:])
                self.broker_created = False
            if self.volume_created and delete:
                self._docker('volume', 'rm', self.volume); self.volume_created = False

    async def _artifact_container(self):
        result = await self._run_docker_compose_command(['ps', '-q', 'main'])
        identifier = result.stdout.strip()
        if not re.fullmatch('[a-f0-9]{12,64}', identifier):
            raise RuntimeError('No live owned container for artifact transfer')
        return identifier

    def _record_artifact_transfer(self, direction, slot, result):
        with (self.own_root / 'artifact-transfers.jsonl').open('a') as handle:
            handle.write(json.dumps({'direction': direction, 'slot': slot, **result}) + '\n')

    async def download_file(self, source_path, target_path):
        source = str(source_path)
        if source not in ['/workspace/solution/program.json', '/workspace/solution/method.py']:
            return await super().download_file(source_path, target_path)
        if not Path(target_path).resolve().is_relative_to(self.trial_paths.artifacts_dir.resolve()):
            raise ValueError('Collected artifact target is outside this trial')
        container = await self._artifact_container()
        result = await asyncio.to_thread(bounded_artifact_io.download, DOCKER, container, source, target_path)
        self._record_artifact_transfer('download', source, result)

    async def upload_file(self, source_path, target_path):
        target = str(target_path)
        ordinary = target in ['/workspace/solution/program.json', '/workspace/solution/method.py']
        oracle = self.author_oracle and target in ['/solution/solve.sh', '/solution/reference_method.py']
        if not ordinary and not oracle:
            return await super().upload_file(source_path, target_path)
        container = await self._artifact_container()
        result = await asyncio.to_thread(bounded_artifact_io.upload, DOCKER, container,
                                         source_path, target, 'root' if oracle else '1000:1000')
        self._record_artifact_transfer('upload', target, result)

    async def upload_dir(self, source_dir, target_dir):
        if str(target_dir) != '/solution' or not self.author_oracle:
            return await super().upload_dir(source_dir, target_dir)
        source_dir = Path(source_dir)
        if source_dir.is_symlink() or {p.name for p in source_dir.iterdir()} != {'solve.sh', 'reference_method.py'}:
            raise ValueError('Author Oracle source tree differs from the declared two files')
        for name in ['solve.sh', 'reference_method.py']:
            await self.upload_file(source_dir / name, '/solution/' + name)
