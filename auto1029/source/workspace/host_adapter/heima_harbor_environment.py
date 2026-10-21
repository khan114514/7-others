"""Actual native Harbor Docker provider with offline, lane-owned IPC.

No host task/private/log bind mounts. Native Harbor uploads and downloads its
own files through engine copy. The broker is outside the candidate namespace.
"""
import asyncio
import base64
import json
import shlex
import subprocess
from pathlib import Path

import yaml
from harbor.environments.docker.docker import DockerEnvironment
from harbor.models.task.config import NetworkMode
from cli_provider_relay import ProviderRelay
from evaluator_bridge import EvaluatorBridge

from host_paths import PRIVATE_ROOT as PRIVATE
from host_paths import DOCKER_EXECUTABLE as DOCKER
BASE = 'python:3.12-slim@sha256:02108f5d322dd89f1c9e552442c25acb0543dfdbc455693a5599624f20d9155d'


def docker(*argv):
    result = subprocess.run([DOCKER, *argv], capture_output=True, text=True, timeout=60)
    if result.returncode:
        raise RuntimeError('Own Docker operation failed: ' + result.stderr[-1000:])
    return result.stdout


class HeimaDockerEnvironment(DockerEnvironment):
    def __init__(self, *args, role='gpt', **kwargs):
        if role not in {'gpt', 'seed'}:
            raise ValueError('Unknown independent lane')
        self.role = role
        self.is_verifier = '__verifier__' in str(kwargs.get('session_id', ''))
        # Native separate-verifier defaults include a host reward bind. Keep
        # rewards in the private namespace and use the audited export path.
        mounts = kwargs.pop('mounts', []) or []
        if any(str(m.get('target') if isinstance(m,dict) else m.target) not in
               {'/logs/verifier','/logs/agent','/logs/artifacts'} for m in mounts):
            raise ValueError('Unexpected native host mount request')
        kwargs['mounts'] = []
        self.relay = None
        self.broker = None
        self.queue = None
        self.verifier_queue = None
        self.verifier_broker = None
        self.verifier_relay = None
        super().__init__(*args, **kwargs)
        self.audit = PRIVATE / 'harbor-runtime' / self.session_id
        self.audit.mkdir(parents=True, exist_ok=True)
        overlay = self.audit / 'queue-compose.yaml'
        self.queue = 'heima-queue-' + self.context_id.hex if self.context_id else 'heima-queue-' + self.session_id
        self.verifier_queue = self.queue + '-verifier'
        document = {'services': {'main': {'environment': {'LATENT_LANE': role},
                     'volumes': [{'type': 'volume', 'source': self.queue, 'target': '/run/latentcli/queue'},
                                 {'type': 'volume', 'source': self.verifier_queue, 'target': '/run/heima-verifier/queue'}]}},
                    'volumes': {self.queue: {'external': True}, self.verifier_queue: {'external': True}}}
        overlay.write_text(yaml.safe_dump(document))
        self.extra_docker_compose_paths.append(overlay)

    @staticmethod
    def _requires_egress_control(**kwargs):
        # Physical network_mode:none provides the exact requested NONE policy.
        # No network namespace sidecar or host security alteration is needed.
        return False

    @property
    def capabilities(self):
        return super().capabilities.model_copy(update={'disable_internet': True, 'mounted': False,
                                                       'dynamic_network_policy': False})

    def validate_network_policy_support(self, network_policy=None):
        value = network_policy or self.network_policy
        if value.network_mode != NetworkMode.NO_NETWORK:
            raise ValueError('This offline task provider only supports network_mode=none')

    def _write_mounts_compose_file(self):
        # Logs live in private tmpfs and are downloaded using native Harbor.
        # Never expose a host private directory through generated log mounts.
        return None

    def _mount_targets(self, *, writable_only=False):
        # There are no native host bind targets in this provider. Harbor's
        # default chmod777 hints must not expose the private verifier tmpfs.
        return []

    async def ensure_dirs(self, dirs, *, chmod=True):
        if not dirs: return None
        command = self._ensure_dirs_command(dirs, chmod=False)
        private = [str(p) for p in dirs if str(p).startswith(('/logs/verifier', '/tests', '/solution', '/run/heima-verifier'))]
        if private:
            command += ' && chmod 700 ' + ' '.join(shlex.quote(p) for p in private)
        return await self.exec(command, user='root')

    async def empty_dirs(self, dirs, *, chmod=True):
        if not dirs: return None
        command = self._empty_dirs_command(dirs, chmod=False)
        private = [str(p) for p in dirs if str(p).startswith(('/logs/verifier', '/tests', '/solution', '/run/heima-verifier'))]
        if private:
            command += ' && chmod 700 ' + ' '.join(shlex.quote(p) for p in private)
        return await self.exec(command, user='root')

    async def prepare_logs_for_host(self):
        return

    async def restore_candidate(self, files):
        writer = "import json,sys;from pathlib import Path;files=json.load(sys.stdin);base=Path('/workspace/solution');\nfor name,value in files.items():\n p=Path(name);assert not p.is_absolute() and '..' not in p.parts;target=base/p;target.parent.mkdir(parents=True,exist_ok=True);target.write_text(value)\n"
        result = await self._run_docker_compose_command(['exec', '-T', '-u', '1000:1000', 'main', 'python', '-c', writer],
                                                       stdin_data=json.dumps(files).encode(), timeout_sec=30, check=False)
        if result.return_code: raise RuntimeError('Own candidate restore failed')

    async def export_candidate(self):
        # Serialize data in the candidate namespace. Never copy untrusted
        # symlinks or special files into the host evidence tree.
        exporter = "import json,sys;sys.path.insert(0,'/opt/latentcli');from snapshot_files import snapshot;print(json.dumps(snapshot('/workspace/solution'),ensure_ascii=False))"
        result = await self.exec(command='python -c ' + shlex.quote(exporter), user='1000:1000', timeout_sec=30)
        if result.return_code: raise RuntimeError('Bounded candidate export failed')
        return json.loads(result.stdout)

    async def _upload_data(self, files, target):
        # Engine cp rejects writes on a readonly rootfs even when the target is
        # tmpfs. Serialize regular files into their owner namespace instead.
        target = str(target)
        user = '1000:1000' if target.startswith(('/logs/agent', '/workspace/solution', '/home/candidate')) else 'root'
        writer = '''import base64,json,sys
from pathlib import Path
frame=json.load(sys.stdin);base=Path(frame['target']);base.mkdir(parents=True,exist_ok=True)
for name,encoded in frame['files'].items():
 relative=Path(name);assert not relative.is_absolute() and '..' not in relative.parts
 path=base/relative;path.parent.mkdir(parents=True,exist_ok=True)
 path.write_bytes(base64.b64decode(encoded,validate=True))
'''
        payload=json.dumps({'target':target,'files':files}).encode()
        if len(payload)>32*1024*1024:raise ValueError('Native file upload budget exceeded')
        result=await self._run_docker_compose_command(['exec','-T','-u',user,'-w','/','-e','HOME=/root' if user=='root' else 'HOME=/home/candidate',
              '-e','PYTHONPATH=','main','python','-I','-c',writer],stdin_data=payload,timeout_sec=60,check=False)
        if result.return_code:raise RuntimeError('Native bounded file upload failed: '+(result.stdout or '')[-1000:])

    async def upload_dir(self, source_dir, target_dir):
        source=Path(source_dir)
        files={str(p.relative_to(source)):base64.b64encode(p.read_bytes()).decode()
               for p in source.rglob('*') if p.is_file() and not p.is_symlink()}
        await self._upload_data(files,target_dir)

    async def upload_file(self, source_path, target_path):
        target=Path(target_path)
        await self._upload_data({target.name:base64.b64encode(Path(source_path).read_bytes()).decode()},target.parent)

    async def _download_regular_files(self, source, target, single=False):
        # Docker archive copy did not export files from private tmpfs mounts.
        # Read them through their owner namespace without following symlinks.
        reader = '''import base64,json,os,stat,sys
from pathlib import Path
source=Path(sys.argv[1]);single=sys.argv[2]=='1';files={};total=0
base=source.parent if single else source
paths=[source] if single else []
if not single:
 for directory,dirs,names in os.walk(base,followlinks=False):
  dirs[:]=[d for d in dirs if not (Path(directory)/d).is_symlink()]
  paths.extend(Path(directory)/name for name in names)
for path in paths:
 try:
  fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
  try:
   info=os.fstat(fd)
   if not stat.S_ISREG(info.st_mode):continue
   if info.st_size>16*1024*1024:raise ValueError('Native export file too large')
   raw=os.read(fd,16*1024*1024+1)
  finally:os.close(fd)
 except OSError:continue
 total+=len(raw)
 if total>32*1024*1024:raise ValueError('Native export total too large')
 files[str(path.relative_to(base))]=base64.b64encode(raw).decode()
print(json.dumps(files))
'''
        user = '1000:1000' if str(source).startswith(('/logs/agent', '/workspace/solution', '/home/candidate')) else 'root'
        result = await self._run_docker_compose_command(
            ['exec','-T','-u',user,'-w','/','main','python','-I','-c',reader,str(source),'1' if single else '0'],
            timeout_sec=60,check=False)
        if result.return_code: raise RuntimeError('Native namespace export failed')
        files=json.loads(result.stdout)
        target=Path(target);base=target.parent if single else target
        base.mkdir(parents=True,exist_ok=True)
        for name,encoded in files.items():
            relative=Path(name)
            if relative.is_absolute() or '..' in relative.parts:raise ValueError('Invalid native export path')
            path=target if single else base/relative
            if any(part.is_symlink() for part in [path,*path.parents]):raise ValueError('Host export symlink refused')
            path.parent.mkdir(parents=True,exist_ok=True)
            path.write_bytes(base64.b64decode(encoded,validate=True))
        with (self.audit/'namespace-exports.jsonl').open('a') as stream:
            stream.write(json.dumps({'source':str(source),'files':list(files),'user':user})+'\n')

    async def download_dir(self, source_dir, target_dir):
        if str(source_dir) == '/workspace/solution':
            # Native artifact handoff transfers submitted executable source,
            # not large CPU research traces or a candidate-written reward.
            files = await self.export_candidate()
            selected = {name:value for name,value in files.items() if Path(name).suffix == '.py' or name=='method_notes.json'}
            if sum(Path(name).suffix=='.py' for name in selected)>64 or sum(len(v.encode()) for v in selected.values())>2*1024*1024:
                raise ValueError('Submitted source exceeds evaluator bundle limits')
            base=Path(target_dir);base.mkdir(parents=True,exist_ok=True)
            for name,value in selected.items():
                path=base/name
                if any(p.is_symlink() for p in [path,*path.parents]):raise ValueError('Host handoff symlink refused')
                path.parent.mkdir(parents=True,exist_ok=True);path.write_text(value)
            (self.audit/'submission-handoff.json').write_text(json.dumps({'source':str(source_dir),'files':list(selected),
                 'candidate_code_imported_into_host':False},indent=2)+'\n')
            return
        await self._download_regular_files(source_dir,target_dir)

    async def download_file(self, source_path, target_path):
        await self._download_regular_files(source_path,target_path,single=True)

    async def start(self, force_build):
        await asyncio.to_thread(docker, 'volume', 'create', '--driver', 'local', '--opt', 'type=tmpfs',
                                '--opt', 'device=tmpfs', '--opt', 'o=size=128m,uid=1000,gid=1000,mode=0700', self.queue)
        await asyncio.to_thread(docker, 'volume', 'create', '--driver', 'local', '--opt', 'type=tmpfs',
                                '--opt', 'device=tmpfs', '--opt', 'o=size=128m,uid=0,gid=0,mode=0700', self.verifier_queue)
        self.broker = 'heima-broker-' + self.queue.removeprefix('heima-queue-')
        await asyncio.to_thread(docker, 'run', '-d', '--name', self.broker, '--network', 'none', '--read-only',
                                '--cap-drop', 'ALL', '--security-opt', 'no-new-privileges', '--user', '1000:1000',
                                '--memory', '256m', '--mount', f'type=volume,source={self.queue},target=/q/{self.role}',
                                BASE, 'sleep', 'infinity')
        self.verifier_broker = self.broker + '-verifier'
        await asyncio.to_thread(docker, 'run', '-d', '--name', self.verifier_broker, '--network', 'none', '--read-only',
                                '--cap-drop', 'ALL', '--security-opt', 'no-new-privileges', '--user', '0:0',
                                '--memory', '256m', '--mount', f'type=volume,source={self.verifier_queue},target=/q/{self.role}',
                                BASE, 'sleep', 'infinity')
        await super().start(force_build)
        cid = (await self._run_docker_compose_command(['ps', '-q', 'main'], check=True)).stdout.strip()
        inspected = json.loads(await asyncio.to_thread(docker, 'inspect', cid))[0]
        host = inspected['HostConfig']
        if host['NetworkMode'] != 'none' or not host['ReadonlyRootfs'] or host.get('Privileged') or host.get('Devices') or host.get('DeviceRequests') or host['CapDrop'] != ['ALL']:
            raise RuntimeError('Native task physical confinement differs')
        if any(m['Type'] == 'bind' for m in inspected['Mounts']):
            raise RuntimeError('Unexpected native task host bind mount')
        (self.audit / 'container-inspect.json').write_text(json.dumps(inspected, indent=2) + '\n')
        probe = await self.exec(command="codex sandbox -c 'sandbox_mode=\"workspace-write\"' python3 -c \"from pathlib import Path;Path('/workspace/solution/native-probe.txt').write_text('verified');print('native tool write verified')\"", user='1000:1000', timeout_sec=30)
        (self.audit / 'native-probe.json').write_text(probe.model_dump_json(indent=2))
        if probe.return_code:
            raise RuntimeError('Native Codex sandbox probe failed; provider keys not retrieved')
        if not self.is_verifier:
            reset = await self.exec(command='cp /opt/latentcli/starter.py /workspace/solution/method.py && cp /opt/latentcli/starter_notes.json /workspace/solution/method_notes.json', user='1000:1000', timeout_sec=30)
            if reset.return_code:
                raise RuntimeError('Native Starter installation failed')
            public_probe = await self.exec(command='python -c ' + shlex.quote("import hashlib,json\nfrom pathlib import Path\nbase=Path('/workspace/public_assets');manifest=json.loads((base/'manifest.json').read_text());raw=(base/'dev/samples.json').read_bytes();rows=json.loads(raw)\nassert len(rows)==96 and hashlib.sha256(raw).hexdigest()==manifest['samples_sha256']\nfor row in rows:\n assert hashlib.sha256((base/'dev'/row['image']).read_bytes()).hexdigest()==row['image_sha256']\ntry:\n (base/'dev/forbidden-write.txt').write_text('probe')\n raise RuntimeError('Public assets unexpectedly writable')\nexcept PermissionError:pass\nexcept OSError as error:\n assert error.errno==30\nprint(json.dumps({'samples':96,'samples_sha256':manifest['samples_sha256'],'all_image_hashes_pass':True,'readonly':True}))\n"), user='1000:1000', timeout_sec=30)
            (self.audit / 'public-dev-probe.json').write_text(public_probe.model_dump_json(indent=2))
            if public_probe.return_code:
                raise RuntimeError('Actual public Dev visibility or readonly probe failed')
            self.relay = ProviderRelay(DOCKER, self.broker, self.audit / 'routes',
                                  (Path(__file__).parent / 'queue_broker.py').read_text(),
                                  evaluator=EvaluatorBridge(self.audit / 'evaluation'))
        self.verifier_relay = ProviderRelay(DOCKER, self.verifier_broker, self.audit / 'verifier-routes',
                                           (Path(__file__).parent / 'queue_broker.py').read_text(),
                                           evaluator=EvaluatorBridge(self.audit / 'verifier-evaluation', verifier=True))

    async def exec(self, command, *args, **kwargs):
        if self.relay is not None and '/tests/test.sh' in command:
            # Verification no longer needs the candidate's request transport.
            # Closing it also prevents background candidate jobs monopolizing
            # its evaluation lane while the private grader runs.
            await asyncio.to_thread(self.relay.close)
            self.relay = None
        result = await super().exec(command, *args, **kwargs)
        if '/tests/test.sh' in command:
            with (self.audit / 'verifier-commands.jsonl').open('a') as stream:
                stream.write(json.dumps({'command': command, 'result': result.model_dump()}) + '\n')
        return result

    async def _compose_exec(self, command, *, service, cwd, env, timeout_sec, user):
        if str(user).split(':')[0] in {'root', '0'}:
            # Root verification must never source an agent-owned login profile
            # or Python search path through the container's candidate HOME.
            env = {**(env or {}), 'HOME': '/root', 'BASH_ENV': '/dev/null', 'ENV': '/dev/null',
                   'PYTHONPATH': '', 'PYTHONHOME': ''}
            cwd = '/'
        return await super()._compose_exec(command, service=service, cwd=cwd, env=env,
                                           timeout_sec=timeout_sec, user=user)

    async def stop(self, delete):
        try:
            await super().stop(delete)
        finally:
            if self.relay is not None:
                await asyncio.to_thread(self.relay.close)
            if self.verifier_relay is not None:
                await asyncio.to_thread(self.verifier_relay.close)
            if self.broker:
                await asyncio.to_thread(docker, 'rm', '-f', self.broker)
            if self.queue:
                await asyncio.to_thread(docker, 'volume', 'rm', self.queue)
            if self.verifier_broker:
                await asyncio.to_thread(docker, 'rm', '-f', self.verifier_broker)
            if self.verifier_queue:
                await asyncio.to_thread(docker, 'volume', 'rm', self.verifier_queue)
