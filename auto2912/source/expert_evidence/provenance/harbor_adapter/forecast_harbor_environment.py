"""Native Harbor Docker environment with independent keyless model lanes."""
import asyncio
import json
from pathlib import Path
import shlex
import uuid
import weakref

from harbor.environments.docker.docker import DockerEnvironment
from harbor.models.task.config import NetworkMode
import yaml
from cli_provider_relay import ProviderRelay
from evaluator_bridge import EvaluatorBridge,docker,DOCKER

PRIVATE=Path(__file__).resolve().parent
BASE='python:3.12-slim@sha256:02108f5d322dd89f1c9e552442c25acb0543dfdbc455693a5599624f20d9155d'


class ForecastDockerEnvironment(DockerEnvironment):
    _forecast_loop_build_locks=weakref.WeakKeyDictionary()
    def __init__(self,*args,role='gpt',enable_models=False,**kwargs):
        if role not in ['gpt','seed']:raise ValueError('Unknown physical lane')
        self.role=role
        self.enable_models=enable_models and '__verifier__' not in str(kwargs.get('session_id',''))
        kwargs['mounts']=[]
        self.relay=None
        self.broker=None
        self.queue=None
        super().__init__(*args,**kwargs)
        # Harbor 0.23's inherited class cache retains asyncio.Lock objects
        # across asyncio.run calls. Share builds only within the active loop.
        loop=asyncio.get_running_loop()
        self._image_build_locks=self._forecast_loop_build_locks.setdefault(loop,{})
        self.audit=PRIVATE/'harbor-runtime'/self.session_id
        self.audit.mkdir(parents=True,exist_ok=True)
        if self.enable_models:
            self.queue='forecast-queue-'+(self.context_id.hex if self.context_id else uuid.uuid4().hex)
            overlay=self.audit/'lane-compose.yaml'
            overlay.write_text(yaml.safe_dump({'services':{'main':{'environment':{'FORECAST_LANE':role},
                'volumes':[{'type':'volume','source':self.queue,'target':'/run/forecastcli/queue'}]}},
                'volumes':{self.queue:{'external':True}}}))
            self.extra_docker_compose_paths.append(overlay)

    @staticmethod
    def _requires_egress_control(**kwargs):return False

    @property
    def capabilities(self):
        return super().capabilities.model_copy(update={'disable_internet':True,'mounted':False,'dynamic_network_policy':False})

    def validate_network_policy_support(self,network_policy=None):
        if (network_policy or self.network_policy).network_mode!=NetworkMode.NO_NETWORK:
            raise ValueError('Only physically offline environments are supported')

    def _write_mounts_compose_file(self):return None

    def _mount_targets(self,*,writable_only=False):return []

    async def ensure_dirs(self,dirs,*,chmod=True):
        if dirs:return await self.exec(self._ensure_dirs_command(dirs,chmod=False),user='root')

    async def empty_dirs(self,dirs,*,chmod=True):
        if dirs:return await self.exec(self._empty_dirs_command(dirs,chmod=False),user='root')

    async def prepare_logs_for_host(self):return None

    async def start(self,force_build):
        if self.queue:
            await asyncio.to_thread(docker,'volume','create','--driver','local','--opt','type=tmpfs',
                '--opt','device=tmpfs','--opt','o=size=128m,uid=1000,gid=1000,mode=0700',self.queue)
            self.broker='forecast-broker-'+self.queue.removeprefix('forecast-queue-')
            await asyncio.to_thread(docker,'run','-d','--name',self.broker,'--network','none','--read-only',
                '--cap-drop','ALL','--security-opt','no-new-privileges','--user','1000:1000',
                '--memory','256m','--mount',f'type=volume,source={self.queue},target=/q/{self.role}',BASE,'sleep','infinity')
        await super().start(force_build)
        ready=await self.exec(command='python -I -c '+shlex.quote("import time;from pathlib import Path;deadline=time.monotonic()+20\nwhile not Path('/run/forecastcli/ready.txt').exists():\n if time.monotonic()>deadline:raise RuntimeError('Trusted task startup deadline exceeded')\n time.sleep(.05)\nprint('trusted scorer and native CLI configuration ready')"),user='1000:1000',timeout_sec=25)
        if ready.return_code:raise RuntimeError('Trusted task startup was not ready')
        response=await self._run_docker_compose_command(['ps','-q','main'],check=True)
        identifier=response.stdout.strip()
        inspected=json.loads((await asyncio.to_thread(docker,'inspect',identifier)).stdout)[0]
        configuration=inspected['HostConfig']
        if configuration['NetworkMode']!='none' or configuration.get('Privileged') or configuration.get('DeviceRequests'):
            raise RuntimeError('Physical actor confinement differs from task contract')
        if any(mount['Type']=='bind' for mount in inspected['Mounts']):
            raise RuntimeError('No host bind is permitted in the actor environment')
        (self.audit/'container-inspect.json').write_text(json.dumps(inspected,indent=2)+'\n')
        probe=await self.exec(command='python -I -c '+shlex.quote("import os;from pathlib import Path;assert os.geteuid()==1000;assert int([s.split()[1] for s in Path('/proc/self/status').read_text().splitlines() if s.startswith('CapEff:')][0],16)==0;\ntry:Path('/opt/verifier-assets/dev_targets.npz').read_bytes();raise RuntimeError('Development targets readable')\nexcept PermissionError:pass\nprint('unprivileged actor; protected development targets inaccessible')"),user='1000:1000',timeout_sec=30)
        (self.audit/'isolation-probe.json').write_text(probe.model_dump_json(indent=2))
        if probe.return_code:raise RuntimeError('Development-target isolation or actor capability probe failed')
        if self.enable_models:
            native=await self.exec(command="codex sandbox -c 'sandbox_mode=\"workspace-write\"' python -c \"from pathlib import Path;Path('/workspace/solution/native-probe.txt').write_text('verified');print('native sandbox verified')\"",user='1000:1000',timeout_sec=30)
            (self.audit/'native-sandbox-probe.json').write_text(native.model_dump_json(indent=2))
            if native.return_code:raise RuntimeError('Native CLI sandbox unavailable; no provider key requested')
            self.relay=ProviderRelay(DOCKER,self.broker,self.audit/'routes',(PRIVATE/'queue_broker.py').read_text())

    async def _compose_exec(self,command,*,service,cwd,env,timeout_sec,user):
        if str(user).split(':')[0] in ['root','0']:
            env={**(env or {}),'HOME':'/root','BASH_ENV':'/dev/null','ENV':'/dev/null','PYTHONPATH':'','PYTHONHOME':''}
            cwd='/'
        return await super()._compose_exec(command,service=service,cwd=cwd,env=env,timeout_sec=timeout_sec,user=user)

    async def export_candidate(self):
        command='python -I -c '+shlex.quote("import json,sys;sys.path.insert(0,'/opt/forecastcli');from snapshot_files import snapshot;print(json.dumps(snapshot('/workspace/solution')))" )
        result=await self.exec(command,user='1000:1000',timeout_sec=30)
        if result.return_code:raise RuntimeError('Own bounded candidate export failed')
        return json.loads(result.stdout)

    async def download_dir(self,source_dir,target_dir):
        if str(source_dir).rstrip('/')=='/workspace/solution':
            files=await self.export_candidate()
            base=Path(target_dir)
            base.mkdir(parents=True,exist_ok=True)
            for name,value in files.items():
                relative=Path(name)
                if relative.is_absolute() or '..' in relative.parts:raise ValueError('Bounded source handoff refused')
                path=base/relative;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(value)
            return None
        return await super().download_dir(source_dir,target_dir)

    async def stop(self,delete):
        try:await super().stop(delete)
        finally:
            if self.relay:await asyncio.to_thread(self.relay.close)
            if self.broker:await asyncio.to_thread(docker,'rm','-f',self.broker,check=False)
            if self.queue:await asyncio.to_thread(docker,'volume','rm',self.queue,check=False)
