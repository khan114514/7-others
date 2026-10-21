"""Frozen-source post-selection verification; never a model research track."""
import asyncio
import hashlib
import json
from pathlib import Path
import shlex
import time

from harbor.agents.base import BaseAgent
from forecast_harbor_environment import ForecastDockerEnvironment
from evaluator_bridge import docker

PRIVATE=Path(__file__).resolve().parent


class FrozenCandidateAgent(BaseAgent):
    @staticmethod
    def name():return 'frozen-source-handoff-no-model'
    def version(self):return '1'
    def __init__(self,*args,role,**kwargs):
        super().__init__(*args,**kwargs)
        self.role=role
    async def setup(self,environment):pass
    async def run(self,instruction,environment,context):
        base=PRIVATE/'campaign'/self.role/'best_method'
        files={str(p.relative_to(base)):p.read_text() for p in base.rglob('*.py')}
        code="import json;from pathlib import Path;files=json.loads("+repr(json.dumps(files))+ ");base=Path('/workspace/solution');\nfor p in base.rglob('*.py'):p.unlink()\nfor name,value in files.items():\n p=base/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(value)\n"
        response=await environment.exec('python -I -c '+shlex.quote(code),user='1000:1000',timeout_sec=30)
        if response.return_code:raise RuntimeError('Frozen candidate handoff failed')


class FrozenEvaluationEnvironment(ForecastDockerEnvironment):
    def __init__(self,*args,hidden_evaluation=False,**kwargs):
        self.hidden_evaluation=hidden_evaluation
        kwargs['enable_models']=False
        super().__init__(*args,**kwargs)
    async def start(self,force_build):
        await super().start(force_build)
        verifier='__verifier__' in self.session_id
        if not self.hidden_evaluation or not verifier:return
        identifier=(await self._run_docker_compose_command(['ps','-q','main'],check=True)).stdout.strip()
        mkdir=await self.exec('mkdir -p /tests/hidden_assets && chmod 700 /tests/hidden_assets',user='root',timeout_sec=30)
        if mkdir.return_code:raise RuntimeError('Private verifier injection directory failed')
        inputs={'hidden_features.npz':'dev_features.npz','hidden_targets.npz':'dev_targets.npz','protocol.json':'protocol.json'}
        kit=PRIVATE/'post-selection-hidden-kit'
        hashes={}
        for source,target in inputs.items():
            path=kit/source
            await asyncio.to_thread(docker,'cp',str(path),identifier+':/tests/hidden_assets/'+target)
            hashes[target]=hashlib.sha256(path.read_bytes()).hexdigest()
        # docker cp preserves host uid501/gid20. Root has CAP_CHOWN but no
        # CAP_FOWNER/DAC override, so explicitly acquire ownership first.
        protected=await self.exec('chown 0:0 /tests/hidden_assets/* && chmod 600 /tests/hidden_assets/*',user='root',timeout_sec=30)
        (self.audit/'hidden-injection-permissions.json').write_text(protected.model_dump_json(indent=2))
        if protected.return_code:raise RuntimeError('Injected private verifier permissions failed')
        (self.audit/'hidden-injection-receipt.json').write_text(json.dumps({'injected_epoch':time.time(),
            'session_id':self.session_id,'verifier_only':True,'post_build_copy':True,
            'model_requests':False,'credited_model_seconds':0,'files_sha256':hashes},indent=2)+'\n')
