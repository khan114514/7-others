"""Real specified model CLI inside the actual native Harbor task provider."""
import json
import shlex
import time
from pathlib import Path
from harbor.agents.base import BaseAgent
from cli_provider_relay import MODELS
from runtime_snapshot import snapshot

from host_paths import PRIVATE_ROOT as PRIVATE


class QueuedCodexAgent(BaseAgent):
    @staticmethod
    def name(): return 'heima-real-codex'

    def version(self): return '0.160.0'

    def __init__(self, *args, role, resume_folder, prompt_file, evidence_folder, bounded_seconds=1800, **kwargs):
        super().__init__(*args, **kwargs)
        if role not in MODELS: raise ValueError('Unknown fixed model lane')
        self.role = role
        self.resume = Path(resume_folder).resolve()
        self.resume.relative_to(PRIVATE / 'cli-research-preparation')
        self.prompt = Path(prompt_file).read_text()
        self.timeout = bounded_seconds
        self.evidence = Path(evidence_folder).resolve()
        self.evidence.relative_to(PRIVATE / 'native-actor-evidence')
        self.evidence.mkdir(parents=True)

    async def setup(self, environment):
        base = self.resume / self.role / 'candidate'
        files = snapshot(base)
        await environment.restore_candidate(files)

    async def run(self, instruction, environment, context):
        prompt = 'Read /workspace/instruction.md and your own previous research/evaluation records using actual tools.\n' + self.prompt
        writer = "from pathlib import Path;Path('/workspace/solution/current_prompt.txt').write_text(" + repr(prompt) + ")"
        result = await environment.exec(command='python -c ' + shlex.quote(writer), user='1000:1000', timeout_sec=30)
        if result.return_code: raise RuntimeError('CLI prompt setup failed')
        raw_path = self.evidence / 'stdout.jsonl'
        event_path = self.evidence / 'observed-events.jsonl'
        timing = {'model_requested': MODELS[self.role], 'role': self.role, 'reasoning_effort': 'high',
                  'started_epoch': time.time(), 'native_harbor_actual_cli': True}
        with raw_path.open('w') as raw, event_path.open('w') as audit:
            async def observe(line, stream):
                raw.write(line if line.endswith('\n') else line+'\n'); raw.flush()
                try:
                    event = json.loads(line)
                except ValueError:
                    return
                audit.write(json.dumps({'observed_epoch': time.time(), 'event': event}) + '\n');audit.flush()
            try:
                with environment.scoped_output_callback(observe):
                    result = await environment.exec(command=f'python /opt/latentcli/run_codex.py /workspace/solution/current_prompt.txt {self.timeout}',
                                                    user='1000:1000', timeout_sec=self.timeout + 20)
                timing['cli_exit_code'] = result.return_code
            finally:
                timing['completed_epoch'] = time.time()
                (self.evidence / 'cli-timing.json').write_text(json.dumps(timing, indent=2) + '\n')
                # Preserve own files even when the model turn is interrupted.
                files = await environment.export_candidate()
                destination = self.evidence / 'candidate'
                destination.mkdir()
                for name, value in files.items():
                    relative = Path(name)
                    if relative.is_absolute() or '..' in relative.parts or not isinstance(value, str):
                        raise ValueError('Invalid exported candidate data')
                    path = destination / relative
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(value)
