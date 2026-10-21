"""Copy a native author's frozen selected bytes for an independent verifier.

This agent performs no research, model request or candidate Python execution.
"""
import hashlib
import json
from pathlib import Path
from harbor.agents.base import BaseAgent
from release_source_binding import current_binding

PRIVATE = Path(__file__).resolve().parent


class FrozenArtifactAgent(BaseAgent):
    def __init__(self, *args, lane, source_snapshot, expected_files,
                 expected_artifact, frozen_binding, evidence_root, task_path, **kwargs):
        super().__init__(*args, **kwargs)
        if lane not in ['gpt','seed']:
            raise ValueError('Unknown completed native author lane')
        self.lane = lane
        self.snapshot = Path(source_snapshot).resolve()
        self.task_path=Path(task_path).resolve()
        if self.snapshot.is_symlink() or not self.snapshot.is_dir():
            raise ValueError('Selected source must be a real frozen directory')
        self.files = expected_files
        if set(self.files) != {'method.py', 'program.json'}:
            raise ValueError('Only the two declared method artifacts are permitted')
        self.artifact = expected_artifact
        self.binding = frozen_binding
        self.root = Path(evidence_root).resolve()
        if self.root.is_relative_to(self.task_path) or self.root.is_relative_to(PRIVATE):
            raise ValueError('Evidence must remain outside Task and public Runtime')
        self.root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def name():
        return 'frozen-native-selected-method-copy'

    def version(self):
        return '1.0.0'

    def check_source(self):
        if current_binding(self.task_path) != self.binding:
            raise RuntimeError('Frozen task or ordinary runtime changed')
        for name, expected in self.files.items():
            path = self.snapshot / name
            if path.is_symlink() or not path.is_file():
                raise ValueError('Selected artifact must be an original regular file')
            data = path.read_bytes()
            if len(data) > 524288 or hashlib.sha256(data).hexdigest() != expected:
                raise ValueError('Selected source bytes changed: ' + name)
        graph = json.loads((self.snapshot / 'program.json').read_text())
        canonical = hashlib.sha256(json.dumps(graph, sort_keys=True,
                                             separators=(',', ':')).encode()).hexdigest()
        if canonical != self.artifact:
            raise ValueError('Selected canonical graph differs')

    async def setup(self, environment):
        self.check_source()
        if environment.role != self.lane or environment.author_oracle:
            raise RuntimeError('Validation must use the ordinary own physical role')
        for name in self.files:
            await environment.upload_file(self.snapshot / name, '/workspace/solution/' + name)
        actual = await environment.exec(
            'sha256sum /workspace/solution/method.py /workspace/solution/program.json',
            user='1000:1000', timeout_sec=20)
        if actual.return_code:
            raise RuntimeError('Fresh ordinary-container artifact readback failed')
        readback = {Path(line.split()[1]).name: line.split()[0]
                    for line in (actual.stdout or '').splitlines()}
        if readback != self.files:
            raise RuntimeError('Fresh ordinary-container source readback differs')
        proof = {
            'status': 'FROZEN_NATIVE_SELECTED_BYTES_COPIED_AND_FRESH_CONTAINER_SHA_VERIFIED',
            'lane': self.lane, 'artifact_sha256': self.artifact,
            'source_snapshot': str(self.snapshot), 'source_files': self.files,
            'actual_container_files': readback,
            'environment_binding': json.loads(
                (environment.own_root / 'environment-binding.json').read_text()),
            'model_called': False, 'candidate_Python_executed': False,
            'research_credit': 0, 'is_fresh_native_CLI_resume': False,
        }
        (self.root / 'copy-proof.json').write_text(json.dumps(proof, indent=2) + '\n')

    async def run(self, instruction, environment, context):
        self.check_source()
