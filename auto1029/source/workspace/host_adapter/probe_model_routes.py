"""Actual configured CLI/tool route proof; deliberately zero research credit."""
import argparse
import asyncio
import json
import shlex
import time
import uuid
from pathlib import Path

from harbor.models.trial.config import TrialConfig
from harbor.trial.trial import Trial
from cli_provider_relay import MODELS
from host_paths import PRIVATE_ROOT

PROMPT = '''This is a deployment integration probe, not an optimization research turn.
Use an actual shell tool to write configured-route-probe.txt in your current solution directory, containing ROUTE_TOOL_PROBE_ONLY. Read it using an actual tool and report the marker. Do not modify method.py or read other files. Do not invoke evaluate.py. End after this single integration check.'''


async def one(task, role):
    identifier = uuid.uuid4().hex
    folder = PRIVATE_ROOT / 'model-route-probes' / identifier / role
    folder.mkdir(parents=True)
    trial = await Trial.create(TrialConfig(
        task={'path': str(task)}, trial_name='heima-portable-route-' + role + '-' + identifier[:12], trials_dir=folder,
        agent={'name': 'nop'}, environment={'import_path': 'heima_harbor_environment:HeimaDockerEnvironment',
                                          'kwargs': {'role': role}}))
    env = trial.agent_environment
    state = {'phase': 'ACTUAL_CONFIGURED_CLI_ROUTE_PROBE_STARTING', 'role': role,
             'model_requested': MODELS[role], 'reasoning_effort': 'high', 'model_research_seconds': 0,
             'formal_scored_loop': False, 'gpu_submitted': False, 'started_epoch': time.time()}
    events = []
    try:
        await env.start(False)
        writer = "from pathlib import Path;Path('/workspace/solution/route_prompt.txt').write_text(" + repr(PROMPT) + ")"
        setup = await env.exec(command='python -c ' + shlex.quote(writer), user='1000:1000', timeout_sec=30)
        assert setup.return_code == 0
        with (folder / 'stdout.jsonl').open('x') as raw:
            async def observe(line, stream):
                raw.write(line if line.endswith('\n') else line + '\n')
                raw.flush()
                try:
                    events.append(json.loads(line))
                except ValueError:
                    pass
            with env.scoped_output_callback(observe):
                execution = await env.exec(command='python /opt/latentcli/run_codex.py /workspace/solution/route_prompt.txt 180',
                                           user='1000:1000', timeout_sec=200)
        (folder / 'execution.json').write_text(execution.model_dump_json(indent=2))
        assert execution.return_code == 0
        assert any(e.get('type') == 'turn.completed' for e in events)
        commands = [e['item'] for e in events if e.get('type') == 'item.completed'
                    and e.get('item', {}).get('type') == 'command_execution']
        assert commands and any(e.get('exit_code') == 0 for e in commands)
        probe = await env.exec(command='cat /workspace/solution/configured-route-probe.txt', user='1000:1000', timeout_sec=30)
        assert probe.return_code == 0 and probe.stdout.strip() == 'ROUTE_TOOL_PROBE_ONLY'
        # Finished routes are closed and flushed before auditing identities.
        await asyncio.to_thread(env.relay.close)
        env.relay = None
        routes = [json.loads(p.read_text()) for p in (env.audit / 'routes').glob(role + '-*-route.json')]
        finished = [r for r in routes if r.get('model_response_completed')]
        assert finished and all(r.get('served_identifier_matches_requested') is True for r in finished)
        assert all(r.get('model_identifiers_in_provider_response') == [MODELS[role]] for r in finished)
        state.update(phase='ACTUAL_CONFIGURED_CLI_MODEL_AND_TOOL_ROUTE_PASS', completed_epoch=time.time(),
                     cli_exit_code=execution.return_code, tool_commands_succeeded=sum(e.get('exit_code') == 0 for e in commands),
                     tool_commands_failed=sum(e.get('exit_code') != 0 for e in commands),
                     finished_model_responses=len(finished), provider_identifier_matches=True,
                     private_route_evidence_directory=str((env.audit / 'routes').relative_to(PRIVATE_ROOT)))
    except Exception as error:
        state.update(phase='FAILED_PRESERVE_EVIDENCE', error_type=type(error).__name__,
                     error=str(error)[:800], completed_epoch=time.time())
        raise
    finally:
        (folder / 'result.json').write_text(json.dumps(state, indent=2) + '\n')
        await env.stop(True)
    return {'role': role, 'phase': state['phase'], 'folder': str(folder), 'model_research_seconds': 0}


async def main(task, roles):
    results = await asyncio.gather(*(one(task, role) for role in roles), return_exceptions=True)
    for result in results:
        if isinstance(result, Exception):
            print(json.dumps({'phase': 'FAILED_PRESERVE_EVIDENCE', 'error_type': type(result).__name__}), flush=True)
        else:
            print(json.dumps(result), flush=True)
    if any(isinstance(r, Exception) for r in results):
        raise SystemExit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', required=True, type=Path)
    parser.add_argument('--roles', nargs='+', choices=['gpt', 'seed'], default=['gpt', 'seed'])
    args = parser.parse_args()
    if len(set(args.roles)) != len(args.roles):
        parser.error('Duplicate role refused')
    asyncio.run(main(args.task.resolve(), args.roles))
