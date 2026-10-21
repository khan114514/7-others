"""Native Chat Completions agent harness (direct, no Responses wire).

This is a purpose-built, truthfully labeled QA client ("native-ccagent"). It
speaks OpenAI Chat Completions with native function/tool calling against a
loopback, keyless proxy; the host relay holds the real credential. The model
directs every action and authors all content. This harness never writes a
review, method, or QA conclusion itself.
"""
import argparse
import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

WORKDIR = Path('/workspace/solution')
OUTPUT = Path('/output')
READ_ROOTS = [WORKDIR, OUTPUT, Path('/opt/bifrost'), Path('/home/candidate'),Path('/tmp')]
WRITE_ROOTS = [WORKDIR,Path('/tmp')]
SHELL_OUTPUT_CAP = 64 * 1024
READ_CAP = 256 * 1024

SYSTEM = ('You are the original Seed researcher continuing your own authenticated history and workspace through native-ccagent-1.0 ChatCompletions tools CLI, not a Codex resume. Investigate the current public task and implement only your own method.py/program.json in /workspace/solution. Run builders and the declared public Dev evaluator in this sandbox. The current public Task includes the newly authorized bounded graph compilation interface; read its public instructions and catalog. Choose meaningful methods yourself; preserve failed and worse experiments, source artifacts, notes and actual tool/provider clocks. Do not delegate, request peer/Reference/Hidden/QA material, manufacture time or repeat unchanged graphs to occupy time. Restore the best valid own method at each turn end. If meaningful research is exhausted, write evidence and your own reason in research-exhausted.json and stop. Prior exhaustion is preserved separately. All historical evidence applies to its original source, not new scores.')

TOOLS = [
    {
        'type': 'function',
        'function': {
            'name': 'shell',
            'description': 'Execute a public-task research shell command, your builder, or declared public Dev evaluator. '
                           'Runs in /workspace/solution. Returns stdout, stderr, exit_code. No network is available.',
            'parameters': {
                'type': 'object',
                'properties': {
                    'command': {'type': 'string', 'description': 'The shell command line to execute via /bin/sh -c'},
                    'timeout': {'type': 'integer', 'description': 'Seconds, default 120, max 1800', 'default': 120},
                },
                'required': ['command'],
            },
        },
    },
    {
        'type': 'function',
        'function': {
            'name': 'read_file',
            'description': 'Read a UTF-8 text file from /workspace, /output, /input, /skills or HOME, with optional offset/limit.',
            'parameters': {
                'type': 'object',
                'properties': {
                    'path': {'type': 'string'},
                    'offset': {'type': 'integer', 'description': '0-based byte offset', 'default': 0},
                    'max_bytes': {'type': 'integer', 'description': 'default 65536, max 262144', 'default': 65536},
                },
                'required': ['path'],
            },
        },
    },
    {
        'type': 'function',
        'function': {
            'name': 'write_file',
            'description': 'Write a UTF-8 text file. Only /workspace and /output are writable. Creates parent directories.',
            'parameters': {
                'type': 'object',
                'properties': {
                    'path': {'type': 'string'},
                    'content': {'type': 'string'},
                },
                'required': ['path', 'content'],
            },
        },
    },
    {
        'type': 'function',
        'function': {
            'name': 'list_dir',
            'description': 'List directory entries (name, type, size). Defaults to /workspace.',
            'parameters': {
                'type': 'object',
                'properties': {'path': {'type': 'string'}},
            },
        },
    },
]
TOOL_NAMES = {t['function']['name'] for t in TOOLS}


def emit(event):
    event.setdefault('epoch',time.time());event.setdefault('monotonic',time.monotonic())
    sys.stdout.write(json.dumps(event, separators=(',', ':')) + '\n')
    sys.stdout.flush()


def under(path, roots):
    return any(path == root or root in path.parents for root in roots)


def resolve(path, roots):
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = WORKDIR / candidate
    resolved = candidate.resolve()
    if not under(resolved, roots):
        raise PermissionError('path outside allowed roots: ' + str(resolved))
    return resolved


def tool_shell(arguments):
    command = arguments['command']
    timeout = min(int(arguments.get('timeout', 120)), 1800)
    env = {'PATH': os.environ.get('PATH', '/usr/local/bin:/usr/bin:/bin'),
           'HOME': os.environ.get('HOME', '/home/candidate'),
           'LANG': os.environ.get('LANG', 'C.UTF-8'),
           'PYTHONDONTWRITEBYTECODE': '1','PYTHONPATH':'/opt/bifrost','BIFROST_LANE':'seed'}
    started = time.time()
    try:
        proc = subprocess.run(['/bin/sh', '-c', command], cwd=str(WORKDIR), env=env,
                              capture_output=True, timeout=timeout, text=True, errors='replace')
        stdout, stderr, code = proc.stdout, proc.stderr, proc.returncode
        timed_out = False
    except subprocess.TimeoutExpired as error:
        stdout = error.stdout or ''
        stderr = (error.stderr or '') + '\n[timeout after %ss]' % timeout
        code, timed_out = 124, True
    def cap(value):
        value = value if isinstance(value, str) else value.decode(errors='replace')
        if len(value) > SHELL_OUTPUT_CAP:
            value = value[:SHELL_OUTPUT_CAP] + '\n[truncated %d bytes]' % (len(value) - SHELL_OUTPUT_CAP)
        return value
    return {'exit_code': int(code), 'stdout': cap(stdout), 'stderr': cap(stderr),
            'timed_out': timed_out, 'duration_seconds': round(time.time() - started, 3)}, int(code)


def tool_read(arguments):
    path = resolve(arguments['path'], READ_ROOTS)
    offset = max(int(arguments.get('offset', 0)), 0)
    max_bytes = min(int(arguments.get('max_bytes', 65536)), READ_CAP)
    if not path.is_file():
        raise FileNotFoundError(str(path))
    raw = path.read_bytes()
    chunk = raw[offset:offset + max_bytes]
    data = chunk.decode(errors='replace')
    return {'path': str(path), 'size': len(raw), 'offset': offset, 'content': data,
            'truncated': offset + len(chunk) < len(raw)}


def tool_write(arguments):
    path = resolve(arguments['path'], WRITE_ROOTS)
    path.parent.mkdir(parents=True, exist_ok=True)
    content = arguments['content']
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(content)
    tmp.replace(path)
    return {'path': str(path), 'bytes': len(content.encode())}


def tool_list(arguments):
    target = resolve(arguments.get('path') or str(WORKDIR), READ_ROOTS)
    if not target.is_dir():
        raise FileNotFoundError(str(target))
    entries = []
    for child in sorted(target.iterdir()):
        entries.append({'name': child.name,
                        'type': 'dir' if child.is_dir() else ('symlink' if child.is_symlink() else 'file'),
                        'size': child.stat().st_size if child.is_file() else None})
    return {'path': str(target), 'entries': entries}


DISPATCH = {'shell': tool_shell, 'read_file': tool_read, 'write_file': tool_write, 'list_dir': tool_list}


def execute_tool(call):
    name = call['function']['name']
    raw_args = call['function'].get('arguments') or '{}'
    tool_started=time.time()
    emit({'type':'tool.started','call_id':call.get('id'),'name':name})
    try:
        arguments = json.loads(raw_args)
        if name not in TOOL_NAMES:
            raise ValueError('unknown tool: ' + name)
        result, shell_code = DISPATCH[name](arguments), None
        if name == 'shell':
            result, shell_code = result
        ok = True
    except Exception as error:  # return the failure to the model as a tool result
        result = {'error_type': type(error).__name__, 'error': str(error)[:1000]}
        ok, shell_code = False, 1
    item_type = 'command_execution' if name == 'shell' else ('file_' + name)
    item = {'type': item_type, 'name': name}
    if name == 'shell':
        item['exit_code'] = shell_code if shell_code is not None else result.get('exit_code', 1)
    emit({'type': 'item.completed', 'call_id': call.get('id'), 'item': item, 'ok': ok,'started_epoch':tool_started,'completed_epoch':time.time()})
    return json.dumps(result, ensure_ascii=False)


RETRY_STATUS = {500, 502, 503, 504}


def stream_chat(base_url, model, messages):
    body = {'model': model, 'messages': messages, 'tools': TOOLS, 'tool_choice': 'auto',
            'stream': True, 'stream_options': {'include_usage': True}, 'max_tokens': 16384}
    # The loopback proxy is keyless; no credential is ever present here.
    request = urllib.request.Request(base_url, data=json.dumps(body).encode(),
                                      headers={'Content-Type': 'application/json'}, method='POST')

    def collect():
        local = {}
        finish_reason = None
        usage = None
        try:
            response = urllib.request.urlopen(request, timeout=300)
        except urllib.error.HTTPError as error:
            if error.code in RETRY_STATUS:
                raise
            raw = error.read(65536).decode(errors='replace')
            raise RuntimeError('upstream HTTP %s: %s' % (error.code, raw[:500]))
        except (urllib.error.URLError, socket.timeout):
            raise
        if response.status != 200:
            raise RuntimeError('upstream status %s' % response.status)
        with response:
            for line in response:
                line = line.decode(errors='replace').strip()
                if not line or not line.startswith('data:'):
                    continue
                payload = line[5:].strip()
                if payload == '[DONE]':
                    break
                data = json.loads(payload)
                if data.get('usage'):
                    usage = data['usage']
                for choice in data.get('choices', []):
                    if choice.get('index', 0) != 0:
                        raise RuntimeError('multiple choices unsupported')
                    if choice.get('finish_reason'):
                        finish_reason = choice['finish_reason']
                    delta = choice.get('delta', {})
                    index = 0
                    for tc in delta.get('tool_calls', []) or []:
                        index = tc.get('index', index)
                        slot = local.setdefault(index, {'id': tc.get('id'), 'type': 'function',
                                                        'function': {'name': '', 'arguments': ''}})
                        if tc.get('id'):
                            slot['id'] = tc['id']
                        fn = tc.get('function') or {}
                        if fn.get('name'):
                            slot['function']['name'] = fn['name']
                        if fn.get('arguments'):
                            slot['function']['arguments'] += fn['arguments']
                        index += 1
                    if delta.get('reasoning_content'):
                        emit({'type': 'reasoning.delta', 'text': delta['reasoning_content']})
                    if delta.get('content'):
                        local.setdefault('content', '')
                        local['content'] += delta['content']
        calls = [local[k] for k in sorted(k for k in local if isinstance(k, int))]
        content = local.get('content')
        assistant = {'role': 'assistant'}
        if calls:
            assistant['content'] = content or ''
            assistant['tool_calls'] = calls
        else:
            assistant['content'] = content or ''
        return assistant, calls, finish_reason, usage

    # Bounded backoff retry for transient transport failures. A turn only
    # executes tools after a complete, well-formed stream, so re-sending a
    # dropped read-only request has no side effects.
    delays = [60, 60, 60, 60, 60]
    last = None
    for attempt in range(5):
        try:
            return collect()
        except (urllib.error.URLError, socket.timeout, ConnectionError, BrokenPipeError) as error:
            last = error
            if attempt == 4:
                break
            emit({'type': 'transport.retry', 'attempt': attempt + 1,
                  'reason': type(error).__name__, 'sleep': delays[attempt]})
            time.sleep(delays[attempt])
    raise RuntimeError('upstream transport failed after retries: ' + str(last))


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-url', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--session-id', required=True)
    parser.add_argument('--prompt-file')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--transcript', required=True)
    parser.add_argument('--max-turns', type=int, default=120)
    parser.add_argument('--max-seconds', type=int, default=10800)
    args = parser.parse_args()

    transcript = Path(args.transcript)
    if args.resume:
        messages = json.loads(transcript.read_text())
        followup = Path(args.prompt_file).read_text()
        messages.append({'role':'system','content':SYSTEM})
        messages.append({'role': 'user', 'content': followup})
    else:
        prompt = Path(args.prompt_file).read_text()
        messages = [{'role': 'system', 'content': SYSTEM},
                    {'role': 'user', 'content': prompt}]

    emit({'type': 'thread.started', 'thread_id': args.session_id, 'epoch': time.time()})
    deadline = time.monotonic() + args.max_seconds
    outcome = 'error'
    turn = 0
    try:
        while turn < args.max_turns and time.monotonic() < deadline:
            turn += 1
            emit({'type': 'turn.started', 'turn': turn})
            assistant, calls, finish_reason, usage = stream_chat(args.base_url, args.model, messages)
            messages.append(assistant)
            emit({'type': 'turn.completed', 'turn': turn, 'finish_reason': finish_reason, 'usage': usage})
            transcript.parent.mkdir(parents=True, exist_ok=True)
            transcript.write_text(json.dumps(messages, ensure_ascii=False))
            if calls:
                for call in calls:
                    result = execute_tool(call)
                    messages.append({'role': 'tool', 'tool_call_id': call.get('id'), 'content': result})
                transcript.write_text(json.dumps(messages, ensure_ascii=False))
                continue
            if finish_reason == 'length':
                outcome = 'incomplete'
                break
            outcome = 'stop'
            break
    except Exception as error:
        emit({'type': 'error', 'error_type': type(error).__name__, 'error': str(error)[:1000]})
        transcript.write_text(json.dumps(messages, ensure_ascii=False))
        sys.exit(1)
    emit({'type': 'agent.finished', 'outcome': outcome, 'turns': turn})
    # 'length' is a clean provider boundary the host can resume; exit 0.
    sys.exit(0 if outcome in ('stop', 'incomplete') else 1)


if __name__ == '__main__':
    run()
