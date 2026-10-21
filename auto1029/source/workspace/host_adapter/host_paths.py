"""Trusted host configuration; no deployment bindings enter an Actor image."""
import json
import os
import re
import shlex
import stat
import urllib.parse
from pathlib import Path


def load(path):
    path = Path(path)
    if not path.is_absolute():
        raise ValueError('Host config requires an absolute path')
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.geteuid() or stat.S_IMODE(info.st_mode) != 0o600:
            raise PermissionError('Host config requires current trusted operator ownership and mode0600')
        raw = os.read(fd, 65537)
        if len(raw) > 65536:
            raise ValueError('Host config exceeds budget')
    finally:
        os.close(fd)
    value = json.loads(raw)
    required = {'private_root', 'docker_executable', 'gpu_ssh_host', 'gpu_service_argv',
                'credential_environment_names', 'credential_commands', 'http_proxy', 'verifier_gpu_service_argv'}
    if set(value) != required:
        raise ValueError('Unexpected host configuration fields')
    for name in ['private_root', 'docker_executable']:
        item = value[name]
        if not isinstance(item, str) or not Path(item).is_absolute() or '\x00' in item:
            raise ValueError('Host path must be an absolute nonsecret deployment path')
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.@-]{0,253}', value['gpu_ssh_host']):
        raise ValueError('Invalid SSH host or alias')
    for name in ['credential_environment_names', 'credential_commands']:
        if set(value[name]) != {'gpt', 'seed'}:
            raise ValueError('Both fixed roles require explicit credential configuration')
    for role, name in value['credential_environment_names'].items():
        if not re.fullmatch(r'[A-Z][A-Z0-9_]{0,127}', name):
            raise ValueError('Invalid credential environment variable name')
    for command in [value['gpu_service_argv'], value['verifier_gpu_service_argv'], *value['credential_commands'].values()]:
        if not isinstance(command, list) or len(command) > 32 or any(
                not isinstance(arg, str) or not arg or len(arg) > 4096 or '\x00' in arg or '\n' in arg for arg in command):
            raise ValueError('Invalid trusted argv')
        if command and not Path(command[0]).is_absolute():
            raise ValueError('Trusted command requires an absolute executable')
    if any(not value[key] or '--role' in value[key] for key in ['gpu_service_argv','verifier_gpu_service_argv']):
        raise ValueError('GPU argv must omit the physical role selected by the bridge')
    proxy = value['http_proxy']
    if proxy is not None:
        if not isinstance(proxy, str):
            raise ValueError('Proxy must be a nonsecret URL or null')
        parsed = urllib.parse.urlsplit(proxy)
        if parsed.scheme not in {'http', 'https'} or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError('Proxy URL cannot contain credentials or capability parameters')
    root = Path(value['private_root'])
    if root.is_symlink():
        raise ValueError('Host private root cannot be a symlink')
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    info = root.stat()
    if info.st_uid != os.geteuid() or info.st_mode & 0o077:
        raise PermissionError('Host evidence directory requires mode0700')
    return value


CONFIG_PATH = Path(os.environ['HEIMA_HOST_CONFIG'])
CONFIG = load(CONFIG_PATH)
PRIVATE_ROOT = Path(CONFIG['private_root'])
DOCKER_EXECUTABLE = CONFIG['docker_executable']


def gpu_command(role, *, verifier=False):
    if role not in {'gpt', 'seed'}:
        raise ValueError('Unknown fixed physical role')
    if type(verifier) is not bool:
        raise ValueError('Trusted verifier routing flag must be boolean')
    key = 'verifier_gpu_service_argv' if verifier else 'gpu_service_argv'
    remote = shlex.join([*CONFIG[key], '--role', role])
    return ['ssh', '-o', 'BatchMode=yes', '-o', 'ClearAllForwardings=yes', CONFIG['gpu_ssh_host'], remote]
