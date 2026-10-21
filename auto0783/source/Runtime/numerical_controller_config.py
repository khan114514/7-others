"""Trusted launcher configuration. No candidate-selected host, path or command."""
import argparse
import hashlib
import json
import re
import shlex
from pathlib import Path,PurePosixPath

MANIFEST_SHA='61d7fabc0298197c99c63317cfc906360d3dfd06e78741cdf4ac728ecf5d94fe'
SCHEMA='bifrost-trusted-controller/1'
KEYS={'schema','ssh_target','python','remote_root','gpu_uuid','locks','state_directory'}
PROBE='''import fcntl,json,subprocess,sys
handles=[]
try:
 for path in sys.argv[2:]:
  handle=open(path,'r');handles.append(handle)
  fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
 raw=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.used','--format=csv,noheader,nounits'],text=True).strip().split(',')
 print(json.dumps({'available':raw[0].strip()==sys.argv[1] and int(raw[1])<=64,'gpu_uuid':raw[0].strip(),'memory_mib':int(raw[1]),'both_existing_leases_available':True}))
except BlockingIOError:
 print(json.dumps({'available':False,'both_existing_leases_available':False,'reason':'An existing scheduling lease is held'}))
finally:
 for handle in handles:handle.close()
'''


def absolute_path(value):
    if not isinstance(value,str) or not value or any(x in value for x in ['\x00','\n','\r']):raise ValueError('Invalid trusted path')
    path=PurePosixPath(value)
    if not path.is_absolute() or '..' in path.parts or str(path)=='/':raise ValueError('Expected a bounded absolute trusted path')
    return value


def validate(value):
    if not isinstance(value,dict) or set(value)!=KEYS or value['schema']!=SCHEMA:raise ValueError('Unknown trusted controller configuration')
    if not isinstance(value['ssh_target'],str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]{0,127}',value['ssh_target']):raise ValueError('Use a configured SSH host alias, never options or a command')
    for key in ['python','remote_root','state_directory']:absolute_path(value[key])
    if not isinstance(value['gpu_uuid'],str) or not re.fullmatch(r'GPU-[a-fA-F0-9]{8}(?:-[a-fA-F0-9]{4}){3}-[a-fA-F0-9]{12}',value['gpu_uuid']):raise ValueError('Invalid assigned GPU identity')
    if not isinstance(value['locks'],list) or len(value['locks'])<2 or len(value['locks'])>8 or len(set(value['locks']))!=len(value['locks']):raise ValueError('Supply the distinct existing shared scheduler lock paths')
    for path in value['locks']:absolute_path(path)
    return value


def worker_command(config):
    root=config['remote_root']
    arguments=['env','PYTHONPATH='+root,config['python'],'-u',root+'/numerical_worker.py','--root',root,'--manifest-sha256',MANIFEST_SHA,'--gpu-uuid',config['gpu_uuid']]
    for path in config['locks']:arguments.extend(['--lock',path])
    return ['ssh','-T','-o','BatchMode=yes','-o','ClearAllForwardings=yes','-o','ServerAliveInterval=15','-o','ServerAliveCountMax=3',config['ssh_target'],shlex.join(arguments)]


def probe_command(config):
    return ['ssh','-T','-o','BatchMode=yes','-o','ClearAllForwardings=yes','-o','ConnectTimeout=10',config['ssh_target'],shlex.join([config['python'],'-I','-c',PROBE,config['gpu_uuid'],*config['locks']])]


def load_config():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--start-new-worker',action='store_true')
    args=parser.parse_args()
    if args.config.is_symlink() or not args.config.is_file() or args.config.stat().st_size>65536:raise ValueError('Trusted configuration must be a bounded regular file')
    config=validate(json.loads(args.config.read_text()))
    if not args.start_new_worker:
        print(json.dumps({'status':'DESCRIBE_ONLY_NO_WORKER_OR_GPU_STARTED','manifest_sha256':MANIFEST_SHA,'command':worker_command(config),'existing_locks_are_never_created_or_removed':True}))
        raise SystemExit(0)
    return config
