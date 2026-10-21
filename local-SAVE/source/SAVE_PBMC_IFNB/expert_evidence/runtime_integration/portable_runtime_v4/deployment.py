"""Trusted local deployment input. Never passed to Actor or numeric child."""
import json,os,stat
from pathlib import Path
def read():
 p=Path(os.environ['SAVE_DEPLOYMENT_FILE']).resolve()
 if not p.is_file() or p.is_symlink() or stat.S_IMODE(p.stat().st_mode)!=0o600:raise ValueError('Private deployment profile must be a regular 0600 file')
 value=json.loads(p.read_text());allowed={'controller_task_root','remote_root','gpu_hosts','ssh_host','ssh_user','ssh_key_file','gpt_keychain_service','gpt_keychain_account','ark_profile','local_proxy'}
 if set(value)!=allowed:raise ValueError('Deployment fields differ from the supported profile')
 if value['local_proxy'] not in [None,'http://127.0.0.1:7892']:raise ValueError('Only the separately authorized local proxy is supported')
 for host in value['gpu_hosts']:
  if set(host)!={'port','gpu','locks'} or not isinstance(host['port'],int) or not host['locks']:raise ValueError('Complete existing-lock GPU identity required')
 return value
