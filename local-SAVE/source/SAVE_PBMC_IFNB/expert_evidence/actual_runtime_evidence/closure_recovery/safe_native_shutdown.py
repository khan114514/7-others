"""Stop only the owned native CLI/proxy trees, preserving container keepalive."""
CODE = '''import os,signal,json
from pathlib import Path
rows={}
for p in Path('/proc').glob('[0-9]*'):
 try:
  fields={x.split(':',1)[0]:x.split(':',1)[1].strip() for x in (p/'status').read_text().splitlines() if ':' in x}
  rows[int(p.name)]={'uid':fields['Uid'].split()[0],'ppid':int(fields['PPid']),
                    'exe':str((p/'exe').resolve()),'argv':(p/'cmdline').read_bytes().decode(errors='replace').split(chr(0))}
 except (OSError,KeyError,ValueError):pass
protected={1};current=os.getpid()
while current and current not in protected:
 protected.add(current);current=rows.get(current,{}).get('ppid',0)
roots={pid for pid,r in rows.items() if pid not in protected and r['uid']=='1000' and
       (r['exe']=='/opt/save-cli/codex' or
        (Path(r['exe']).name.startswith('python') and '/opt/save-cli/local_model_proxy.py' in r['argv']))}
selected=set(roots)
while True:
 extra={pid for pid,r in rows.items() if r['ppid'] in selected and r['uid']=='1000' and pid not in protected}
 if extra<=selected:break
 selected|=extra
stopped=[]
for pid in sorted(selected,reverse=True):
 try:os.kill(pid,signal.SIGKILL);stopped.append(pid)
 except (ProcessLookupError,PermissionError):pass
print(json.dumps({'roots':sorted(roots),'stopped':stopped,'container_keepalive_preserved':True}))
'''
