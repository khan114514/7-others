"""Trusted fixed broker; request roles come from physical volumes, never payloads."""
import json
import os
import re
import stat
import sys
import threading
import time
from pathlib import Path

ROOTS = {role:Path('/q')/role for role in ['gpt','seed','verifier']}
LIMIT = 2*1024*1024
UUID = re.compile(r'^[0-9a-f]{32}$')
dead = threading.Event()

def regular(path):
    fd = os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_size > LIMIT:
            raise ValueError('Bad request frame')
        return os.read(fd,LIMIT+1)
    finally:
        os.close(fd)

def incoming():
    try:
        while True:
            line = sys.stdin.buffer.readline(LIMIT+1)
            if not line:
                return
            if len(line) > LIMIT:
                raise ValueError('Oversized control frame')
            frame = json.loads(line)
            role,nonce,sequence = frame['role'],frame['id'],frame['sequence']
            if role not in ROOTS or not UUID.fullmatch(nonce) or type(sequence) is not int or not 0<=sequence<100000:
                raise ValueError('Bad trusted frame identity')
            if frame['kind'] not in ['header','chunk','end']:
                raise ValueError('Bad trusted response kind')
            value = {key:val for key,val in frame.items() if key != 'role'}
            target = ROOTS[role]/(nonce+'.'+str(sequence)+'.res')
            pending = target.with_suffix('.pending-response')
            fd = os.open(pending,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
            with os.fdopen(fd,'wb') as handle:
                handle.write(json.dumps(value,separators=(',',':')).encode())
            pending.replace(target)
    finally:
        dead.set()

threading.Thread(target=incoming,daemon=True).start()
while not dead.is_set():
    for role,root in ROOTS.items():
        if not root.is_dir():
            continue
        for path in list(root.glob('*.req'))[:8]:
            nonce = path.name[:-4]
            if not UUID.fullmatch(nonce):
                continue
            try:
                payload = json.loads(regular(path))
                if set(payload) != {'id','operation','payload'} or payload['id'] != nonce:
                    raise ValueError('Bad request identity')
                if payload['operation'] not in ['responses','evaluate','transport_probe']:
                    raise ValueError('Unsupported operation')
                path.unlink()
                print(json.dumps({'role':role,**payload},separators=(',',':')),flush=True)
            except Exception:
                # Invalid lane input is never promoted to a host command.
                try:
                    path.unlink()
                except FileNotFoundError:
                    pass
    time.sleep(0.05)
