"""Start protected Public/Dev service before dropping to the coding user."""
import os,pathlib,subprocess,sys,time

if os.getuid()!=0:
    raise SystemExit('The task entrypoint requires startup identity root')
logs=pathlib.Path('/logs/agent');logs.mkdir(parents=True,exist_ok=True);logs.chmod(0o1777)
log=(logs/'public_service.log').open('ab',buffering=0)
service=subprocess.Popen([sys.executable,'-I','/opt/public_eval/dev_service.py'],stdout=log,stderr=subprocess.STDOUT)
ready=pathlib.Path('/tmp/.public-valuation/ready');deadline=time.monotonic()+15
while not ready.is_file():
    if service.poll() is not None:raise SystemExit('Protected Public/Dev service startup failed')
    if time.monotonic()>deadline:raise SystemExit('Protected Public/Dev service startup timed out')
    time.sleep(.05)
os.setgroups([]);os.setgid(1000);os.setuid(1000)
command=sys.argv[1:] or ['sleep','infinity']
os.execvp(command[0],command)
