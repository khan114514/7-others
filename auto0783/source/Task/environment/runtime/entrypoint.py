"""Configure a keyless Codex instance and start its lane-only loopback proxy."""
import os
import subprocess
import sys
from pathlib import Path
import shutil

role = os.environ.get('BIFROST_LANE')
models = {'gpt':'gpt-6.1-sol','seed':'doubao-seed-2-1-turbo-260628'}
if role not in models:
    raise SystemExit('A known physical lane is required')
workspace = Path('/workspace/solution')
workspace.mkdir(parents=True, exist_ok=True)
for source in Path('/opt/bifrost/starter').iterdir():
    destination = workspace/source.name
    if not destination.exists():
        shutil.copyfile(source, destination)
home = Path(os.environ['HOME']) / '.codex'
home.mkdir(parents=True,exist_ok=True)
(home/'config.toml').write_text('''model_provider = "lane_bridge"
model = "%s"
review_model = "%s"
model_reasoning_effort = "high"
disable_response_storage = true
[model_providers.lane_bridge]
name = "Trusted lane bridge"
base_url = "http://127.0.0.1:8765"
wire_api = "responses"
requires_openai_auth = false
[features]
apps = false
code_mode_host = true
code_mode = true
image_generation = true
goals = true
[shell_environment_policy]
inherit = "none"
''' % (models[role],models[role]))
subprocess.Popen([sys.executable,'/opt/bifrost/local_model_proxy.py'],
                 stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
os.execvp(sys.argv[1],sys.argv[1:])
