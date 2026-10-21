"""Configure the actual Codex CLI without provider keys in its container."""
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

MODELS = {"gpt": "gpt-6.1-sol", "seed": "doubao-seed-2-1-turbo-260628"}
role = os.environ.get("LATENT_LANE")
if role not in MODELS:
    raise SystemExit("A known independent lane is required")
home = Path.home() / ".codex"
home.mkdir(parents=True, exist_ok=True)
(home / "config.toml").write_text('''model_provider = "lane_bridge"
model = "%s"
review_model = "%s"
model_reasoning_effort = "high"
disable_response_storage = true
approval_policy = "never"
sandbox_mode = "workspace-write"
[model_providers.lane_bridge]
name = "Trusted lane bridge"
base_url = "http://127.0.0.1:8765"
wire_api = "responses"
requires_openai_auth = false
[features]
apps = false
code_mode_host = true
[shell_environment_policy]
inherit = "none"
set = { PATH = "/usr/local/bin:/usr/bin:/bin", HOME = "/home/candidate", LANG = "C.UTF-8" }
''' % (MODELS[role], MODELS[role]))
subprocess.Popen([sys.executable, "/opt/latentcli/local_model_proxy.py"], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
for _ in range(100):
    try:
        with socket.create_connection(("127.0.0.1", 8765), timeout=0.1):
            break
    except OSError:
        time.sleep(0.05)
else:
    raise SystemExit("The keyless CLI proxy did not start")
os.execvp(sys.argv[1], sys.argv[1:])
