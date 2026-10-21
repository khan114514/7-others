"""Install the unchanged public initial method into Harbor's private tmpfs."""
from pathlib import Path
import shutil
out=Path('/workspace/solution');out.mkdir(parents=True,exist_ok=True)
method=out/'method.py'
if method.exists():raise RuntimeError('Initial workspace must be empty')
shutil.copyfile('/workspace/starter/method.py',method)

(out/'solve.sh').symlink_to('/tests/solve.sh')
