"""Build-time manifest. Run before distribution, never from test.sh."""
import json
from pathlib import Path
from config import sha

root = Path(__file__).resolve().parent.parent
files = []
for path in sorted(root.rglob('*')):
    relative = path.relative_to(root)
    if path.is_file() and not path.is_symlink() and '__pycache__' not in relative.parts and path.name != 'protected.json':
        files.append({'file': str(relative), 'sha256': sha(path)})
(root/'protected.json').write_text(json.dumps({'schema': 1, 'files': files}, indent=2)+'\n')
print(json.dumps({'protected_files': len(files)}))
