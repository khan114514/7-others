"""Online build phase only; each public asset is content-addressed."""
import argparse
import json
from pathlib import Path
import urllib.request
from config import sha

parser = argparse.ArgumentParser()
parser.add_argument('--destination', required=True)
args = parser.parse_args()
root = Path(args.destination)
manifest = json.loads((Path(__file__).resolve().parent.parent/'assets.json').read_text())
for row in manifest['files']:
    destination = root/row['file']
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size == row['bytes'] and sha(destination) == row['sha256']:
        continue
    stage = destination.with_suffix(destination.suffix+'.download')
    with urllib.request.urlopen(row['url'], timeout=120) as source, stage.open('wb') as output:
        while block := source.read(8 << 20):
            output.write(block)
    if stage.stat().st_size != row['bytes'] or sha(stage) != row['sha256']:
        raise RuntimeError('Downloaded asset integrity mismatch: '+row['file'])
    stage.replace(destination)
    print(json.dumps({'asset':row['file'],'sha256':row['sha256']}), flush=True)
