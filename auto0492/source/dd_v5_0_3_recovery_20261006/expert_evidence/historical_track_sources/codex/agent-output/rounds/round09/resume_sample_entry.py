"""Replay the frozen sampler and verify existing PNGs before continuing."""
import hashlib
import io
import json
from pathlib import Path
import runpy
import sys

sys.path.insert(0, 'historical-codex-workspace')
from runtime_guard import apply_guard


def main():
    print(json.dumps({'resume_execution_guard': apply_guard()}), flush=True)
    import torchvision.utils
    original_save = torchvision.utils.save_image
    checked = 0

    def verify_existing(tensor, fp, **kwargs):
        nonlocal checked
        path = Path(fp)
        if path.exists():
            payload = io.BytesIO()
            original_save(tensor, payload, format='PNG', **kwargs)
            generated = payload.getvalue()
            original = path.read_bytes()
            if hashlib.sha256(generated).digest() != hashlib.sha256(original).digest():
                raise RuntimeError(f'Frozen-seed replay differs at {path.name}')
            checked += 1
            if checked == 1 or checked % 1000 == 0:
                print(json.dumps({'existing_png_exact_replay_verified': checked}), flush=True)
            return
        return original_save(tensor, fp, **kwargs)

    torchvision.utils.save_image = verify_existing
    runpy.run_path('historical-codex-workspace/official_entry.py', run_name='__main__')
    print(json.dumps({'existing_png_exact_replay_verified_total': checked}), flush=True)


if __name__ == '__main__':
    main()
