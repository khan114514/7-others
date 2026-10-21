"""Retry the unchanged official FID on a complete interrupted sample set."""
import argparse
import faulthandler
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, 'historical-codex-workspace')
from runtime_guard import apply_guard


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--images', required=True)
    parser.add_argument('--fid-output', required=True)
    args = parser.parse_args()
    faulthandler.enable(all_threads=True)
    print(json.dumps({'execution_guard': apply_guard()}), flush=True)
    import torch
    torch.cuda.set_per_process_memory_fraction(.30)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    sys.path.insert(0, str(Path.cwd()))
    from evaluation.fid_score import calculate_fid_given_paths
    images = Path(args.images)
    paths = sorted(images.glob('*.png'), key=lambda p: int(p.stem))
    assert len(paths) == 50000
    assert [int(p.stem) for p in paths] == list(range(50000))
    fid = calculate_fid_given_paths(
        (None, str(images)), batch_size=256, device='cuda', dims=2048,
        num_workers=8,
        load_act=['historical-resources/models/dd1/fid_stats', None],
        save_act=[None, None],
    )
    Path(args.fid_output).write_text(f'FID: {fid}')
    print(json.dumps({'fid_50k': fid, 'finished_unix': time.time(),
                     'peak_torch_allocated_mib': torch.cuda.max_memory_allocated() / 2**20,
                     'peak_torch_reserved_mib': torch.cuda.max_memory_reserved() / 2**20}), flush=True)


if __name__ == '__main__':
    main()
