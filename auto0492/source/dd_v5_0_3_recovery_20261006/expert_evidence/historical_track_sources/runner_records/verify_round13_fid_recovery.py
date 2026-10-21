import datetime, hashlib, importlib.util, json, math, pathlib, re, shutil, time

BASE = pathlib.Path('runtime:historical_continuation/round13_fid_verification')
WORK = pathlib.Path('historical-seed-models/round13_cfg7_ultragentle')
OUT = pathlib.Path('historical-seed-workspace/agent-output/rounds/round13_cfg7_ultragentle')
AGENT = OUT.parent.parent
REPO = pathlib.Path('historical-resources/autoresearch_tasks/repos/distilled_decoding/VAR')
STATS = pathlib.Path('historical-resources/models/dd1/fid_stats')
BASE.mkdir(parents=True, exist_ok=True)
def now(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as file:
        for chunk in iter(lambda:file.read(1024*1024), b''): h.update(chunk)
    return h.hexdigest()

started = now()
images = sorted((WORK/'sample/images').glob('*.png'), key=lambda p:int(p.stem))
assert len(images) == 50000 and {int(p.stem) for p in images} == set(range(50000))
source = REPO/'evaluation/fid_score.py'
observed = source.read_bytes()
(BASE/'fid_score.agent_observed.py').write_bytes(observed)
# Restore the exact debug line observed before the remote agent removed it.
# The corrected statistics-directory argument never enters this debug branch.
patched = b'pass  # removed ipdb debug statement'
original = b'import ipdb; ipdb.set_trace()'
assert observed.count(patched) == 1 or (patched not in observed and original in observed)
restored = observed.replace(patched, original)
scorer = BASE/'fid_score.debug_restored.py'
scorer.write_bytes(restored)
shutil.copy2(OUT/'run.log', BASE/'source_run_log.before_verification.txt')
shutil.copy2(OUT/'fid_result.txt', BASE/'agent_fid_result.txt')
receipt = {'started_at_utc':started, 'sample_count':50000,
           'source_scorer_sha256':hashlib.sha256(observed).hexdigest(),
           'debug_restored_scorer_sha256':sha(scorer),
           'scorer_change':'Only the previously observed ipdb debug line was restored in an isolated copy; active metric code is byte-identical to observed source.',
           'stats_sha256':sha(STATS/'statistic.pth'),
           'inception_sha256':sha(pathlib.Path('historical-runtime-cache/torch/hub/checkpoints/pt_inception-2015-12-05-6726825d.pth')),
           'generation_seed':42, 'generation_batch_size':16,
           'fid_batch_size':256,'fid_dims':2048,'fid_workers':8,
           'images_regenerated':False,'training_rerun':False,
           'original_failed_log_preserved':True,
           'formal_protocol_valid':False}
(BASE/'started.json').write_text(json.dumps(receipt,indent=2)+'\n')
import torch
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
spec = importlib.util.spec_from_file_location('codex_isolated_fid_score',scorer)
module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
t = time.monotonic()
fid = float(module.calculate_fid_given_paths(
    (None,str(WORK/'sample/images')),batch_size=256,device='cuda',dims=2048,
    num_workers=8,load_act=[str(STATS),None],save_act=[None,None]))
assert math.isfinite(fid) and fid >= 0
agent_fid = float(re.search(r'FID:\s*([-+0-9.eE]+)',(BASE/'agent_fid_result.txt').read_text())[1])
receipt.update({'fid_50k':fid,'agent_fid_50k':agent_fid,'same_fid':fid==agent_fid,
                'fid_elapsed_seconds':time.monotonic()-t,'torch_version':torch.__version__})
assert receipt['same_fid'], 'Independent recovered-scorer FID differs; retain results for investigation'
raw = WORK/'sample/fid.txt'
if raw.exists():
    assert float(re.search(r'FID:\s*([-+0-9.eE]+)',raw.read_text())[1]) == fid
else:
    raw.write_text(f'FID: {fid}\n')
    receipt['raw_file_created_from_actual_verification'] = True
manifest = BASE/'sample_sha256.verified.txt'
with manifest.open('w') as file:
    for p in images: file.write(sha(p)+'  '+str(p)+'\n')
assert len(manifest.read_text().splitlines()) == 50000
source_manifest = OUT/'sample_sha256.txt'
source_lines = source_manifest.read_text().splitlines() if source_manifest.exists() else []
if len(source_lines) != 50000:
    if source_manifest.exists(): shutil.copy2(source_manifest, BASE/'source_sample_sha256.incomplete.txt')
    shutil.copy2(manifest,source_manifest)
    receipt['incomplete_source_manifest_recovered'] = True
else:
    assert source_manifest.read_bytes() == manifest.read_bytes()
grid = OUT/'sample_grid64.png'; assert grid.is_file()
shutil.copy2(grid,BASE/'sample_grid64.png')
receipt.update({'completed_at_utc':now(),'sample_manifest_sha256':sha(manifest),
                'grid_sha256':sha(grid),'raw_fid_sha256':sha(raw),
                'source_log_sha256_at_collection':sha(OUT/'run.log'),
                'original_pngs_preserved':True})
(BASE/'verification.json').write_text(json.dumps(receipt,indent=2)+'\n')
closure = {'closed':True,'closed_at_utc':receipt['completed_at_utc'],
           'kind':'FID-only verification and evidence recovery over the original 50000 PNG files',
           'raw_fid_50k':fid,'raw_fid_source':str(raw),'raw_fid_sha256':sha(raw),
           'sample_manifest_sha256':sha(source_manifest),'sample_count':50000,
           'grid_sha256':sha(grid),'sampling_seed':42,'sampling_batch_size':16,
           'formal_protocol_valid':False,'original_run_log_sha256':sha(OUT/'run.log'),
           'original_run_log_unchanged_by_operator':True,
           'reason':'Original runner supplied a statistics file where a directory is required. Agent recovered FID; isolated scorer with original debug line restored reproduced the exact value. Full image manifest collected without shell glob expansion. Original failure and recovery logs remain.',
           'verification_record':str(BASE/'verification.json'),'original_pngs_preserved':True}
(AGENT/'ROUND13_RECOVERY_CLOSURE.json').write_text(json.dumps(closure,indent=2)+'\n')
with (AGENT/'RUNNER_ACTIONS.log').open('a') as file:
    file.write(now()+' ROUND13_EVIDENCE_RECOVERY: exact FID verified over original PNGs; isolated scorer restores original debug line; statistics directory corrected; seed42 mismatch retained.\n')
print(json.dumps(receipt),flush=True)
