import datetime,hashlib,json,math,pathlib,re,shutil
BASE=pathlib.Path('runtime:historical_continuation/round13_fid_verification')
WORK=pathlib.Path('historical-seed-models/round13_cfg7_ultragentle')
OUT=pathlib.Path('historical-seed-workspace/agent-output/rounds/round13_cfg7_ultragentle')
AGENT=OUT.parent.parent
BASE.mkdir(parents=True,exist_ok=True)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
fidfile=OUT/'fid_result.txt'
fid=float(re.search(r'FID:\s*([-+0-9.eE]+)',fidfile.read_text())[1])
assert math.isfinite(fid) and fid>=0
log=OUT/'run.log';text=log.read_text()
assert 'ModuleNotFoundError: No module named' in text and f'FID-50k: {fid}' in text
manifest=OUT/'sample_sha256.txt';lines=manifest.read_text().splitlines()
assert len(lines)==50000
assert all(re.fullmatch(r'[0-9a-f]{64}\s+.+\.png',line) for line in lines)
assert {int(pathlib.Path(line.split(maxsplit=1)[1].strip()).stem) for line in lines}==set(range(50000))
grid=OUT/'sample_grid64.png';assert grid.is_file()
from PIL import Image
with Image.open(grid) as image: grid_size=list(image.size)
source=pathlib.Path('historical-resources/autoresearch_tasks/repos/distilled_decoding/VAR/evaluation/fid_score.py')
observed=source.read_bytes();(BASE/'fid_score.agent_observed.py').write_bytes(observed)
needle=b'pass  # removed ipdb debug statement'
assert observed.count(needle)==1
reconstructed=observed.replace(needle,b'import ipdb; ipdb.set_trace()')
(BASE/'fid_score.debug_line_reconstructed.py').write_bytes(reconstructed)
shutil.copy2(log,BASE/'source_run_log.at_collection.txt')
shutil.copy2(fidfile,BASE/'agent_fid_result.txt')
shutil.copy2(manifest,BASE/'source_sample_sha256.complete.txt')
shutil.copy2(grid,BASE/'sample_grid64.png')
raw=WORK/'sample/fid.txt'
assert not raw.exists()
shutil.copy2(fidfile,raw)
images_remaining=sum(1 for _ in (WORK/'sample/images').glob('*.png'))
receipt={'collected_at_utc':now(),'raw_fid_50k':fid,'source_fid_file':str(fidfile),
 'sample_count_from_source_manifest':50000,'current_png_count':images_remaining,
 'grid_size':grid_size,'grid_sha256':sha(grid),'sample_manifest_sha256':sha(manifest),
 'source_log_sha256_at_collection':sha(log),'original_failed_log_preserved':True,
 'fid_recovery_observed_in_source_log':True,'independent_fid_remeasurement_completed':False,
 'independent_verification_failure':'Remote source runner cleaned original 50000 PNG files before expert-side FID-only verification could start. Failed attempt log retained.',
 'fid_scorer_agent_observed_sha256':sha(source),
 'debug_line_reconstructed_scorer_sha256':hashlib.sha256(reconstructed).hexdigest(),
 'scorer_change_observed':'Agent replaced the debug ipdb line with pass in an error branch. Correct statistics-directory argument avoids that branch. Reconstructed copy restores only the previously observed line; it is not an independently archived original.',
 'stats_sha256':sha(pathlib.Path('historical-resources/models/dd1/fid_stats/statistic.pth')),
 'sampling_seed':42,'sampling_batch_size':16,'formal_protocol_valid':False,
 'raw_sample_fid_file_created_by_operator_from_actual_source_result':True,
 'operator_did_not_retrain_or_regenerate':True}
(BASE/'collection.json').write_text(json.dumps(receipt,indent=2)+'\n')
closure={'closed':True,'closed_at_utc':receipt['collected_at_utc'],
 'kind':'Operator evidence closure from actual completed agent FID recovery and full original-image SHA256 manifest; not an independent FID rerun',
 'raw_fid_50k':fid,'raw_fid_source':str(raw),'raw_fid_sha256':sha(raw),
 'sample_manifest_sha256':sha(manifest),'sample_count':50000,'grid_sha256':sha(grid),
 'sampling_seed':42,'sampling_batch_size':16,'formal_protocol_valid':False,
 'original_run_log_sha256_at_collection':sha(log),'original_run_log_unchanged_by_operator':True,
 'original_pngs_preserved':images_remaining==50000,
 'independent_fid_remeasurement_completed':False,
 'collection_record':str(BASE/'collection.json'),
 'reason':'Original runner passed a statistics file instead of a directory. Source log records a completed directory-corrected FID calculation. Agent also removed the debug ipdb line and cleaned PNG bytes after producing the full 50000-image SHA256 manifest. Preserve these facts and seed42 protocol deviation.'}
(AGENT/'ROUND13_RECOVERY_CLOSURE.json').write_text(json.dumps(closure,indent=2)+'\n')
with (AGENT/'RUNNER_ACTIONS.log').open('a') as file:
 file.write(now()+' ROUND13_EVIDENCE_CLOSURE: actual recovered FID and full original 50000-image manifest collected; PNGs already removed by source runner; extra FID verification not completed; seed42 and scorer debug-line changes recorded.\n')
print(json.dumps(receipt),flush=True)
