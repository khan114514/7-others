"""Trusted Hidden validation entry. Never exposed through the development API.

Only source bytes enter PolicyExecutor; candidate modules are never imported by
this GPU-host process. Private outputs must not be returned to model actors.
"""
import argparse
import hashlib
import itertools
import json
import math
import os
import re
import stat
import statistics
import time
from pathlib import Path

from deployment_paths import PRIVATE_ROOT as PRIVATE
ASSETS = PRIVATE / 'hidden-injected-v1'
JOBS = PRIVATE / 'hidden-validation-jobs'
SEEDS = [17, 29, 43, 71, 101]


def sha(path):
    return hashlib.sha256(regular(path, 64 * 1024 * 1024)).hexdigest()


def regular(path, limit):
    path = Path(path).absolute()
    if path.resolve() != path:
        raise ValueError('Symlink path is forbidden')
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_size > limit:
            raise ValueError('Expected bounded regular file')
        with os.fdopen(fd, 'rb', closefd=False) as stream:
            value = stream.read(limit + 1)
        if len(value) > limit:
            raise ValueError('File exceeds budget')
        return value
    finally:
        os.close(fd)


def atomic(path, value):
    pending = path.with_suffix('.pending.json')
    pending.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    pending.replace(path)


def preflight(job):
    if os.geteuid() != 0:
        raise PermissionError('Trusted root-only private validator required')
    if not re.fullmatch('[0-9a-f]{32}', job):
        raise ValueError('Invalid private job identity')
    folder = JOBS / job
    request = json.loads(regular(folder / 'request.json', 3 * 1024 * 1024))
    if set(request) != {'files', 'method_notes', 'source_kind'}:
        raise ValueError('Unexpected private request fields')
    if request['source_kind'] not in {'native_verified_artifact', 'expert_hook_probe'}:
        raise ValueError('Unknown trusted source provenance')
    if not isinstance(request['method_notes'], dict) or not request['method_notes'] or len(json.dumps(request['method_notes']).encode()) > 65536:
        raise ValueError('Required bounded method metadata missing')
    files = request['files']
    if not isinstance(files, dict) or 'method.py' not in files or not 1 <= len(files) <= 64:
        raise ValueError('Invalid source bundle')
    for name, value in files.items():
        if not isinstance(name, str) or not isinstance(value, str):
            raise ValueError('Source names and contents must be text')
        path = Path(name)
        if path.is_absolute() or '..' in path.parts or path.suffix != '.py' or len(value.encode()) > 1024 * 1024:
            raise ValueError('Invalid bounded Python source path')
    encoded = json.dumps(files, sort_keys=True, separators=(',', ':')).encode()
    if len(encoded) > 2 * 1024 * 1024:
        raise ValueError('Source bundle exceeds budget')
    contract = json.loads(regular(PRIVATE / 'hidden-validation-protocol-v2.json', 65536))
    signature = sha(PRIVATE / 'hidden-validation-protocol-v2.json')
    if contract['dev_protocol_sha256'] != sha(PRIVATE / 'policy-score-protocol-v2.json') or contract['formal_seeds'] != SEEDS:
        raise ValueError('Frozen development protocol differs')
    if sha(ASSETS / 'samples.json') != contract['hidden_samples_sha256']:
        raise ValueError('Injected Hidden rows differ from frozen private source')
    records = json.loads(regular(ASSETS / 'samples.json', 8 * 1024 * 1024))
    if len(records) != contract['hidden_samples_per_seed'] or len({r['id'] for r in records}) != len(records):
        raise ValueError('Incomplete or duplicate Hidden rows')
    for record in records:
        image = Path(record['image'])
        if image.is_absolute() or '..' in image.parts or image.parts[0] != 'images':
            raise ValueError('Invalid injected image path')
        if sha(ASSETS / image) != record['image_sha256']:
            raise ValueError('Injected image hash differs')
    audit = {'phase': 'PRIVATE_HIDDEN_PREFLIGHT_PASS_NO_GPU_SCORE',
             'private_protocol_sha256': signature, 'development_protocol_sha256': contract['dev_protocol_sha256'],
             'hidden_samples_sha256': contract['hidden_samples_sha256'], 'verified_images': len(records),
             'policy_sha256': hashlib.sha256(encoded).hexdigest(), 'source_kind': request['source_kind'],
             'candidate_imported_by_host': False, 'public_api_hidden_route': False,
             'model_research_seconds': 0, 'completed_epoch': time.time()}
    atomic(folder / 'preflight.json', audit)
    return folder, files, contract, records, audit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--job', required=True)
    parser.add_argument('--mode', choices=['preflight', 'smoke', 'full'], required=True)
    args = parser.parse_args()
    folder, files, contract, records, audit = preflight(args.job)
    if args.mode == 'preflight':
        print(json.dumps(audit), flush=True)
        return
    # Imports are all trusted installed scorer code, after private validation.
    import torch
    from gpu_official_encoder_smoke import wait_for_idle
    from isolated_policy import PolicyExecutor
    from mcq_metrics import parse_answer
    from policy_decoder_v1 import GuardedPolicyEncoder
    from run_inference_pilot import ROOT
    from run_policy_score_v2 import summarize

    target = folder / ('result-' + args.mode)
    target.mkdir(mode=0o700)
    seeds = SEEDS if args.mode == 'full' else [17]
    records = records if args.mode == 'full' else records[:2]
    status = {'phase': 'WAITING_FOR_GPU_EXCLUDED', 'mode': args.mode,
              'private_protocol_sha256': audit['private_protocol_sha256'], 'started_epoch': time.time(),
              'model_research_seconds': 0, 'samples_per_seed': len(records), 'seeds': seeds}
    atomic(target / 'status.json', status)
    lease = wait_for_idle()
    executors = {}
    try:
        status.update(phase='GPU_EXPERIMENT_RUNNING', gpu_started_epoch=time.time())
        atomic(target / 'status.json', status)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)
        torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(False)
        encoder = GuardedPolicyEncoder()
        ordinary = {'method.py': regular(PRIVATE / 'anchor_baseline_policy.py', 1024 * 1024).decode()}
        sources = {'calibration': ordinary, 'baseline': ordinary, 'candidate': files}
        startup = {}
        for role, source in sources.items():
            before = time.perf_counter()
            executors[role] = PolicyExecutor(source)
            startup[role] = time.perf_counter() - before
        for executor in executors.values():
            encoder.policy = executor
            encoder.generate_verified(ROOT / 'source/Heima/asserts/1987-BMW-E30-M3.png',
                                      'Which automotive brand does this car belong to?', 17, True)
        summaries = []
        for seed in seeds:
            directory = target / ('seed_' + str(seed))
            directory.mkdir()
            pairs = []
            for record in records:
                rng = int.from_bytes(hashlib.sha256(f"verified-decoding-validation-v1/{seed}/{record['id']}".encode()).digest()[:8], 'little') & ((1 << 63) - 1)
                labels = ['calibration', 'baseline', 'candidate']
                order = list(list(itertools.permutations(labels))[rng % math.factorial(len(labels))])
                pair = {'id': record['id'], 'seed': seed, 'role_order': order}
                for role in order:
                    encoder.policy = executors[role]
                    value = encoder.generate_verified(ASSETS / record['image'], record['question'] + '\nPlease give the option letter in your conclusion.', rng, True)
                    value['latency_seconds'] += startup[role] / (len(records) * len(seeds))
                    value['parsed_answer'] = parse_answer(value['output'], record['question'])
                    value['correct'] = value['parsed_answer'] == record['answer']
                    pair[role] = value
                pair['exact_token_equality'] = pair['baseline']['token_ids'] == pair['candidate']['token_ids']
                pair['calibration_token_equality'] = pair['baseline']['token_ids'] == pair['calibration']['token_ids']
                pair['answer_agreement'] = pair['baseline']['parsed_answer'] == pair['candidate']['parsed_answer']
                pairs.append(pair)
                with (directory / 'private-per-sample.jsonl').open('a') as stream:
                    stream.write(json.dumps(pair, ensure_ascii=False) + '\n')
                status.update(seed=seed, completed_questions=len(pairs))
                atomic(target / 'status.json', status)
            roles = {role: summarize([p[role] for p in pairs]) for role in labels}
            metrics = {role: statistics.mean(p[role]['latency_seconds'] / p['calibration']['latency_seconds'] for p in pairs) for role in ['baseline', 'candidate']}
            equality = statistics.mean(p['exact_token_equality'] for p in pairs)
            agreement = statistics.mean(p['answer_agreement'] for p in pairs)
            gates = contract['quality_gates']
            valid = all(p['calibration_token_equality'] for p in pairs) and equality >= gates['minimum_token_agreement'] and agreement >= gates['minimum_answer_agreement'] and roles['candidate']['accuracy'] >= roles['baseline']['accuracy'] - gates['maximum_accuracy_drop'] and all(v['parsed_answer_rate'] >= gates['minimum_parsed_answer_rate'] for v in roles.values())
            summary = {'seed': seed, 'roles': roles, 'latency_ratio_metrics': metrics,
                       'token_agreement_fraction': equality, 'answer_agreement_fraction': agreement, 'quality_pass': valid}
            summaries.append(summary)
            atomic(directory / 'summary.json', summary)
        baseline = statistics.mean(s['latency_ratio_metrics']['baseline'] for s in summaries)
        measured = statistics.mean(s['latency_ratio_metrics']['candidate'] for s in summaries)
        valid = all(s['quality_pass'] for s in summaries)
        score = (baseline - measured) / (baseline - contract['U'])
        result = dict(audit, phase='COMPLETE_PRIVATE_HIDDEN_VALIDATION', mode=args.mode,
                      B=baseline, U=contract['U'], measured_metric=measured,
                      normalized_score=score if valid else -1.0, unconditional_normalized_score=score,
                      all_quality_gates_pass=valid, seed_results=summaries, seeds=seeds,
                      samples_per_seed=len(records), complete_private_protocol=args.mode == 'full',
                      official_platform_hidden_claim=False, formal_model_trajectory_claim=False,
                      scorer_sha256=sha(Path(__file__)), decoder_sha256=sha(PRIVATE / 'policy_decoder_v1.py'),
                      policy_startup_seconds_included_amortized=startup, completed_epoch=time.time())
        atomic(target / 'result.json', result)
        status.update(phase='COMPLETE', completed_epoch=result['completed_epoch'])
        atomic(target / 'status.json', status)
        print(json.dumps({'phase': result['phase'], 'mode': args.mode,
                          'complete_private_protocol': result['complete_private_protocol'],
                          'official_platform_hidden_claim': False}), flush=True)
    finally:
        for executor in executors.values():
            executor.close()
        for lock in lease:
            lock.close()


if __name__ == '__main__':
    main()
