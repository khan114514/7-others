"""One frozen GPU scorer for ordinary Baseline, Reference and candidate policies."""
import argparse
import hashlib
import json
import itertools
import math
import statistics
import time
from pathlib import Path

import torch
from gpu_official_encoder_smoke import wait_for_idle
from isolated_policy import PolicyExecutor
from mcq_metrics import parse_answer
from policy_decoder_v1 import GuardedPolicyEncoder
from run_inference_pilot import ROOT, sha

SEEDS = [17, 29, 43, 71, 101]
PRIVATE = ROOT / 'private'


def atomic(path, value):
    pending = path.with_suffix('.pending.json')
    pending.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    pending.replace(path)


def summarize(rows):
    return {'samples': len(rows), 'mean_latency_seconds': statistics.mean(r['latency_seconds'] for r in rows),
            'accuracy': statistics.mean(r['correct'] for r in rows),
            'parsed_answer_rate': statistics.mean(r['parsed_answer'] is not None for r in rows),
            'forward_calls': sum(r['target_forward_calls'] for r in rows),
            'generated_tokens': sum(r['generated_tokens'] for r in rows),
            'total_sampled_tokens': sum(r['total_sampled_tokens'] for r in rows),
            'guard_replays': sum(r['guard_replay'] for r in rows),
            'policy_calls': sum(r['policy_calls'] for r in rows),
            'policy_seconds': sum(r['policy_seconds'] for r in rows)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['anchors', 'candidate'], required=True)
    parser.add_argument('--split', choices=['pilot', 'dev'], default='dev')
    parser.add_argument('--output', required=True)
    parser.add_argument('--request')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--seeds', default=','.join(map(str, SEEDS)))
    args = parser.parse_args()
    target = Path(args.output).resolve()
    target.relative_to(PRIVATE)
    target.mkdir(mode=0o700)
    seeds = [int(v) for v in args.seeds.split(',')]
    if not seeds or any(v not in SEEDS for v in seeds):
        raise ValueError('Unsupported seed set')
    dataset = PRIVATE / 'evaluation-splits-v1' / args.split
    records = json.loads((dataset / 'samples.json').read_text())
    if args.limit:
        records = records[:args.limit]
    anchor = PRIVATE / ('frozen-policy-anchors-v2-' + args.split)
    contract = json.loads((PRIVATE / 'policy-score-protocol-v2.json').read_text())
    signature = sha(PRIVATE / 'policy-score-protocol-v2.json')
    if args.mode == 'anchors':
        files = {role: {'method.py': (PRIVATE / ('anchor_' + role + '_policy.py')).read_text()}
                 for role in ['baseline', 'reference']}
        files['calibration'] = {'method.py': (PRIVATE / 'anchor_baseline_policy.py').read_text()}
    else:
        request_path = Path(args.request).resolve()
        request_path.relative_to(PRIVATE / 'evaluator-jobs')
        request = json.loads(request_path.read_text())
        files = {'candidate': request['files'], 'calibration': {'method.py': (PRIVATE / 'anchor_baseline_policy.py').read_text()}}
        anchored = json.loads((anchor / 'comparison_summary.json').read_text())
        if anchored['protocol_sha256'] != signature or not anchored['all_quality_gates_pass']:
            raise ValueError('Valid matching anchors required')
    status = {'phase': 'WAITING_FOR_GPU_EXCLUDED', 'started_epoch': time.time(), 'protocol_sha256': signature,
              'split': args.split, 'mode': args.mode, 'samples_per_seed': len(records), 'seeds': seeds}
    atomic(target / 'status.json', status)
    print(json.dumps(status), flush=True)
    lease = wait_for_idle()
    status.update(phase='GPU_EXPERIMENT_RUNNING', gpu_started_epoch=time.time())
    atomic(target / 'status.json', status)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(True)
    torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(False)
    encoder = GuardedPolicyEncoder()
    executors = {}
    startup = {}
    try:
        for role, source in files.items():
            before = time.perf_counter()
            executors[role] = PolicyExecutor(source)
            startup[role] = time.perf_counter() - before
        demo = ROOT / 'source/Heima/asserts/1987-BMW-E30-M3.png'
        for role, executor in executors.items():
            encoder.policy = executor
            encoder.generate_verified(demo, 'Which automotive brand does this car belong to?', 17, True)
        seed_summaries = []
        own_outputs = []
        for seed in seeds:
            directory = target / ('seed_' + str(seed)); directory.mkdir()
            pairs = []
            for record in records:
                image = dataset / record['image']
                if sha(image) != record['image_sha256']:
                    raise ValueError('Changed image hash')
                rng = int.from_bytes(hashlib.sha256(f"verified-decoding-validation-v1/{seed}/{record['id']}".encode()).digest()[:8], 'little') & ((1 << 63) - 1)
                prompt = record['question'] + '\nPlease give the option letter in your conclusion.'
                labels = ['calibration', 'baseline', 'reference'] if args.mode == 'anchors' else ['calibration', 'candidate']
                order = list(list(itertools.permutations(labels))[rng % math.factorial(len(labels))])
                pair = {'id': record['id'], 'seed': seed, 'role_order': order}
                for role in order:
                    encoder.policy = executors[role]
                    result = encoder.generate_verified(image, prompt, rng, True)
                    result['latency_seconds'] += startup[role] / (len(records) * len(seeds))
                    result['parsed_answer'] = parse_answer(result['output'], record['question'])
                    result['correct'] = result['parsed_answer'] == record['answer']
                    pair[role] = result
                candidate_role = 'reference' if args.mode == 'anchors' else 'candidate'
                quality_baseline = 'baseline' if args.mode == 'anchors' else 'calibration'
                pair['exact_token_equality'] = pair[quality_baseline]['token_ids'] == pair[candidate_role]['token_ids']
                pair['calibration_token_equality'] = pair['calibration']['token_ids'] == pair[quality_baseline]['token_ids']
                pair['answer_agreement'] = pair[quality_baseline]['parsed_answer'] == pair[candidate_role]['parsed_answer']
                pairs.append(pair)
                with (directory / 'per-sample.jsonl').open('a') as stream:
                    stream.write(json.dumps(pair, ensure_ascii=False) + '\n')
                status.update(seed=seed, completed_questions=len(pairs), phase='GPU_EXPERIMENT_RUNNING')
                atomic(target / 'status.json', status)
                print(json.dumps({'seed': seed, 'completed': len(pairs), 'mode': args.mode}), flush=True)
                if args.mode == 'candidate' and seed == seeds[0] and len(own_outputs) < 24:
                    own_outputs.append({'question': record['question'], 'token_ids': pair['candidate']['token_ids'],
                                        'output': pair['candidate']['output'], 'latency_seconds': pair['candidate']['latency_seconds'],
                                        'guard_replay': pair['candidate']['guard_replay']})
            roles = {role: summarize([pair[role] for pair in pairs]) for role in labels}
            metrics = {role: statistics.mean(pair[role]['latency_seconds'] / pair['calibration']['latency_seconds'] for pair in pairs) for role in labels if role != 'calibration'}
            equality = statistics.mean(pair['exact_token_equality'] for pair in pairs)
            agreement = statistics.mean(pair['answer_agreement'] for pair in pairs)
            passed = all(pair['calibration_token_equality'] for pair in pairs) and equality >= .99 and agreement >= .99 and roles[candidate_role]['accuracy'] >= roles[quality_baseline]['accuracy'] - .01 and all(v['parsed_answer_rate'] >= .9 for v in roles.values())
            summary = {'seed': seed, 'roles': roles, 'latency_ratio_metrics': metrics, 'token_agreement_fraction': equality,
                       'answer_agreement_fraction': agreement, 'quality_pass': passed}
            seed_summaries.append(summary)
            atomic(directory / 'summary.json', summary)
        candidate = statistics.mean(s['latency_ratio_metrics'][candidate_role] for s in seed_summaries)
        raw_candidate = statistics.mean(s['roles'][candidate_role]['mean_latency_seconds'] for s in seed_summaries)
        raw_calibration = statistics.mean(s['roles']['calibration']['mean_latency_seconds'] for s in seed_summaries)
        if args.mode == 'anchors':
            baseline = statistics.mean(s['latency_ratio_metrics']['baseline'] for s in seed_summaries)
            sigma = statistics.stdev(s['latency_ratio_metrics']['baseline'] for s in seed_summaries) if len(seeds) > 1 else None
        else:
            baseline = anchored['B']
            sigma = anchored['baseline_sample_standard_deviation_n_minus_1']
        valid = all(s['quality_pass'] for s in seed_summaries)
        reduction = (baseline - candidate) / (baseline - contract['U'])
        complete_primary = args.split == 'dev' and seeds == SEEDS and len(records) == 96
        result = {'status': 'COMPLETE_POLICY_SCORER_RUN', 'B': baseline, 'measured_metric': candidate, 'metric_units': 'dimensionless paired latency ratio',
                  'measured_latency_seconds': raw_candidate, 'same_batch_calibration_latency_seconds': raw_calibration,
                  'U': contract['U'], 'normalized_score': reduction if valid else -1.0,
                  'unconditional_normalized_score': reduction, 'same_batch_raw_latency_reduction': 1 - raw_candidate / raw_calibration, 'all_quality_gates_pass': valid,
                  'baseline_sample_standard_deviation_n_minus_1': sigma,
                  'reference_headroom_ge_3_sigma_B': sigma is not None and candidate - contract['U'] >= 3 * sigma,
                  'gain_ge_3_sigma_B': sigma is not None and baseline - candidate >= 3 * sigma,
                  'normalized_reference_in_015_080': .15 <= reduction <= .8,
                  'seed_results': seed_summaries, 'seeds': seeds, 'samples_per_seed': len(records),
                  'split': args.split, 'complete_primary_protocol': complete_primary,
                  'protocol_sha256': signature, 'scorer_sha256': sha(Path(__file__)),
                  'decoder_sha256': sha(PRIVATE / 'policy_decoder_v1.py'),
                  'data_sha256': sha(dataset / 'samples.json'), 'policy_startup_seconds_included_amortized': startup,
                  'training': 'NOT_APPLICABLE', 'formal_anchor_claim': False,
                  'remaining_gate': 'Native Harbor and full task/controller isolation proof',
                  'gpu_started_epoch': status['gpu_started_epoch'], 'completed_epoch': time.time()}
        if args.mode == 'anchors':
            result['R'] = candidate
            atomic(target / 'comparison_summary.json', result)
        else:
            result['own_development_outputs'] = own_outputs
            atomic(target / 'result.json', result)
        status.update(phase='COMPLETE', completed_epoch=result['completed_epoch'])
        atomic(target / 'status.json', status)
        print(json.dumps(result), flush=True)
    finally:
        for executor in executors.values(): executor.close()
        for lock in lease: lock.close()


if __name__ == '__main__':
    main()
