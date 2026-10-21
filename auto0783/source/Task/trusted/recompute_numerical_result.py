"""Independent CPU audit of raw timings and persisted final latent tensors.

This does not import the evaluator or graph interpreter and initializes no CUDA
context. It is an evidence recomputation, not an independent method rerun.
"""
import argparse
import collections
import hashlib
import json
import math
import statistics
from pathlib import Path
import torch
from safetensors import safe_open


def close(actual, expected):
    if not math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-8):
        raise ValueError('Persisted aggregate disagrees with independent raw-data recomputation')


def close_float32_reduction(actual, expected):
    # CPU and CUDA reduction trees can differ by float32 roundoff. Raw timing
    # aggregates still use the tight double-precision check above. This only
    # compares two independently computed latent metrics; it changes no gate.
    if expected == 0.0 and actual != 0.0:
        raise ValueError('Persisted zero latent error is not independently exact')
    if not math.isclose(actual, expected, rel_tol=1e-6, abs_tol=1e-8):
        raise ValueError('Persisted latent metric differs beyond float32 reduction tolerance')
    if (actual <= 0.001) != (expected <= 0.001):
        raise ValueError('CPU and GPU latent metrics disagree at the unchanged quality threshold')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--job', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    root = Path(args.job)
    torch.set_num_threads(2)
    result = json.loads((root / 'result.json').read_text())
    if result.get('execution_status') != 'COMPLETED' or result.get('child_exit_code') != 0:
        raise ValueError('Only completed numerical runs can be recomputed')
    raw = json.loads((root / 'raw-stage-times.json').read_text())
    records = result['records']
    groups = collections.defaultdict(list)
    for item in raw:
        if not all(math.isfinite(item[k]) and item[k] >= 0 for k in
                   ['baseline_ms', 'candidate_ms', 'relative_rmse', 'scaled_max_abs']):
            raise ValueError('Raw numerical evidence has invalid values')
        groups[(item['case'], item['sampling_seed'], item['repetition'])].append(item)
    if len(groups) != len(records):
        raise ValueError('Raw repetition set is incomplete')
    totals = []
    quality = True
    for record in records:
        group = groups[(record['case'], record['sampling_seed'], record['repetition'])]
        if sorted(item['step'] for item in group) != list(range(result['steps'])):
            raise ValueError('A repetition is missing or duplicating stages')
        b = sum(item['baseline_ms'] for item in group)
        t = sum(item['candidate_ms'] for item in group)
        close(b, record['baseline_total_ms']); close(t, record['candidate_total_ms'])
        errors = [max(item[field] for item in group) for field in ['relative_rmse', 'scaled_max_abs']]
        close(errors[0], record['worst_relative_rmse']); close(errors[1], record['worst_scaled_max_abs'])
        quality &= max(errors) <= 0.001
        totals.append({'case': record['case'], 'seed': record['sampling_seed'], 'b': b, 't': t})
    case_stats = []
    for case in sorted(set(item['case'] for item in totals)):
        matches = [item for item in totals if item['case'] == case]
        if len(set(item['seed'] for item in matches)) != 1:
            raise ValueError('A case changes seed between repetitions')
        case_stats.append({'case': case, 'seed': matches[0]['seed'],
                           'b': statistics.median(item['b'] for item in matches),
                           't': statistics.median(item['t'] for item in matches)})
    seed_stats = []
    for seed in sorted(set(item['seed'] for item in case_stats)):
        matches = [item for item in case_stats if item['seed'] == seed]
        if result['is_formal_protocol'] and len(matches) != 2:
            raise ValueError('Formal protocol requires two cases per seed')
        seed_stats.append({'sampling_seed': seed, 'case_count': len(matches),
                           'baseline_ms': statistics.mean(item['b'] for item in matches),
                           'candidate_ms': statistics.mean(item['t'] for item in matches)})
    baseline = statistics.mean(item['baseline_ms'] for item in seed_stats)
    candidate = statistics.mean(item['candidate_ms'] for item in seed_stats)
    sd = statistics.stdev(item['baseline_ms'] for item in seed_stats)
    close(baseline, result['baseline_mean_ms']); close(candidate, result['candidate_mean_ms'])
    close(sd, result['baseline_sample_sd_ms'])
    if result['is_formal_protocol'] and (len(totals) != 18 or len(raw) != 4608 or len(seed_stats) != 3 or result['steps'] != 256):
        raise ValueError('Formal budget is incomplete')
    latent_checks = []
    for item in result['final_quality']:
        path = root / ('case-' + str(item['case']) + '-final-latents.safetensors')
        with safe_open(path, framework='pt', device='cpu') as handle:
            a = handle.get_tensor('candidate_final_latents').float()
            b = handle.get_tensor('baseline_final_latents').float()
        if not torch.isfinite(a).all() or not torch.isfinite(b).all():
            raise ValueError('Persisted autonomous final latents are nonfinite')
        difference = a - b
        relative = float(difference.square().mean().sqrt() / (b.square().mean().sqrt() + 1e-12))
        scaled = float(difference.abs().max() / (1 + b.abs().max()))
        close_float32_reduction(relative, item['relative_rmse'])
        close_float32_reduction(scaled, item['scaled_max_abs'])
        quality &= max(relative, scaled) <= 0.001
        latent_checks.append({'case': item['case'], 'relative_rmse': relative, 'scaled_max_abs': scaled,
                              'file_sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    quality &= result['peak_cuda_allocated_bytes'] <= 34 * 1024**3
    if any(item['relative_rmse'] != 0 or item['scaled_max_abs'] != 0 for item in result['source_alignment']):
        raise ValueError('Starter/published source alignment is not bit-exact')
    if quality != result['quality_pass']:
        raise ValueError('Independent quality result differs')
    reward = -1.0 if not quality else (baseline - candidate) / baseline
    close(reward, result['reward'])
    evidence = {'status': 'PASS', 'run_id': result['run_id'], 'is_formal_protocol': result['is_formal_protocol'],
                'raw_stages': len(raw), 'repetitions': len(totals), 'per_seed': seed_stats,
                'baseline_mean_ms': baseline, 'candidate_mean_ms': candidate, 'baseline_sample_sd_ms': sd,
                'all_repetition_baseline_sample_sd_ms': statistics.stdev(item['b'] for item in totals),
                'latency_improvement_ms': baseline - candidate, 'predeclared_noise_3sigma_pass': baseline - candidate >= 3 * sd,
                'additional_all_repetition_noise_3sigma_pass': baseline - candidate >= 3 * statistics.stdev(item['b'] for item in totals),
                'quality_pass': quality, 'reward': reward, 'latent_checks': latent_checks,
                'result_sha256': hashlib.sha256((root / 'result.json').read_bytes()).hexdigest(),
                'raw_stage_times_sha256': hashlib.sha256((root / 'raw-stage-times.json').read_bytes()).hexdigest(),
                'program_file_sha256': hashlib.sha256((root / 'program.json').read_bytes()).hexdigest(),
                'recompute_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'final_latent_metric_comparison': {'relative_tolerance': 1e-6, 'absolute_tolerance': 1e-8,
                    'quality_threshold': 0.001, 'threshold_decision_must_match': True,
                    'persisted_zero_requires_exact_CPU_zero': True},
                'cuda_context_initialized': torch.cuda.is_initialized(), 'is_independent_method_rerun': False,
                'formal_trajectory_effective_seconds': 0}
    assert not evidence['cuda_context_initialized']
    Path(args.output).write_text(json.dumps(evidence, indent=2) + '\n')
    print(json.dumps(evidence, indent=2))


if __name__ == '__main__':
    main()
